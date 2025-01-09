import random
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def L(x):
    return x

# MinMaxNormalization on [0,1]
def unit(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))  

    
def kde_pdf_cdf_function(data, factor):
    # Ensure the data is normalized to [0, 1]. Implementation of 'unit' is assumed but not defined.
    # Consider defining or importing the 'unit' function explicitly.
    uniform_data = unit(data)
    
    # Use Gaussian KDE to estimate the probability density function (PDF)
    kde = gaussian_kde(uniform_data)
    
    # Define the range for x values, assuming normalized data between 0 and 1
    x_vals = np.linspace(0, 1, 1000)
    
    # Compute PDF values
    pdf_vals = kde(x_vals)
    
    # Create an interpolation function for the PDF. Extrapolation is allowed beyond the defined range.
    pdf_func = interp1d(x_vals, pdf_vals, bounds_error=False, fill_value="extrapolate")
    
    # Compute the cumulative density function (CDF) by integrating the PDF
    cdf_vals = cumulative_trapezoid(pdf_vals, x_vals, initial=0)
    
    # Normalize the CDF so that its range is [0, 1]
    cdf_vals /= cdf_vals[-1]
    
    # Define a custom CDF function that handles out-of-range values with y = x logic.
    def custom_cdf_func(x):
        x = np.asarray(x)
        return np.where(
            x < 0, x, np.where(
                x > 1, x, 
                interp1d(x_vals, cdf_vals, bounds_error=False, fill_value="extrapolate")(x)
            )
        )
    
    # Define an extended function that modifies the CDF based on the factor.
    def extended_comp_func(x):
        x = np.asarray(x)  # Convert input to a numpy array for consistent handling
        return np.where(
            x < 0, x, np.where(
                x > 1, x, 
                custom_cdf_func(x) - (1 - factor) * (custom_cdf_func(x) - L(x))  # Note: 'L(x)' is assumed to be defined
            )
        )
    
    # Create an inverse function for 'extended_comp_func' using Brent's method
    def inverse_comp_func(y_vals, x_min=-np.inf, x_max=np.inf):
        def inverse_func(y):
            # Handle out-of-range cases: Return input directly when outside [0, 1]
            if y < 0:
                return y
            elif y > 1:
                return y
            # For values within [0, 1], solve using Brent's method
            else:
                return brentq(
                    lambda x: extended_comp_func(x) - y, 
                    max(x_min, 0), min(x_max, 1)
                )
        
        # Process scalar or array inputs for y_vals
        if np.isscalar(y_vals):  # Single value case
            return inverse_func(y_vals)
        else:  # Handle lists or arrays
            return np.array([inverse_func(y) for y in np.atleast_1d(y_vals)])

    return extended_comp_func, inverse_comp_func


def trajectory_matrix(time_series, K):
    """
    Create a trajectory matrix for Singular Spectrum Analysis (SSA).
    :param time_series: Input time series
    :param K: Window length (SSA hyperparameter)
    :return: Trajectory matrix of shape (K, N-K+1)
    """
    # Validate K to ensure it is less than the time series length
    assert K < len(time_series), "Window length K must be less than the length of the time series."
    
    N = len(time_series)
    L = N - K + 1  # Number of columns in the trajectory matrix
    X = np.zeros((K, L))

    # Populate the trajectory matrix with overlapping segments
    for i in range(L):
        X[:, i] = time_series[i:i + K]
    
    return X


def SSA_decomposition(time_series, K):
    """
    Decompose a time series using Singular Spectrum Analysis (SSA).
    :param time_series: Input time series
    :param K: Window length
    :return: List of decomposed components
    """
    # Step 1: Generate the trajectory matrix
    X = trajectory_matrix(time_series, K)
    
    # Step 2: Perform Singular Value Decomposition (SVD)
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Step 3: Reconstruct components from singular values
    components = []
    for i in range(K):
        # Reconstruct the i-th component using the outer product
        component_matrix = np.outer(U[:, i], Sigma[i] * Vt[i, :])
        
        # Convert the reconstructed component back to a time series
        component = np.zeros(len(time_series))
        count = np.zeros(len(time_series))
        
        for j in range(component_matrix.shape[1]):
            component[j:j + K] += component_matrix[:, j]
            count[j:j + K] += 1
        
        # Normalize by the number of overlapping elements
        component /= count
        components.append(component)
    
    return np.array(components)


def set_seed(seed):
    """
    Set the seed for reproducibility across libraries and frameworks.
    :param seed: Seed value
    """
    torch.manual_seed(seed)  # PyTorch random seed
    random.seed(seed)  # Python's random module seed
    np.random.seed(seed)  # NumPy random seed
    
    # If CUDA is available, set the seed for all GPU devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set CuDNN to deterministic mode for reproducibility (useful for CUDA operations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(data, labels, train_ratio=0.7, batch_size=32):
    """
    Prepare data for training and testing, with sequential splitting and DataLoader setup.
    :param data: Input data (array-like or tensor)
    :param labels: Corresponding labels (array-like or tensor)
    :param train_ratio: Proportion of data used for training (default=0.7)
    :param batch_size: Batch size for the DataLoader (default=32)
    :return: DataLoaders for train and test sets
    """
    # Convert to tensors if inputs are not already tensors
    if not torch.is_tensor(data):
        data_tensor = torch.tensor(data, dtype=torch.float32).clone().detach()
    else:
        data_tensor = data.clone().detach()

    if not torch.is_tensor(labels):
        labels_tensor = torch.tensor(labels, dtype=torch.float32).clone().detach()
    else:
        labels_tensor = labels.clone().detach()

    # Split the dataset into training and testing sets
    dataset_size = len(data_tensor)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Sequential splitting to maintain time-series order
    train_data = data_tensor[:train_size]
    train_labels = labels_tensor[:train_size]
    test_data = data_tensor[train_size:]
    test_labels = labels_tensor[train_size:]

    # Create TensorDataset instances
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create DataLoaders with `drop_last=True` to discard incomplete batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


def get_total_label_length(test_loader):
    """
    Compute the total number of labels across all batches in the DataLoader.
    :param test_loader: DataLoader for test data
    :return: Total number of labels
    """
    total_length = 0
    for _, labels in test_loader:
        total_length += labels.size(0)  # Add the number of labels in the current batch
    return total_length


class TCNBlock(nn.Module):
    """
    Define a single Temporal Convolutional Network (TCN) block.
    """
    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        """
        :param input_channels: Number of input channels
        :param output_channels: Number of output channels
        :param kernel_size: Size of the convolution kernel
        :param dilation: Dilation factor for the convolution
        """
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,  # Padding to maintain sequence length
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        """
        Forward pass through the TCN block.
        :param x: Input tensor of shape (batch_size, channels, seq_len)
        :return: Output tensor after convolution, batch normalization, and ReLU
        """
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]  # Remove extra padding from the front
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class SelfAttention(nn.Module):
    """
    Define a self-attention layer.
    """
    def __init__(self, input_dim):
        """
        :param input_dim: Dimensionality of the input
        """
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Compute self-attention.
        :param x: Input tensor of shape (batch_size, seq_len, input_dim)
        :return: Output tensor after attention mechanism
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        d_k = Q.size(-1)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attn_weights = self.softmax(attn_scores)

        # Compute weighted sum of values
        attn_output = torch.bmm(attn_weights, V)
        return attn_output


class TCNWithAttention(nn.Module):
    """
    Combine TCN layers with a self-attention mechanism for enhanced sequence modeling.
    """
    def __init__(self, input_size, num_channels, kernel_size, dilations):
        """
        :param input_size: Dimensionality of the input
        :param num_channels: List of output channels for each TCN layer
        :param kernel_size: Size of the convolution kernel
        :param dilations: List of dilation factors for each TCN layer
        """
        super(TCNWithAttention, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = dilations[i]

            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))

        self.network = nn.Sequential(*layers)
        self.attention = SelfAttention(num_channels[-1])
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        """
        Forward pass for the model.
        :param x: Input tensor of shape (batch_size, seq_len, input_dim)
        :return: Model output of shape (batch_size, 1)
        """
        x = x.permute(0, 2, 1)  # Convert to (batch_size, channels, seq_len)
        x = self.network(x)
        x = x.permute(0, 2, 1)  # Convert back to (batch_size, seq_len, channels)
        x = self.attention(x)

        # Select the last timestep for prediction
        x = x[:, -1, :]  # (batch_size, channels)

        # Linear layer for final prediction
        x = self.linear(x)  # (batch_size, 1)
        return x


def pred_generation(original_data, predictions, factor, total_label_length):
    """
    Generate real-world predictions by applying inverse CDF transformation.
    :param original_data: Original time series data
    :param predictions: Predicted values from the model
    :param factor: Factor controlling the CDF transformation
    :param total_label_length: Total number of labels in the test data
    :return: Transformed real-world predictions
    """
    Num = total_label_length  # Number of test samples
    len_cut = 24  # Length of the input window
    X_val = [original_data[i:i + len_cut + 1] for i in range(len(original_data) - len_cut)][-Num:]
    Y_val = original_data[len_cut:][-Num:]  # Ground truth for validation

    real_pred = []
    for i in range(Num):
        # Normalize the input window
        M, N = np.max(X_val[i]), np.min(X_val[i])
        input_data = (X_val[i] - N) / (M - N)
        
        # Generate PDF and CDF functions
        f1, f2 = kde_pdf_cdf_function(input_data, factor)
        
        # Apply inverse CDF transformation
        real_pred.append(f2(predictions[i]) * (M - N) + N)
    
    return np.array(real_pred)


Adaptive Distribution Transformation Approach for
 Cryptocurrency Price Prediction
 
 Abstract—Cryptocurrency price prediction is challenging due
 to the high volatility of its time-series data, which fluctuates
 significantly even over short periods. While the sliding window
 strategy preserves temporal dependencies and improves data
 usage efficiency, short-term windows remain vulnerable to ex
treme volatility, limiting prediction performance. To address this
 issue, we propose sliding frequency suppression (SFS), a data
 preprocessing technique that adjusts the cumulative distribu
tion function (CDF) within each window to mitigate extreme
 distributions concentrated around specific price ranges. The SFS
 approach is integrated with singular spectrum analysis (SSA) and
 temporal convolutional networks (TCN). Additionally, we apply
 a self-attention mechanism at the final TCN layer to highlight
 essential information. We assess the model’s performance across
 various hyperparameter settings within SSA and SFS to evaluate
 their impact. Experimental results show that the proposed
 strategy improves RMSE by 16.8% across all price domains and
 15.7% within the P > 95 price domain compared to vanilla
 model. Moreover, the optimal hyperparameter values span a
 broad range within the feasible parameter space, demonstrating
 that the strategy is robust and delivers consistent performance
 gains without extensive fine-tuning.

 Python version: 3.12.4
 
 Pytorch version: 2.4.0

import pandas as pd
import model
import backtest
import permutation
import monte_carlo
import json
import numpy as np

# Main script
train_start = '2021-01-01 00:00:00'
train_end = '2024-01-01 00:00:00'
test_start = '2024-01-01 01:00:00'
test_end = '2025-01-01 00:00:00'

window = 50
thres1 = 3
thres2 = -3
models = model.zscore #model input
forward_test = 'no'  # Set to 'yes' to perform forward test, 'no' to run original code
mode = 'monte_carlo' # monte_carlo, permutation 

window_range = range(40, 60, 1)  # Window from 10 to 100, step 10
thres1_range = np.round(np.arange(2.5, 3.5, 0.1),3)  # thres1 from 0.1 to 1.0, step 0.1

# read data
prices = pd.read_csv(r'/Users/chunhen/Documents/Quant Project/candle/BTCUSDT_1h_candle_binance.csv')
prices['datetime'] = pd.to_datetime(prices['datetime'])
prices = prices.set_index('datetime')
prices = prices.ffill()  # Remove any NaN or inf values after log transformation
train_price = prices[train_start:train_end]
test_price = prices[test_start:test_end]

if __name__ == '__main__':
    # Calculate metrics for train data
    train_df, train_metrics = backtest.calculate_metrics(train_price, window, thres1, thres2)
    print("Train Metrics:")
    print(json.dumps(train_metrics, indent=4))
    train_df.to_csv('bt_train.csv')

    if forward_test == 'yes':
        # Calculate metrics for test data
        test_df, test_metrics = backtest.calculate_metrics(test_price, window, thres1, thres2)
        print("\nTest Metrics:")
        print(json.dumps(test_metrics, indent=4))
        test_df.to_csv('bt_test.csv')
        backtest.plot_equity_curve(train_df, test_df, forward_test)
    else:
        backtest.plot_equity_curve(train_df)

    if mode == 'permutation':
        best_params, results_df = permutation.optimize_parameters(train_price, window_range, thres1_range)
        print("Best Parameters for Maximum Sharpe Ratio:")
        print(json.dumps(best_params.to_dict(), indent=4))

        # Save results to CSV
        permutation.plot_heatmap(results_df)

    elif mode == 'monte_carlo':
        metrics_df, equity_curves = monte_carlo.monte_carlo_simulation(train_price, window, thres1, thres2, num_simulations=1000, time_horizon=365*24)

        # Save metrics to CSV
        metrics_df.to_csv('monte_carlo_metrics.csv')

        # Print summary statistics
        print("\nMonte Carlo Simulation Summary Statistics:")
        print(metrics_df.describe())

        monte_carlo.check_mdd_percentiles(metrics_df)

        # Plot distributions of metrics
        monte_carlo.plot_metrics_distribution(metrics_df)

        # Plot sample equity curves
        monte_carlo.plot_sample_equity_curves(equity_curves, num_samples=10)

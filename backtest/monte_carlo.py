import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import backtest

# Monte Carlo Simulation function
def monte_carlo_simulation(historical_prices, window, thres1, thres2, num_simulations=1000, time_horizon=24*365):
    # Calculate historical returns and their statistics
    returns = historical_prices['close'].pct_change().dropna()
    mu = returns.mean()  # Mean of returns
    sigma = returns.std()  # Standard deviation of returns

    # If time_horizon is not specified, use the length of historical data
    if time_horizon is None:
        time_horizon = len(historical_prices)

    # Initialize lists to store metrics from each simulation
    all_metrics = []
    all_equity_curves = []

    # Generate simulated price paths
    for _ in tqdm(range(num_simulations), desc="Running Monte Carlo Simulations"):
        # Generate random returns based on historical mean and std
        simulated_returns = np.random.normal(mu, sigma, time_horizon)
        
        # Create simulated price path starting from the last historical close
        initial_price = historical_prices['close'].iloc[-1]
        simulated_prices = [initial_price]
        for ret in simulated_returns:
            simulated_prices.append(simulated_prices[-1] * (1 + ret))
        
        # Create DataFrame for simulated prices
        sim_df = pd.DataFrame({
            'close': simulated_prices
        }, index=pd.date_range(start=historical_prices.index[-1], periods=time_horizon + 1, freq='h'))

        # Run the trading strategy on the simulated prices
        sim_df, metrics = backtest.calculate_metrics(sim_df, window, thres1, thres2)

        all_metrics.append(metrics)
        all_equity_curves.append(sim_df['cumu_pnl'])

    # Convert metrics to DataFrame for analysis
    metrics_df = pd.DataFrame(all_metrics)

    return metrics_df, all_equity_curves

# Plot distribution of key metrics
def plot_metrics_distribution(metrics_df):
    metrics_to_plot = ['Sharpe Ratio', 'Cumu Returns', 'MDD', 'CR']
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, i)
        plt.hist(metrics_df[metric].dropna(), bins=50, color='blue', alpha=0.7)
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('metrics_distribution.png')
    plt.show()

# Plot a sample of equity curves
def plot_sample_equity_curves(equity_curves, num_samples=10):
    plt.figure(figsize=(10, 6))
    for i in range(min(num_samples, len(equity_curves))):
        plt.plot(equity_curves[i].index, equity_curves[i], label=f'Simulation {i+1}', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.title('Monte Carlo Equity Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('monte_carlo.png')
    plt.show()


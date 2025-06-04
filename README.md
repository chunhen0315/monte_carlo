# Monte Carlo

This is a backtesting project using Monte Carlo Simulation to assess stability and robustness of trading strategy. 


1. main.py:
This is the main code to perform action included data preprocessing, backtest, forward test, permutaiton and monte carlo simulation. 

2. backtesy.py:
This is the backtest function for position and metrics calculation, such as Sharpe Ratio, Max Drawdown, Annualized Return and Calmar Ratio. 
It also allow to perform train-test split (forward test) when forward_test = 'yes' in main.py

3. model.py:
This is all the model calculation such as Z-score, Minmax Scaling, Robust Scaling and more. 

4. monte_carlo.py:
This is monte carlo function to help examine the risk and profitability of trading strategy by randomized the price data. 

5. permuation.py:
This is permutation to help optimized parameters for our trading strategy.

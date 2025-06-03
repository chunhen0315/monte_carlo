# monte_carlo

This is a trading backtesting project include several procedure to assess stability and robustness of trading strategy. 

main.py:
This is the main code to perform all action included backtest, forward test, permutaiton and monte carlo simulation. 

backtesy.py:
this is the backtest function for position and metrics calculation, such as Sharpe Ratio, Max Drawdown, Annualized Return and Calmar Ratio. 
It also allow to perform train-test split (forward test) when forward_test = 'yes' in main.py

model.py:
This is all the model calculation behind. 

monte_carlo.py:
This is monte carlo simulation to help estimate the risk and profitability of trading strategy by randomized price data. 

permuation.py:
This is permutation to help optimized parameters for our trading strategy.

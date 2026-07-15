# --- OOP REFACTOR: THE QUANTITATIVE ENGINE ---
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t

class QuantitativeEngine:
    def __init__(self, tickers, start_date, end_date):
        """Initializes the engine and automatically fetches the market DNA."""
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        print(f"Initializing Quantitative Engine for {len(tickers)} assets...")
        self.raw_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        self.daily_returns = self.raw_data.pct_change().dropna()
        self.mean_returns = self.daily_returns.mean()
        self.cov_matrix = self.daily_returns.cov()
        print("Market DNA successfully extracted.\n")

    def optimize_portfolio(self, num_portfolios=5000, risk_free_rate=0.04):
        """Runs the Markowitz Efficient Frontier to find the optimal Sharpe Ratio."""
        print(f"Running {num_portfolios} MPT simulations...")
        all_weights = np.zeros((num_portfolios, len(self.tickers)))
        ret_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)

        # To ensure reproducible results
        np.random.seed(4)

        for x in range(num_portfolios):
            weights = np.array(np.random.random(len(self.tickers)))
            weights = weights / np.sum(weights)
            all_weights[x, :] = weights

            ret_arr[x] = np.sum((self.mean_returns * weights) * 252)
            vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
            sharpe_arr[x] = (ret_arr[x] - risk_free_rate) / vol_arr[x]

        max_sharpe_idx = sharpe_arr.argmax()
        self.optimal_weights = all_weights[max_sharpe_idx, :]

        self.expected_return = ret_arr[max_sharpe_idx]
        self.volatility = vol_arr[max_sharpe_idx]

        print("--- Optimization Complete ---")
        print(f"Expected Return: {ret_arr[max_sharpe_idx]:.2%}")
        print(f"Volatility: {vol_arr[max_sharpe_idx]:.2%}")
        return self.optimal_weights

    def run_monte_carlo(self, weights, time_horizon=30, initial_investment=10000, num_simulations=10000):
        """Stress tests a specific set of weights over a set time horizon."""
        print(f"\nRunning {num_simulations} Monte Carlo simulations over {time_horizon} days...")
        simulation_results = np.zeros(num_simulations)

        np.random.seed(4)

        df = 10
        for i in range(num_simulations):
            simulated_returns = multivariate_t.rvs(
                loc=self.mean_returns.values,
                shape=self.cov_matrix.values,
                df=df,
                size=time_horizon
            )
            portfolio_daily_ret = np.dot(simulated_returns, weights)
            final_value = initial_investment * np.prod(1 + portfolio_daily_ret)
            simulation_results[i] = final_value

        percentile_5 = np.percentile(simulation_results, 5)
        var_95 = initial_investment - percentile_5

        print(f"95% Value at Risk (VaR): ${var_95:.2f}")
        return simulation_results, var_95
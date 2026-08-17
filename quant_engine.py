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
        max_weight=0.40

        for i in range(num_portfolios):
            valid_weights = False
            while not valid_weights:            # for max weight optimization
                weights = np.random.random(len(self.tickers))
                weights /= np.sum(weights)
                
                if not np.any(weights > max_weight):
                    valid_weights = True 
            
            all_weights[i, :] = weights
            ret_arr[i] = np.sum(self.mean_returns * weights) * 252
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
            if vol_arr[i] > 0:
                sharpe_arr[i] = (ret_arr[i] - risk_free_rate) / vol_arr[i]
            else:
                sharpe_arr[i] = -np.inf

        max_sharpe_idx = sharpe_arr.argmax()
        self.optimal_weights = all_weights[max_sharpe_idx, :]

        self.expected_return = ret_arr[max_sharpe_idx]
        self.volatility = vol_arr[max_sharpe_idx]

        print("--- Optimization Complete ---")
        print(f"Expected Return: {ret_arr[max_sharpe_idx]:.2%}")
        print(f"Volatility: {vol_arr[max_sharpe_idx]:.2%}")
        return self.optimal_weights
    # Added initial_investment to the parameters to match your run_monte_carlo method
    def walk_forward_optimize(self, initial_investment=10000, train_window_size=252, test_window_size=63, num_portfolios=5000, transaction_fee_pct=0.0):
        """
        Performs Walk-Forward Optimization with transaction costs.
        
        :param initial_investment: The starting capital passed from the website frontend.
        :param train_window_size: Trading days for training (e.g., 252 for 1 year).
        :param test_window_size: Trading days for testing (e.g., 63 for 1 quarter).
        :param num_portfolios: Number of MPT simulations per training fold.
        :param transaction_fee_pct: Brokerage fee + slippage per trade (0.001 = 0.1%).
        """
        print(f"\n--- Starting Walk-Forward Optimization ---")
        total_days = len(self.daily_returns)
        
        if total_days < (train_window_size + test_window_size):
            raise ValueError("Not enough historical data for the specified window sizes.")

        # Dynamically use the passed variable instead of a hardcoded 10000.0
        portfolio_values = [float(initial_investment)] 
        dates = [self.daily_returns.index[train_window_size - 1]]
        
        current_idx = 0
        previous_weights = np.zeros(len(self.tickers)) # Starts as all cash

        weights_history = []
        rebalance_dates = []
        total_fees_paid = 0.0
        
        while current_idx + train_window_size + test_window_size <= total_days:
            # 1. Slice the Data
            train_data = self.daily_returns.iloc[current_idx : current_idx + train_window_size]
            test_data = self.daily_returns.iloc[current_idx + train_window_size : current_idx + train_window_size + test_window_size]
            
            # 2. Extract In-Sample Market DNA
            fold_mean_returns = train_data.mean()
            fold_cov_matrix = train_data.cov()
            
            # 3. Optimize Weights (MPT Simulation)
            all_weights = np.zeros((num_portfolios, len(self.tickers)))
            sharpe_arr = np.zeros(num_portfolios)
            np.random.seed(4)
            max_weight = 0.40
            
            for i in range(num_portfolios):
                valid_weights = False
                while not valid_weights:
                    weights = np.random.random(len(self.tickers))
                    weights /= np.sum(weights)
                    if not np.any(weights > max_weight):
                        valid_weights = True
                
                all_weights[i, :] = weights
                ret = np.sum(fold_mean_returns * weights) * 252
                vol = np.sqrt(np.dot(weights.T, np.dot(fold_cov_matrix, weights))) * np.sqrt(252)
                sharpe_arr[i] = ret / vol if vol != 0 else 0
                
            optimal_weights = all_weights[sharpe_arr.argmax(), :]
            
            # 4. Calculate Transaction Costs (Portfolio Turnover)
            turnover = np.sum(np.abs(optimal_weights - previous_weights))
            cost_in_dollars = portfolio_values[-1] * turnover * transaction_fee_pct

            total_fees_paid += cost_in_dollars
            weights_history.append(optimal_weights)
            rebalance_dates.append(self.daily_returns.index[current_idx + train_window_size])
            
            # Deduct the cost from the starting value of this test period
            current_portfolio_value = portfolio_values[-1] - cost_in_dollars
            
            # 5. Out-of-Sample Forward Testing
            test_returns = (test_data * optimal_weights).sum(axis=1)
            
            for date, daily_ret in test_returns.items():
                current_portfolio_value = current_portfolio_value * (1 + daily_ret)
                portfolio_values.append(current_portfolio_value)
                dates.append(date)
                
            # 6. Roll Forward Setup
            previous_weights = optimal_weights 
            current_idx += test_window_size

        wf_equity_curve = pd.DataFrame({'Portfolio_Value': portfolio_values}, index=dates)
        
        # Create a DataFrame for the weights evolution
        weights_df = pd.DataFrame(weights_history, index=rebalance_dates, columns=self.tickers)
        
        print("--- Walk-Forward Optimization Complete ---")
        
        # Return a dictionary containing everything the Streamlit frontend needs
        return {
            "equity_curve": wf_equity_curve,
            "weights_history": weights_df,
            "total_fees_paid": total_fees_paid
        }

    def run_monte_carlo(self, weights, time_horizon=30, initial_investment=10000, num_simulations=10000):
        """Stress tests a specific set of weights over a set time horizon."""
        print(f"\nRunning {num_simulations} Monte Carlo simulations over {time_horizon} days...")
        simulation_results = np.zeros(num_simulations)

        np.random.seed(4)

        df = 6
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
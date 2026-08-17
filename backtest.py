import numpy as np
import pandas as pd
from scipy.stats import t, chi2

class Backtester:
    def __init__(self, engine):
        """
        Initializes the Backtester with a QuantitativeEngine instance.
        """
        self.engine = engine

    def walk_forward_optimize(self, initial_investment=10000, train_window_size=252, test_window_size=63, num_portfolios=5000, transaction_fee_pct=0.0):
        """
        Performs Walk-Forward Optimization with transaction costs.
        """
        print(f"\n--- Starting Walk-Forward Optimization ---")
        total_days = len(self.engine.daily_returns)
        
        if total_days < (train_window_size + test_window_size):
            raise ValueError("Not enough historical data for the specified window sizes.")

        portfolio_values = [float(initial_investment)] 
        dates = [self.engine.daily_returns.index[train_window_size - 1]]
        
        current_idx = 0
        previous_weights = np.zeros(len(self.engine.tickers)) 

        weights_history = []
        rebalance_dates = []
        total_fees_paid = 0.0
        
        all_test_returns = []
        all_predicted_vars = []
        
        while current_idx + train_window_size + test_window_size <= total_days:
            # 1. Slice the Data
            train_data = self.engine.daily_returns.iloc[current_idx : current_idx + train_window_size]
            test_data = self.engine.daily_returns.iloc[current_idx + train_window_size : current_idx + train_window_size + test_window_size]
            
            # 2. Extract In-Sample Market DNA
            fold_mean_returns = train_data.mean()
            fold_cov_matrix = train_data.cov()
            
            # 3. Optimize Weights (MPT Simulation)
            all_weights = np.zeros((num_portfolios, len(self.engine.tickers)))
            sharpe_arr = np.zeros(num_portfolios)
            np.random.seed(4)
            max_weight = 0.40
            
            for i in range(num_portfolios):
                valid_weights = False
                while not valid_weights:
                    weights = np.random.random(len(self.engine.tickers))
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
            rebalance_dates.append(self.engine.daily_returns.index[current_idx + train_window_size])
            
            # Deduct the cost from the starting value of this test period
            current_portfolio_value = portfolio_values[-1] - cost_in_dollars
            
            # 5. Out-of-Sample Forward Testing
            test_returns = (test_data * optimal_weights).sum(axis=1)
            
            # Predict 1-day VaR (95%) for the test period using t-distribution (df=6) to match Monte Carlo
            port_mean = np.sum(fold_mean_returns * optimal_weights)
            port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(fold_cov_matrix, optimal_weights)))
            daily_var_95 = port_mean + t.ppf(0.05, df=6) * port_vol
            
            all_test_returns.extend(test_returns.values)
            all_predicted_vars.extend([daily_var_95] * len(test_returns))
            
            for date, daily_ret in test_returns.items():
                current_portfolio_value = current_portfolio_value * (1 + daily_ret)
                portfolio_values.append(current_portfolio_value)
                dates.append(date)
                
            # 6. Roll Forward Setup
            previous_weights = optimal_weights 
            current_idx += test_window_size

        wf_equity_curve = pd.DataFrame({'Portfolio_Value': portfolio_values}, index=dates)
        
        weights_df = pd.DataFrame(weights_history, index=rebalance_dates, columns=self.engine.tickers)
        
        print("--- Walk-Forward Optimization Complete ---")
        
        kupiec_results = self.kupiec_pof_test(
            actual_returns=np.array(all_test_returns),
            predicted_vars=np.array(all_predicted_vars)
        )
        
        return {
            "equity_curve": wf_equity_curve,
            "weights_history": weights_df,
            "total_fees_paid": total_fees_paid,
            "kupiec_results": kupiec_results,
            "test_returns": all_test_returns,
            "predicted_vars": all_predicted_vars
        }

    def kupiec_pof_test(self, actual_returns, predicted_vars, confidence_level=0.95):
        """
        Kupiec's Proportion of Failures (POF) test.
        """
        print(f"\n--- Running Kupiec's POF Test ({confidence_level:.0%} VaR) ---")
        breaches = actual_returns < predicted_vars
        x = breaches.sum()
        N = len(actual_returns)
        p = 1 - confidence_level
        
        observed_p = x / N if N > 0 else 0
        
        print(f"Total Periods: {N}")
        print(f"Breaches (Exceptions): {x}")
        print(f"Expected Breach Rate: {p:.2%}")
        print(f"Observed Breach Rate: {observed_p:.2%}")
        
        if x == 0:
            LR = -2 * (N * np.log(1 - p))
        elif x == N:
            LR = -2 * (N * np.log(p))
        else:
            LR = -2 * ((N - x) * np.log(1 - p) + x * np.log(p) - 
                       ((N - x) * np.log(1 - observed_p) + x * np.log(observed_p)))
                       
        p_value = 1 - chi2.cdf(LR, 1)
        is_calibrated = p_value > 0.05
        
        print(f"Kupiec POF Test Statistic: {LR:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Model Well-Calibrated (Fail to Reject H0)? {is_calibrated}\n")
        
        return {
            'breaches': int(x),
            'total_periods': N,
            'expected_rate': p,
            'observed_rate': observed_p,
            'test_statistic': LR,
            'p_value': p_value,
            'is_calibrated': is_calibrated
        }

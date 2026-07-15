import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quant_engine import QuantitativeEngine 

# --- 1. UI Setup & Header ---
st.set_page_config(page_title="Quant Portfolio Optimizer", layout="wide")
st.title("📈 Algorithmic Portfolio Optimizer & Risk Engine")
st.markdown("Enter your preferred stock tickers to mathematically calculate the optimal Sharpe Ratio weights and stress-test the portfolio.")

# --- 2. Dynamic User Inputs (Sidebar) ---
st.sidebar.header("Engine Parameters")

with st.sidebar.expander("📚 Common Ticker Reference"):
    st.markdown("""
    **Technology**
    * **AAPL:** Apple
    * **MSFT:** Microsoft
    * **GOOGL:** Google
    * **NVDA:** Nvidia
    * **META:** Meta (Facebook)
    
    **Finance & Banking**
    * **JPM:** JPMorgan Chase
    * **BAC:** Bank of America
    * **V:** Visa
    
    **Energy & Defense**
    * **XOM:** ExxonMobil
    * **CVX:** Chevron
    * **LMT:** Lockheed Martin
    
    **Consumer & Retail**
    * **AMZN:** Amazon
    * **WMT:** Walmart
    * **KO:** Coca-Cola
    """)

ticker_input = st.sidebar.text_input("Stock Tickers (comma-separated)", "AAPL, JPM, LMT, XOM")
tickers = [t.strip().upper() for t in ticker_input.split(',')]

# --- NEW: Train / Test Split Dates ---
st.sidebar.subheader("Backtesting Timeline")
start_date = st.sidebar.date_input("Training Start Date", value=pd.to_datetime('2023-01-01'))
split_date = st.sidebar.date_input("Train/Test Split Date (Go Live)", value=pd.to_datetime('2026-01-01'))
end_date = st.sidebar.date_input("Testing End Date", value=pd.to_datetime('2026-07-01'))

investment = st.sidebar.number_input("Initial Investment ($)", value=10000, step=1000)

# --- 3. Execution ---
if st.sidebar.button("Run Quantitative Optimization"):
    
    # Create our two clean navigation tabs
    tab1, tab2 = st.tabs(["📊 Backtesting & Performance", "🔮 30-Day Forward Prediction"])
    
    # ==========================================
    # TAB 1: THE PAST (Historical Backtesting)
    # ==========================================
    with tab1:
        with st.spinner(f'Training Algorithm on Past Data ({start_date} to {split_date})...'):
            
            # PHASE 1: THE PAST (TRAINING)
            train_engine = QuantitativeEngine(tickers=tickers, start_date=start_date, end_date=split_date)
            optimal_weights = train_engine.optimize_portfolio(num_portfolios=5000)
            
            st.subheader("Target Asset Allocation (Trained on Past Data)")
            cols = st.columns(len(tickers))
            for i, col in enumerate(cols):
                col.metric(label=tickers[i], value=f"{optimal_weights[i]:.2%}")
                
            st.divider()
                
        with st.spinner(f'Forward Testing on Unseen Data ({split_date} to {end_date})...'):
            
            # PHASE 2: THE FUTURE (TESTING)
            st.subheader("Out-of-Sample Performance vs. Equal Weight Benchmark")
            
            test_engine = QuantitativeEngine(tickers=tickers, start_date=split_date, end_date=end_date)
            test_returns = test_engine.daily_returns
            
            # Algorithm uses locked weights from Phase 1
            algo_daily_returns = np.dot(test_returns, optimal_weights)
            algo_cumulative = investment * np.cumprod(1 + algo_daily_returns)
            
            # Dynamic Benchmark Performance
            even_split = 1.0 / len(tickers)
            benchmark_weights = np.array([even_split] * len(tickers))
            benchmark_daily_returns = np.dot(test_returns, benchmark_weights)
            benchmark_cumulative = investment * np.cumprod(1 + benchmark_daily_returns)
            
            # Plotting the Comparison
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(test_returns.index, algo_cumulative, label='Optimized Strategy', color='forestgreen', linewidth=2)
            ax.plot(test_returns.index, benchmark_cumulative, label=f'Equal Weight Benchmark ({even_split:.1%})', color='gray', linestyle='dashed', linewidth=2)
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # --- Advanced Performance Metrics ---
            st.divider()
            st.subheader("Expected vs. Actual Performance Breakdown")
            
            actual_return = (algo_cumulative[-1] - investment) / investment
            benchmark_return = (benchmark_cumulative[-1] - investment) / investment
            
            m1, m2, m3, m4 = st.columns(4)
            
            m1.metric("MPT Expected Return", f"{train_engine.expected_return:.2%}" if hasattr(train_engine, 'expected_return') else "N/A")
            m1.caption("Based on Training Data")
            
            m2.metric("MPT Expected Volatility", f"{train_engine.volatility:.2%}" if hasattr(train_engine, 'volatility') else "N/A")
            m2.caption("Based on Training Data")
            
            m3.metric("Actual Optimized Return", f"{actual_return:.2%}", delta=f"{(actual_return - benchmark_return):.2%} vs Benchmark")
            m3.caption("Based on Out-of-Sample Data")
            
            m4.metric("Actual Benchmark Return", f"{benchmark_return:.2%}")
            m4.caption("Based on Out-of-Sample Data")

            # --- Final Dollar Amounts ---
            st.divider()
            st.subheader("Final Portfolio Value (Cash)")
            
            val1, val2 = st.columns(2)
            val1.metric(label="Optimized Strategy Final Value", value=f"${algo_cumulative[-1]:,.2f}")
            val2.metric(label="Benchmark Final Value", value=f"${benchmark_cumulative[-1]:,.2f}")

            # --- Disclaimers ---
            st.divider()
            with st.expander("⚙️How this Engine Works"):
                st.markdown("""
                This application is powered by a custom Python backend (`quant_engine.py`) that executes the following algorithmic sequence:
                1. **Data Ingestion:** Fetches historical daily adjusted close prices via the `yfinance` API.
                2. **Monte Carlo Simulation (Training):** Generates thousands of random portfolio weight arrays using `NumPy`. It utilizes Rejection Sampling to discard any arrays that violate the maximum asset concentration limit.
                3. **Mathematical Optimization:** Calculates the annualized expected return and covariance matrix to find the exact weight distribution that maximizes the **Sharpe Ratio** (Return / Risk).
                4. **Forward Validation (Testing):** Locks in the optimized weights and matrix-multiplies them against unseen, out-of-sample data, dynamically tracking cumulative returns against a baseline 1/N equal-weight benchmark.
                """)

            with st.expander("📚 Why does the Optimized Strategy sometimes lose to the Benchmark?"):
                st.markdown("""
                **The Out-of-Sample Reality (Estimation Error)**
                In quantitative finance, it is incredibly common for a mathematically "perfect" optimized portfolio to underperform a naive equal-weight benchmark in out-of-sample testing. This happens due to two primary factors:
                * **Estimation Error:** Mean-variance optimization requires estimating future returns and volatility based purely on historical data. The optimizer acts as an "error maximizer," heavily weighting assets that had a lucky historical run and heavily penalizing assets that had a temporary dip. 
                * **The Robustness of Equal Weighting:** The equal-weight benchmark (1/N) makes absolutely zero assumptions about the future. Because it does not rely on historical data, it suffers from zero estimation error, making it statistically highly robust to sudden market regime shifts.
                **The Takeaway:** If an algorithm crushes the market during the training phase but lags behind the benchmark during the testing phase, it proves the model *overfit* to the past rather than learning a persistent future pattern.
                """)

    # ==========================================
    # TAB 2: THE FUTURE (30-Day Stress Test)
    # ==========================================
    with tab2:
        st.header("30-Day Monte Carlo Stress Test")
        st.markdown("This engine simulates **10,000 independent future market trajectories** over the next 30 days. It uses a **Multivariate Student's t-distribution** to accurately model 'fat-tail' risk and extreme market crashes.")
        
        with st.spinner('Simulating 10,000 random market walks (Fat-Tail Distribution)...'):
            
            # Using your existing backend function, starting from the final cash amount of Phase 2!
            sim_results, var_95_dollars = test_engine.run_monte_carlo(
                weights=optimal_weights, 
                time_horizon=30, 
                initial_investment=algo_cumulative[-1],
                num_simulations=10000
            )
            
            # --- Plotting the Simulation Distribution (Cash Value) ---
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            counts, bins, patches = ax2.hist(sim_results, bins=60, alpha=0.6, color='darkseagreen', edgecolor='white')
            
            # The 5th percentile cutoff in cash
            cutoff = np.percentile(sim_results, 5)
            
            # Color the worst 5% of outcomes (downside risk zone) in deep red
            for patch, left, right in zip(patches, bins[:-1], bins[1:]):
                if left < cutoff:
                    patch.set_facecolor('indianred')
            
            ax2.axvline(cutoff, color='red', linestyle='dashed', linewidth=2, label=f'95% VaR Threshold (${cutoff:,.2f})')
            ax2.axvline(algo_cumulative[-1], color='black', linestyle='dotted', linewidth=1, label=f'Starting Investment Value (${algo_cumulative[-1]:,.2f})')
            ax2.set_xlabel('Simulated Final Portfolio Value ($)')
            ax2.set_ylabel('Number of Scenarios')
            ax2.legend()
            st.pyplot(fig2)
            
            # --- Metrics Dashboard ---
            st.divider()
            v1, v2, v3 = st.columns(3)
            
            expected_final_value = np.mean(sim_results)
            prob_positive = np.mean(sim_results > algo_cumulative[-1])
            
            v1.metric("Average Simulated Value", f"${expected_final_value:,.2f}", f"${(expected_final_value - algo_cumulative[-1]):,.2f} Expected Profit")
            v2.metric("Probability of Profit", f"{prob_positive:.1%}", help="Percentage of simulated paths that finished above your starting value.")
            v3.metric("30-Day Value at Risk (VaR)", f"${var_95_dollars:,.2f}", delta="Maximum Expected Loss", delta_color="inverse")
            # --- NEW: Monte Carlo Graph Explanation ---
            st.divider()
            with st.expander("🔬 Understanding the Stress Test & VaR Graph"):
                st.markdown("""
                ### Decoding the 30-Day Future Projection
                This distribution graph visualizes a **Monte Carlo Simulation**, plotting 10,000 statistically probable future realities for your newly optimized portfolio.
                
                * **Value at Risk (VaR):** The red dashed line represents the **95% Value at Risk**. Statistically, this represents our confidence threshold. We are 95% confident that the portfolio's value will *not* drop below this exact dollar amount over the next 30 days. 
                * **Graph Distribution:** The distribution of future outcomes is represented by the histogram. We assumed a fat-tail distribution, which means the tail of the distribution is especially volatile. (instead f a normal distribution, the tail would be flat)
                * **Limitations of this simulation:** The simulation is **not guaranteed** to accurately predict the market's future behavior. One of the reason is that this assumes that the probability of getting a high return is same as the probability of stock crashing. In practice, this is **not really the case**, and the simulation is **not perfect**. 
                """)
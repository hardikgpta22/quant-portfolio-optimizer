import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from quant_engine import QuantitativeEngine 
from backtest import Backtester
import matplotlib.dates as mdates

# --- 1. UI Setup & Header ---
st.set_page_config(page_title="Quant Portfolio Optimizer", layout="wide")
st.title("📈 Algorithmic Portfolio Optimizer & Risk Engine")
st.markdown("Enter your preferred stock tickers to mathematically calculate the optimal Sharpe Ratio weights and stress-test the portfolio.")

# --- 2. Dynamic User Inputs (Sidebar) ---
st.sidebar.header("Engine Parameters")

with st.sidebar.expander("Common Ticker Reference"):
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
    
    # Create our three clean navigation tabs
    tab1, tab2, tab3 = st.tabs(["Backtesting & Performance", "30-Day Forward Prediction", "Walk-Forward Optimization"])
    
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
            
            # Plotting the Comparison with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_returns.index, y=algo_cumulative, mode='lines', name='Optimized Strategy', line=dict(color='forestgreen', width=2)))
            fig.add_trace(go.Scatter(x=test_returns.index, y=benchmark_cumulative, mode='lines', name=f'Equal Weight Benchmark ({even_split:.1%})', line=dict(color='gray', width=2, dash='dash')))
            fig.update_layout(title="Out-of-Sample Performance", yaxis_title='Portfolio Value ($)', hovermode="x unified", template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
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
            with st.expander("How this Engine Works"):
                st.markdown("""
                This application is powered by a custom Python backend (`quant_engine.py`) that executes the following algorithmic sequence:
                1. **Data Ingestion:** Fetches historical daily adjusted close prices via the `yfinance` API.
                2. **Monte Carlo Simulation (Training):** Generates thousands of random portfolio weight arrays using `NumPy`. It utilizes Rejection Sampling to discard any arrays that violate the maximum asset concentration limit.
                3. **Mathematical Optimization:** Calculates the annualized expected return and covariance matrix to find the exact weight distribution that maximizes the **Sharpe Ratio** (Return / Risk).
                4. **Forward Validation (Testing):** Locks in the optimized weights and matrix-multiplies them against unseen, out-of-sample data, dynamically tracking cumulative returns against a baseline 1/N equal-weight benchmark.
                """)

            with st.expander("Why does the Optimized Strategy sometimes lose to the Benchmark?"):
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
            sim_results, var_95_dollars, cvar_95_dollars = test_engine.run_monte_carlo(
                weights=optimal_weights, 
                time_horizon=30, 
                initial_investment=algo_cumulative[-1],
                num_simulations=10000
            )
            
            # --- Plotting the Simulation Distribution (Cash Value) ---
            cutoff = np.percentile(sim_results, 5)
            
            fig2 = go.Figure()
            # Main histogram
            fig2.add_trace(go.Histogram(x=sim_results, nbinsx=60, marker_color='darkseagreen', opacity=0.6, name='Simulated Outcomes'))
            # Tail histogram overlay
            tail_results = sim_results[sim_results < cutoff]
            fig2.add_trace(go.Histogram(x=tail_results, nbinsx=60, marker_color='indianred', opacity=0.9, name='5% Worst Cases'))
            
            fig2.add_vline(x=cutoff, line_dash="dash", line_color="red", annotation_text=f"95% VaR: ${cutoff:,.2f}")
            fig2.add_vline(x=algo_cumulative[-1], line_dash="dot", line_color="white", annotation_text=f"Starting Value: ${algo_cumulative[-1]:,.2f}")
            
            fig2.update_layout(barmode='overlay', title="Monte Carlo 30-Day Stress Test", xaxis_title="Simulated Final Portfolio Value ($)", yaxis_title="Number of Scenarios", template="plotly_dark", showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
            # --- Metrics Dashboard ---
            st.divider()
            v1, v2, v3, v4 = st.columns(4)
            
            expected_final_value = np.mean(sim_results)
            prob_positive = np.mean(sim_results > algo_cumulative[-1])
            
            v1.metric("Average Simulated Value", f"${expected_final_value:,.2f}", f"${(expected_final_value - algo_cumulative[-1]):,.2f} Expected Profit")
            v2.metric("Probability of Profit", f"{prob_positive:.1%}", help="Percentage of simulated paths that finished above your starting value.")
            v3.metric("30-Day Value at Risk (VaR)", f"${var_95_dollars:,.2f}", delta="Max expected loss (95% conf)", delta_color="inverse")
            v4.metric("30-Day CVaR", f"${cvar_95_dollars:,.2f}", delta="Avg loss in worst 5%", delta_color="inverse")
            # --- NEW: Monte Carlo Graph Explanation ---
            st.divider()
            with st.expander("Understanding the Stress Test & VaR Graph"):
                st.markdown("""
                ### Decoding the 30-Day Future Projection
                This distribution graph visualizes a **Monte Carlo Simulation**, plotting 10,000 statistically probable future realities for your newly optimized portfolio.
                
                * **Value at Risk (VaR):** The red dashed line represents the **95% Value at Risk**. Statistically, this represents our confidence threshold. We are 95% confident that the portfolio's value will *not* drop below this exact dollar amount over the next 30 days. 
                * **Graph Distribution:** The distribution of future outcomes is represented by the histogram. We assumed a fat-tail distribution, which means the tail of the distribution is especially volatile. (instead f a normal distribution, the tail would be flat)
                * **Limitations of this simulation:** The simulation is **not guaranteed** to accurately predict the market's future behavior. One of the reason is that this assumes that the probability of getting a high return is same as the probability of stock crashing. In practice, this is **not really the case**, and the simulation is **not perfect**. 
                """)

    # ==========================================
    # TAB 3: WALK-FORWARD OPTIMIZATION
    # ==========================================
    with tab3:
        st.header("Rigorous Walk-Forward Backtesting")
        st.markdown("Unlike a single train/test split, this model continually rolls forward, retraining on a sliding window of data and testing on the subsequent unseen window. This prevents overfitting and simulates a true live-trading environment.")
        
        with st.status("Executing Walk-Forward Engine...", expanded=True) as status:
            st.write("Fetching historical market DNA...")
            # We initialize a new engine spanning the entire requested timeline
            wfo_engine = QuantitativeEngine(tickers=tickers, start_date=start_date, end_date=end_date)
            
            st.write("Initializing Backtester module...")
            backtester = Backtester(wfo_engine)
            
            st.write(f"Running rolling optimizations (Train: 252 days, Test: 63 days)...")
            wfo_results = backtester.walk_forward_optimize(initial_investment=investment, train_window_size=252, test_window_size=63)
            
            status.update(label="Walk-Forward Optimization Complete!", state="complete", expanded=False)
            
        equity_curve = wfo_results['equity_curve']
        kupiec = wfo_results['kupiec_results']
        
        # Calculate Equal Weight Benchmark for the Walk-Forward period
        even_split = 1.0 / len(tickers)
        benchmark_weights = np.array([even_split] * len(tickers))
        
        # The first date in equity_curve is the start date (cash), actual returns start on index 1
        wfo_dates = equity_curve.index[1:]
        raw_test_returns = wfo_engine.daily_returns.loc[wfo_dates]
        
        benchmark_daily_returns = np.dot(raw_test_returns, benchmark_weights)
        benchmark_cumulative = investment * np.cumprod(1 + benchmark_daily_returns)
        # Prepend the initial investment to match the equity curve's length
        benchmark_cumulative = np.insert(benchmark_cumulative, 0, investment)
        
        st.subheader("Rolling Out-of-Sample Equity Curve")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve['Portfolio_Value'], mode='lines', name='WFO Strategy', line=dict(color='royalblue', width=2)))
        fig3.add_trace(go.Scatter(x=equity_curve.index, y=benchmark_cumulative, mode='lines', name=f'Equal Weight Benchmark ({even_split:.1%})', line=dict(color='gray', width=2, dash='dash')))
        fig3.update_layout(yaxis_title='Portfolio Value ($)', hovermode="x unified", template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)
        
        st.divider()
        st.subheader("VaR Calibration & Risk (Kupiec POF Test)")
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Walk-Forward Periods", f"{kupiec['total_periods']} Days")
        k2.metric("Expected Breach Rate", f"{kupiec['expected_rate']:.1%}")
        
        obs_rate = kupiec['observed_rate']
        exp_rate = kupiec['expected_rate']
        delta_val = f"{(obs_rate - exp_rate):.2%} vs Expected"
        k3.metric("Observed Breach Rate", f"{obs_rate:.1%}", delta=delta_val, delta_color="inverse")
        
        k4.metric("Kupiec p-value", f"{kupiec['p_value']:.4f}")
        
        if kupiec['is_calibrated']:
            st.success(f"✅ **Model Passed Calibration (Fails to reject H0).** The portfolio accurately predicts risk within a 95% confidence interval.")
        else:
            if obs_rate > exp_rate:
                st.error(f"⚠️ **Model Failed Calibration!** The algorithm is severely **underestimating** out-of-sample downside risk (too many breaches).")
            else:
                st.warning(f"⚠️ **Model Failed Calibration!** The algorithm is severely **overestimating** risk (too few breaches). It's too conservative.")

        st.divider()
        with st.expander("Understanding the Kupiec POF Test"):
            st.markdown("""
            ### What is the Kupiec POF Test?
            The **Kupiec Proportion of Failures (POF) test** is a statistical way to check if our risk model is actually working in the real world. 
            
            * **The Goal:** When we set a **95% Value at Risk (VaR)**, we are predicting that the portfolio will only lose more than that amount on **5% of the days**.
            * **The Reality Check:** The Kupiec test looks at how many days the portfolio *actually* lost more than the predicted VaR (called a "breach" or "exception").
            * **The Result:** If we expect breaches on 5% of days, but we actually see them on 10% of days, the model failed (it was too optimistic). The Kupiec test gives us a mathematical p-value to determine if the number of breaches is acceptable or if the risk model is broken.
            """)

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
        
        # Expected from Train Engine
        m1.metric("MPT Expected Return", f"{train_engine.expected_return:.2%}")
        m1.caption("Based on Training Data")
        
        m2.metric("MPT Expected Volatility", f"{train_engine.volatility:.2%}")
        m2.caption("Based on Training Data")
        
        # Actual from Test Engine
        m3.metric("Actual Optimized Return", f"{actual_return:.2%}", delta=f"{(actual_return - benchmark_return):.2%} vs Benchmark")
        m3.caption("Based on Out-of-Sample Data")
        
        m4.metric("Actual Benchmark Return", f"{benchmark_return:.2%}")
        m4.caption("Based on Out-of-Sample Data")

        # --- NEW: Final Dollar Amounts ---
        st.divider()
        st.subheader("Final Portfolio Value (Cash)")
        
        # Create 2 columns for the final cash payout
        val1, val2 = st.columns(2)
        
        # The Optimized Strategy Cash Value (with a dollar comparison to the benchmark)
        val1.metric(
            label="Optimized Strategy Final Value", 
            value=f"${algo_cumulative[-1]:,.2f}",
        )
        
        # The Benchmark Cash Value
        val2.metric(
            label="Benchmark Final Value", 
            value=f"${benchmark_cumulative[-1]:,.2f}"
        )
        # --- NEW: Educational Breakdown ---
        st.divider()
        with st.expander("📚 Why does the Optimized Strategy sometimes lose to the Benchmark?"):
            st.markdown("""
            **The Out-of-Sample Reality (Estimation Error)**
            
            In quantitative finance, it is incredibly common for a mathematically "perfect" optimized portfolio to underperform a naive equal-weight benchmark in out-of-sample testing. This happens due to two primary factors:
            
            * **Estimation Error:** Mean-variance optimization requires estimating future returns and volatility based purely on historical data. The optimizer acts as an "error maximizer," heavily weighting assets that had a lucky historical run and heavily penalizing assets that had a temporary dip. 
            * **The Robustness of Equal Weighting:** The equal-weight benchmark (1/N) makes absolutely zero assumptions about the future. Because it does not rely on historical data, it suffers from zero estimation error, making it statistically highly robust to sudden market regime shifts.
            
            **The Takeaway:** If an algorithm crushes the market during the training phase but lags behind the benchmark during the testing phase, it proves the model *overfit* to the past rather than learning a persistent future pattern.
            """)
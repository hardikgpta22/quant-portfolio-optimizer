# 📈 Quantitative Portfolio Engine & Risk Optimizer

An end-to-end algorithmic portfolio optimization engine built with Python, simulating historical market data to construct the mathematically optimal asset allocation based on the Markowitz Efficient Frontier. 

## 🧮 The Mathematics & Methodology

### 1. Modern Portfolio Theory (MPT)
The optimization engine is built on Harry Markowitz's Modern Portfolio Theory. The algorithm generates thousands of random weight distributions to plot the **Efficient Frontier**, seeking the exact asset allocation that maximizes expected return for a given level of risk. 

The primary optimization metric is the **Sharpe Ratio**:
$$\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}$$
*(Where $R_p$ is portfolio return, $R_f$ is the risk-free rate, and $\sigma_p$ is portfolio standard deviation).*

### 2. Overcoming Look-Ahead Bias (The Graph Behavior)
A common pitfall in financial modeling is "Look-Ahead Bias"—optimizing an algorithm on a specific timeframe and then testing it on that exact same data, which falsely inflates performance. 

### 3. Risk Management & Concentration Limits
Unconstrained optimization models frequently suffer from **Concentration Risk**, where the algorithm dumps the majority of capital into a single historically high-performing asset, destroying true diversification. 

To counteract this, the engine's Monte Carlo simulator utilizes **Rejection Sampling**. It strictly enforces a customizable maximum weight constraint (e.g., no single asset can exceed 40% of the portfolio). Any mathematically generated array that breaches this risk threshold is algorithmically discarded and redrawn, ensuring the final efficient frontier represents realistic, well-diversified portfolios.

To ensure mathematical integrity, this engine uses a strict **Train/Test Split**:
* **The Training Phase (The Past):** The engine calculates the historical covariance matrix and expected returns to find the mathematically perfect weights.
* **The Testing Phase (The Future):** Those weights are locked in and aggressively tested against unseen future market data. 

**Why the Graph Behaves This Way:** Because the algorithm is operating on out-of-sample data, you will see realistic variance. It will not perfectly predict future market shocks, but the optimized strategy is mathematically designed to experience less severe drawdowns (lower volatility) and a higher risk-adjusted return over time compared to a standard equal-weight benchmark.

## 🚀 Features
* **Modern Portfolio Theory (MPT):** Calculates the optimal Sharpe Ratio via 5,000+ simulated weight allocations.
* **Out-of-Sample Backtesting:** Eliminates look-ahead bias by splitting historical data into training (optimization) and testing (forward-validation) periods to accurately gauge real-world performance.
* **Performance Analytics:** Compares optimized strategy returns against an equal-weight benchmark, calculating expected volatility vs. actual cash returns.
* **Live Interactive Web App:** A fully dynamic Streamlit frontend allowing custom stock ticker inputs, investment sizing, and dynamic timeline adjustments.

## 🛠️ Technology Stack
* **Backend:** Python (OOP Architecture)
* **Quantitative Math:** NumPy, SciPy 
* **Data Engineering:** Pandas, yFinance
* **Frontend UI:** Streamlit, Matplotlib

## ⚙️ How to Run Locally
1. Clone this repository.
2. Install the engineering dependencies: `pip install -r requirements.txt`
3. Ignite the web server: `streamlit run app.py`
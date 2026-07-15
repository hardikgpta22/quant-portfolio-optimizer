# 📈 Quantitative Portfolio Engine & Risk Optimizer

Firstly, An end-to-end algorithmic portfolio optimization engine built with Python, simulating historical market data to construct the mathematically optimal asset allocation based on the Markowitz Efficient Frontier. 
And the Second part, running Monte Carlo Simulation on the test result from the optimization engine to simulate and get probability graph of portfolio amount after 30 days, mentioning VaR (Value at Risk) and the expected amount. 

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

### 4. Predictive Stress Testing & Value at Risk (VaR)
Beyond historical backtesting, this engine includes a **30-Day Forward-Looking Monte Carlo Stress Test** to simulate future out-of-sample realities.

#### The Mathematics of the Simulation
Instead of predicting a single future price, the engine executes a stochastic Monte Carlo simulation to draw 10,000 statistically probable future asset paths.
* **Stochastic Path Generation:** It correlates the random walks across assets using the **Cholesky Decomposition** ($L \cdot L^T = \Sigma$) of the portfolio's historical covariance matrix. This ensures structural market relationships and cross-asset correlations remain mathematically intact during the simulation.
* **Fat-Tail Modeling:** Standard financial models assume normal distributions (bell curves), which notoriously underestimate real-world market crashes. This engine deliberately utilizes a **Multivariate Student’s t-distribution** (with 10 degrees of freedom). This mathematical choice explicitly models "fat tails," ensuring black-swan events and extreme outliers are accurately represented.

#### Downside Risk Projection
* **Value at Risk (VaR):** By sorting the 10,000 simulated outcomes, the algorithm empirically isolates the 5th percentile. This produces a strict 95% confidence VaR threshold, answering exactly how much capital is at risk of loss in a severe macroeconomic drawdown over the next month.

## 📉 The Out-of-Sample Phenomenon: Optimization vs. Equal Weight
A core feature of this engine is its strict separation of in-sample training and out-of-sample testing. Users will frequently observe the Optimized Strategy underperforming the Equal-Weight Benchmark during the forward-testing phase. This accurately reflects a well-documented phenomenon in quantitative finance:

1. **Estimation Error Maximization:** Markowitz optimization relies heavily on the historical covariance matrix and mean returns. Small fluctuations in historical data lead to massive changes in the optimal weights, causing the algorithm to over-allocate to historical anomalies.
2. **The 1/N Advantage:** An equal-weight benchmark (1/N) carries zero estimation error because it relies on no historical parameters. In highly volatile, unpredictable market regimes, this "naive" diversification mathematically provides more robust out-of-sample performance than curve-fit historical optimizations. 
3. **Overfitting Detection:** By displaying this performance gap, the engine successfully identifies and visualizes algorithmic overfitting, proving that historical mean-variance dominance does not guarantee future outperformance.

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
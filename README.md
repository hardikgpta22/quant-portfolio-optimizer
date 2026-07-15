# 📈 Quantitative Portfolio Engine & Risk Optimizer

An end-to-end algorithmic portfolio optimization engine built with Python, simulating historical market data to construct the mathematically optimal asset allocation based on the Markowitz Efficient Frontier. 

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
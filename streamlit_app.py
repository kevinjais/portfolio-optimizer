import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("Indian EMS Portfolio Optimization Dashboard")

# Set Risk Free Rate
# Based on average Indian Government bond yields (~7% for 91-day T-bills and 10-year G-Secs)
risk_free_rate = 0.07

# Indian EMS Stocks
tickers = st.multiselect("Select Stocks",['DIXON.NS', 'SYRMA.NS', 'KAYNES.NS', 'AMBER.NS', 'CENTUM.NS','AVALON.NS', 'PGEL.NS', 'DCXINDIA.NS', 'ELIN.NS'],default=['DIXON.NS', 'SYRMA.NS', 'AMBER.NS'])

if len(tickers)<2:
    st.warning("Please select at least two stocks.")
    st.stop()

# Getting the Data
data = yf.download(tickers, start='2023-04-18', end='2025-06-24', auto_adjust=True)['Close']
log_returns = np.log(data / data.shift(1)).dropna()
mean_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252

st.subheader("Price Chart")
st.line_chart(data)

# Returns & Volatility in a Table
st.subheader("Returns & Volatility")
stats_df = pd.DataFrame({
    "Annualized Return": mean_returns,
    "Annualized Volatility": np.sqrt(np.diag(cov_matrix))
})
st.dataframe(stats_df.style.format("{:.2%}"))

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = log_returns.corr()
fig2, ax2 = plt.subplots(figsize=(8, 6))
import seaborn as sns
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2, fmt=".2f", linewidths=0.5)
st.pyplot(fig2)

# Optimization Button
if st.button("Run Monte Carlo + Sharpe Optimization"):
    # Monte Carlo Simulation
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
    # Efficient Frontier Plot
    st.subheader("Efficient Frontier")
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5)
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.set_title('Efficient Frontier')
    fig.colorbar(sc, label='Sharpe Ratio')
    st.pyplot(fig)
    # Sharpe Ratio Optimization
    def neg_sharpe(weights, mean_returns, cov_matrix, rf):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(ret - rf) / vol
        
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    init_guess = np.array([1 / len(tickers)] * len(tickers))

    opt = minimize(neg_sharpe, init_guess, args=(mean_returns, cov_matrix, risk_free_rate),
                   method='SLSQP', bounds=bounds, constraints=constraints)

    opt_weights = opt.x
    opt_ret = np.dot(opt_weights, mean_returns)
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    opt_sharpe = (opt_ret - risk_free_rate) / opt_vol

    st.subheader("Optimized Portfolio Allocation")
    for i, ticker in enumerate(tickers):
        st.write(f"{ticker}: {opt_weights[i]:.2%}")
    st.write(f"**Expected Return**: {opt_ret:.2%}")
    st.write(f"**Volatility**: {opt_vol:.2%}")
    st.write(f"**Sharpe Ratio (Adj. for RFR)**: {opt_sharpe:.2f}")

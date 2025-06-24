EMS Portfolio Optimization Tool

This is a web app project to help users analyze and build stock portfolios made up of Electronics Manufacturing Services (EMS) companies
It uses past stock data to test thousands of different combinations and helps find the best performing portfolios in terms of risk and return
This is built using Python, Streamlit, pandas, NumPy, yfinance, matplotlib, seaborn, scipy to simulate and optimize EMS stock portfolios

What This Tool Does:
1. It collects the daily adjusted closing prices (April 2023 to June 2025) for Listed India EMS companies using yfinance
2. It uses NumPy and pandas to calculate each stocks annualized return & volatility based on daily log returns and compute a covariance matrix to show how the stock prices move together
3. It computes the correlation matrix of returns and uses Seaborns heatmap to visualize it
4. It runs a Monte Carlo simulation by generating 10,000+ random portfolio weight combinations to estimate possible portfolio returns, volatilities and Sharpe ratios (adjusted for a 7% risk-free rate based on average Indian government bond yields)
5. It uses scipy.optimize.minimize with the SLSQP algorithm to find the portfolio weights that maximize the Sharpe ratio while satisfying constraints (weights sum to 1 and are non-negative)
6. It visualizes the Efficient Frontier using matplotlib which shows the tradeoff between portfolio risk and return with color coded Sharpe ratios
7. It displays all the outputs using Streamlit for an interactive user interface

The tool focuses on 9 Listed Indian EMS companies:
1. DIXON Technologies (DIXON.NS)
2. Syrma SGS Technology (SYRMA.NS)
3. Kaynes Technology (KAYNES.NS)
4. Amber Enterprises (AMBER.NS)
5. Centum Electronics (CENTUM.NS)
6. Avalon Technologies (AVALON.NS)
7. PG Electroplast (PGEL.NS)
8. DCX Systems (DCXINDIA.NS)
9. Elin Electronics (ELIN.NS)



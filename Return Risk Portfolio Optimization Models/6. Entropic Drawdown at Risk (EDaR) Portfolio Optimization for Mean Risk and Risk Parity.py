import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import riskfolio as rp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

# MEAN ENTROPIC DRAWDOWN AT RISK (EDaR) OPTIMIZATION #

# 1. IMPORTING DATA

# Date range
start = '2010-01-01'
end = datetime.now() - timedelta(1)

# Tickers of assets
assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
          'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
          'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
assets.sort()

# Downloading data
data = yf.download(assets, start=start, end=end)
data = data.loc[:, "Adj Close"].dropna()
data.columns = assets

# Calculating returns
Y = data.pct_change().dropna()

# 2. ESTIMATING MEAN EDaR PORTFOLIOS
# 2.1 Calculating the portfolio that optimize EDaR ratio
port = rp.Portfolio(returns=Y)

method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:
port.solvers = ['MOSEK']  # It is recommended to use mosek when optimizing EVaR
port.alpha = 0.05  # Significance level for CVaR, EVaR y CDaR
model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'EDaR'  # Risk measure used, this time will be EVaR
obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk-free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w)
print(w.T)

# 2.2 Plotting portfolio composition
ax0 = rp.plot_pie(w=w, title='Sharpe Mean - Entropic Drawdown at Risk', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

# 2.3 Plotting risk measures
# Plotting the drawdown risk measures
ax1 = rp.plot_drawdown(returns=Y,
                      w=w,
                      alpha=0.0,
                      height=8,
                      width=10,
                      height_ratios=[2, 3],
                      ax=None)
plt.show()

# 2.4 Calculate Efficient Frontier
points = 100  # Number of points of the frontier
frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
print(frontier.T.head())

# Plotting the efficient frontier
label = 'Max Risk Adjusted Return Portfolio'  # Title of point
mu = port.mu  # Expected returns
cov = port.cov  # Covariance matrix
returns = port.returns  # Returns of the assets

ax2 = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                       rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                       marker='*', s=16, c='r', height=6, width=10, ax=None)
plt.show()

# Plotting efficient frontier composition
ax3 = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
plt.show()

# 3. ESTIMATING RISK PARITY PORTFOLIOS FOR EDaR
# 3.1 Calculating the risk parity portfolio for EDaR
b = None  # Risk contribution constraints vector
w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
print(w_rp.T)

# 3.2 Plotting portfolio composition
ax4 = rp.plot_pie(w=w_rp, title='Risk Parity EVaR', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

# 3.3 Plotting Risk Composition
ax5 = rp.plot_risk_con(w_rp, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.01,
                       color="tab:blue", height=6, width=10, ax=None)
plt.show()



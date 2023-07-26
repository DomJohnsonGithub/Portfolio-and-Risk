import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import riskfolio as rp
import matplotlib.pyplot as plt
import seaborn as sns
import mosek

sns.set_style("darkgrid")
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

# DOLLAR NEUTRAL PORTFOLIOS #

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

# 2. DOLLAR NEUTRAL PORTFOLIO WITH CONSTRAINT ON STANDARD DEVIATION
# 2.1 Calculating the dollar neutral portfolio
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio
method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Market neutral constraints:

port.sht = True  # Allows short positions
port.uppersht = 1  # Upper bound for sum of negative weights
port.upperlng = 1  # Upper bound for sum of positive weights
port.budget = 0  # Sum of all weights
port.upperdev = 0.20 / 252 ** 0.5  # Upper bound for daily standard deviation

# Estimate optimal portfolio:
model = 'Classic'  # Could be Classic (historical), BL (Black Litterman), FM (Factor Model)
# or BL_FM (Black Litterman with Factor Model)
rm = 'MV'  # Risk measure used, this time will be variance
obj = 'MaxRet'  # For Market Neutral the objective must be
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk-free rate
l = 3  # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print("Sum weights : ", np.round(np.sum(w.to_numpy()), 4))
print(w.T)

# 2.2 Plotting portfolio composition (in absolute values)
title = "Max Return Dollar Neutral with Variance Constraint"
ax0 = rp.plot_pie(w=w,
                  title=title,
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=7,
                  width=10,
                  ax=None)
plt.show()

# Plotting the composition of the portfolio using bar chart
ax1 = rp.plot_bar(w,
                  title="Max Return Dollar Neutral with Variance Constraint",
                  kind="v",
                  others=0.05,
                  nrow=25,
                  height=6,
                  width=10)
plt.show()

# 2.3 Calculate efficient frontier
points = 100  # Number of points of the frontier
port.upperdev = None  # Deleting the upper bound for daily standard deviation

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
print(frontier)

label = "Max Return Dollar Neutral with Variance Constraint"  # Title of point
mu = port.mu  # Expected returns
cov = port.cov  # Covariance matrix
returns = port.returns  # Returns of the assets

ax2 = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                       rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                       marker='*', s=16, c='r', height=6, width=10, ax=None)
plt.show()

# Plotting efficient frontier composition in absolute values
ax3 = rp.plot_frontier_area(w_frontier=np.abs(frontier), cmap="tab20", height=6, width=10, ax=None)
plt.show()

# 3. DOLLAR NEUTRAL PORTFOLIO WITH CONSTRAINT ON CVaR
# 3.1 Calculating Dollar Neutral Portfolio
rm = 'CVaR'  # Risk measure
port.upperCVaR = 0.40 / 252 ** 0.5  # Creating an upper bound for daily CVaR

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

print("Sum weights : ", np.round(np.sum(w.to_numpy()), 4))
print(w.T)

# 3.2 Plotting portfolio composition
title = "Max Return Dollar Neutral with CVaR Constraint"
ax4 = rp.plot_pie(w=w,
                  title=title,
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=7,
                  width=10,
                  ax=None)
plt.show()

# Plotting the composition of the portfolio using bar chart
ax5 = rp.plot_bar(w,
                  title="Max Return Dollar Neutral with CVaR Constraint",
                  kind="v",
                  others=0.05,
                  nrow=25,
                  height=6,
                  width=10)
plt.show()

# 3.3 Calculate efficient frontier
points = 50  # Number of points of the frontier
port.upperCVaR = None  # Deleting the upper bound for daily CVaR

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
print(frontier.T.head())

label = "Max Return Dollar Neutral with CVaR Constraint"  # Title of point
ax6 = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                       rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                       marker='*', s=16, c='r', height=6, width=10, ax=None)
plt.show()

# Plotting efficient frontier composition in absolute values
ax7 = rp.plot_frontier_area(w_frontier=np.abs(frontier), cmap="tab20", height=6, width=10, ax=None)
plt.show()

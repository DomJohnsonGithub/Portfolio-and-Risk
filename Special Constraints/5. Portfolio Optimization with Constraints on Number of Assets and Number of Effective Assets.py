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

# CONSTRAINTS ON NUMBER OF ASSETS #

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

# 2. ESTIMATING MEAN VARIANCE PORTFOLIOS
# 2.1 Calculating the portfolio that maximizes sharpe
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio
method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio
model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'CVaR'  # Risk measure used, this time will be variance
obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk-free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

print(w.T)

# 2.2 Plotting portfolio composition
ax0 = rp.plot_pie(w=w,
                  title='Sharpe Mean CVaR',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=6,
                  width=10,
                  ax=None)
plt.show()

# Number of assets in portfolio
n_assets = np.sum(np.where(np.round(w, 4) > 0, 1, 0)).item()

# Number of effective assets in portfolio
nea = 1 / np.sum(w ** 2).item()

print('Number of Assets:', n_assets)
print('Number of Effective Assets:', nea)

# 2.3 Calculating the portfolio including a constraint on the maximum number of assets
# useful if cant invest in large no. of assets
# First we need to set a solver that support Mixed Integer Programming
port.solvers = ['MOSEK']

# Then we need to set the cardinality constraint (maximum number of assets)
port.card = 5

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)

# 2.4 Plotting portfolio composition
ax1 = rp.plot_pie(w=w,
                  title='Sharpe Mean CVaR',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=6,
                  width=10,
                  ax=None)
plt.show()

# Number of assets in portfolio
n_assets = np.sum(np.where(np.round(w, 4) > 0, 1, 0)).item()

# Number of effective assets in portfolio
nea = 1 / np.sum(w ** 2).item()

print('Number of Assets:', n_assets)
print('Number of Effective Assets:', nea)

# 2.5 Calculating the portfolio including a constraint on the minimum number of effective assets
# helps increase portfolio diversification
# First we need to delete the cardinality constraint
port.card = None

# Then we need to set the constraint on the minimum number of effective assets
port.nea = 12

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)

# 2.6 Plotting portfolio composition
ax2 = rp.plot_pie(w=w,
                  title='Sharpe Mean CVaR',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=6,
                  width=10,
                  ax=None)
plt.show()

# Number of assets in portfolio
n_assets = np.sum(np.where(np.round(w, 4) > 0, 1, 0)).item()

# Number of effective assets in portfolio
nea = 1 / np.sum(w ** 2).item()

print('Number of Assets:', n_assets)
print('Number of Effective Assets:', nea)

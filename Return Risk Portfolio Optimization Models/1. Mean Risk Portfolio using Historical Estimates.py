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

# CLASSIC MEAN RISK OPTIMIZATION #

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

# Calculate returns
Y = data.pct_change().dropna()

# 2. ESTIMATING MEAN VARIANCE PORTFOLIOS

# 2.1 Calculating the portfolio that maximizes Sharpe ratio.
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'MV'  # Risk measure used, this time will be variance
obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk-free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)

# 2.2 Plotting portfolio composition
ax0 = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

# 2.3 Calculate efficient frontier
points = 100  # Number of points of the frontier
frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
print(frontier.T.head())

# Plotting the efficient frontier
# Plotting the efficient frontier

label = 'Max Risk Adjusted Return Portfolio'  # Title of point
mu = port.mu  # Expected returns
cov = port.cov  # Covariance matrix
returns = port.returns  # Returns of the assets

ax1 = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                       rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                       marker='*', s=16, c='r', height=6, width=10, ax=None)
plt.show()

# Plotting efficient frontier composition
ax2 = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
plt.show()

# 3. ESTIMATING MEAN RISK PORTFOLIOS
# calculate optimal portfolios for several risk measures:

# 3.1 Calculating the portfolio that maximizes Return/CVaR ratio
rm = 'CVaR'  # Risk measure

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)

# 3.2 Plotting portfolio composition
ax3 = rp.plot_pie(w=w, title='Sharpe Mean CVaR', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

# 3.3 Calculate efficient frontier
points = 100  # Number of points of the frontier
frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
print(frontier.T.head())

label = 'Max Risk Adjusted Return Portfolio'  # Title of point

ax4 = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                       rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                       marker='*', s=16, c='r', height=6, width=10, ax=None)
plt.show()

# Plotting efficient frontier composition
ax5 = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
plt.show()

# 3.4 Calculate Optimal Portfolios for Several Risk Measures

# Risk Measures available:

# 'MV': Standard Deviation.
# 'MAD': Mean Absolute Deviation.
# 'MSV': Semi Standard Deviation.
# 'FLPM': First Lower Partial Moment (Omega Ratio).
# 'SLPM': Second Lower Partial Moment (Sortino Ratio).
# 'CVaR': Conditional Value at Risk.
# 'EVaR': Entropic Value at Risk.
# 'WR': Worst Realization (Minimax)
# 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
# 'ADD': Average Drawdown of uncompounded cumulative returns.
# 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
# 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
# 'UCI': Ulcer Index of uncompounded cumulative returns.

rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
       'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']

w_s = pd.DataFrame([])
for i in rms:
    w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
    w_s = pd.concat([w_s, w], axis=1)

w_s.columns = rms
w_s.style.format("{:.2%}").background_gradient(cmap='YlGn')
print(w_s)

# Plotting a comparison of assets weights for each portfolio
fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)
w_s.plot.bar(ax=ax)
plt.show()

# 4. CONSTRAINTS ON ASSETS AND ASSET CLASSES

# 4.1 Creating the Constraints
asset_classes = {'Assets': ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
                            'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
                            'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA'],
                 'Industry': ['Consumer Discretionary', 'Consumer Discretionary',
                              'Consumer Discretionary', 'Consumer Staples',
                              'Consumer Staples', 'Energy', 'Financials',
                              'Financials', 'Financials', 'Financials',
                              'Health Care', 'Health Care', 'Industrials', 'Industrials',
                              'Industrials', 'Health Care', 'Industrials',
                              'Information Technology', 'Information Technology',
                              'Materials', 'Telecommunications Services', 'Utilities',
                              'Utilities', 'Telecommunications Services', 'Financials']}

asset_classes = pd.DataFrame(asset_classes)
asset_classes = asset_classes.sort_values(by=['Assets'])

constraints = {'Disabled': [False, False, False, False, False],
               'Type': ['All Assets', 'Classes', 'Classes', 'Classes',
                        'Classes'],
               'Set': ['', 'Industry', 'Industry', 'Industry', 'Industry'],
               'Position': ['', 'Financials', 'Utilities', 'Industrials',
                            'Consumer Discretionary'],
               'Sign': ['<=', '<=', '<=', '<=', '<='],
               'Weight': [0.10, 0.2, 0.2, 0.2, 0.2],
               'Type Relative': ['', '', '', '', ''],
               'Relative Set': ['', '', '', '', ''],
               'Relative': ['', '', '', '', ''],
               'Factor': ['', '', '', '', '']}

constraints = pd.DataFrame(constraints)
print(constraints)

A, B = rp.assets_constraints(constraints, asset_classes)

# 4.2 Optimize the portfolio with the constraints
port.ainequality = A
port.binequality = B

model = 'Classic'
rm = 'MV'
obj = 'Sharpe'
rf = 0

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)

ax6 = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

w_classes = pd.concat([asset_classes.set_index('Assets'), w], axis=1)
print(w_classes)

w_classes = w_classes.groupby(['Industry']).sum()
print(w_classes)

ax7 = rp.plot_pie(w=w_classes, title='Sharpe Mean Variance', others=0.05, nrow=25,
                  cmap="tab20", height=6, width=10, ax=None)
plt.show()

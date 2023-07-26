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

# RELAXED RISK PARITY PORTFOLIO OPTIMIZATION #

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

# 2. ESTIMATING VANILLA RISK PARITY PORTFOLIO
# 2.1 Calculating the vanilla risk parity portfolio
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio
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

w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, hist=hist)
print(w_rp.T)

# 2.2 Plotting portfolio composition
ax0 = rp.plot_pie(w=w_rp, title='Risk Parity Variance', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

# 2.3 Plotting Risk Composition
fig, ax = plt.figure()
ax = rp.plot_risk_con(w_rp,
                      cov=port.cov,
                      returns=port.returns,
                      rm=rm,
                      rf=0,
                      alpha=0.01,
                      color="tab:blue",
                      erc_line=True,
                      height=6,
                      width=10,
                      ax=ax)
plt.show()

# 3. ESTIMATING RELAXED RISK PARITY PORTFOLIOS
# versions A, B and C of the model. The relaxed risk parity is a model that allows
# to incorporate constraints on returns in the vanilla risk parity model
# 3.1 Calculating the relaxed risk parity portfolio version A
b = None  # Risk contribution constraints vector
version = 'A'  # Could be A, B or C
l = 1  # Penalty term, only valid for C version

# Setting the return constraint
port.lowerret = 0.00056488 * 1.5

w_rrp_a = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)
print(w_rrp_a.T)

# 3.2 Plotting portfolio composition
ax2 = rp.plot_pie(w=w_rrp_a, title='Relaxed Risk Parity A', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

# 3.3 Plotting Risk Composition
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the risk composition of the portfolio
ax = rp.plot_risk_con(w_rrp_a,
                      cov=port.cov,
                      returns=port.returns,
                      rm=rm,
                      rf=0,
                      alpha=0.01,
                      color="tab:blue",
                      erc_line=True,
                      height=6,
                      width=10,
                      ax=ax)
plt.show()

# 3.4 Calculating the relaxed risk parity portfolio version B
version = 'B'  # Could be A, B or C
w_rrp_b = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)
print(w_rrp_b.T)

# 3.5 Plotting portfolio composition
ax3 = rp.plot_pie(w=w_rrp_b, title='Relaxed Risk Parity B', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

# 3.6 Plotting Risk Composition
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the risk composition of the portfolio
ax = rp.plot_risk_con(w_rrp_b,
                      cov=port.cov,
                      returns=port.returns,
                      rm=rm,
                      rf=0,
                      alpha=0.01,
                      color="tab:blue",
                      erc_line=True,
                      height=6,
                      width=10,
                      ax=ax)

plt.show()

# 3.7 Calculating the relaxed risk parity portfolio version C
version = 'C'  # Could be A, B or C
w_rrp_c = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)
print(w_rrp_c.T)

# 3.8 Plotting portfolio composition
ax4 = rp.plot_pie(w=w_rrp_c, title='Relaxed Risk Parity C', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

# 3.9 Plotting Risk Composition
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the risk composition of the portfolio
ax = rp.plot_risk_con(w_rrp_c,
                      cov=port.cov,
                      returns=port.returns,
                      rm=rm,
                      rf=0,
                      alpha=0.01,
                      color="tab:blue",
                      erc_line=True,
                      height=6,
                      width=10,
                      ax=ax)

plt.show()

# 4. ESTIMATING RELAXED RISK PARITY PORTFOLIOS WITH LINEAR CONSTRAINTS
# 4.1 Building the Linear Constraints
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

# 4.2 Calculating the relaxed risk parity portfolio version C
A, B = rp.assets_constraints(constraints, asset_classes)

port.ainequality = A
port.binequality = B

w_rrp_c = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)
print(w_rrp_c.T)

# 4.3 Plotting portfolio composition
ax5 = rp.plot_pie(w=w_rrp_c, title='Relaxed Risk Parity C with LC', others=0.05, nrow=25, cmap="tab20",
                  height=6, width=10, ax=None)
plt.show()

w_classes = pd.concat([asset_classes.set_index('Assets'), w_rrp_c], axis=1)
w_classes = w_classes.groupby(['Industry']).sum()
print(w_classes)

# 4.4 Plotting Risk Composition
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the risk composition of the portfolio
ax = rp.plot_risk_con(w_rrp_c,
                      cov=port.cov,
                      returns=port.returns,
                      rm=rm,
                      rf=0,
                      alpha=0.01,
                      color="tab:blue",
                      erc_line=True,
                      height=6,
                      width=10,
                      ax=ax)

plt.show()

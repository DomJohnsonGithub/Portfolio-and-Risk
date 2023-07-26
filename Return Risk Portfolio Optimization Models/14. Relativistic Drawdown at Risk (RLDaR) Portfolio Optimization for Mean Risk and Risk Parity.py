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

# MEAN RELATIVISTIC DRAWDOWN AT RISK (RLDaR) OPTIMIZATION #

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

# 2. ESTIMATING MEAN RLDaR PORTFOLIOS
# 2.1 Calculating the portfolio that optimize return/RLDaR ratio.
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimum portfolio
method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.
port.assets_stats(method_mu=method_mu,
                  method_cov=method_cov)

port.solvers = ['MOSEK']  # It is recommended to use mosek when optimizing GMD
port.sol_params = {'MOSEK': {'mosek_params': {'MSK_IPAR_NUM_THREADS': 2}}}

# Estimate optimal portfolio:
kappa = 0.3
alpha = 0.2
port.kappa = kappa
port.alpha = alpha

model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'EDaR'  # Risk measure used, this time will be Tail Gini Range
obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk-free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)

# 2.2 Plotting portfolio composition
ax0 = rp.plot_pie(w=w,
                  title='Sharpe Mean - RLDaR',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=6,
                  width=10,
                  ax=None)
plt.show()

# 2.3 Plotting risk measures
ax1 = rp.plot_drawdown(returns=Y,
                       w=w,
                       alpha=alpha,
                       kappa=kappa,
                       solver='MOSEK',
                       height=8,
                       width=10,
                       height_ratios=[2, 3],
                       ax=None)
plt.show()

# 2.4 Calculate efficient frontier
points = 100  # Number of points of the frontier
frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
print(frontier.T.head())

label = 'Max Risk Adjusted Return Portfolio'  # Title of point
mu = port.mu  # Expected returns
cov = port.cov  # Covariance matrix
returns = port.returns  # Returns of the assets

ax2 = rp.plot_frontier(w_frontier=frontier,
                       mu=mu,
                       cov=cov,
                       returns=returns,
                       rm=rm,
                       rf=rf,
                       alpha=alpha,
                       kappa=kappa,
                       solver='MOSEK',
                       cmap='viridis',
                       w=w,
                       label=label,
                       marker='*',
                       s=16,
                       c='r',
                       height=6,
                       width=10,
                       ax=None)
plt.show()

# Plotting efficient frontier composition
ax3 = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
plt.show()

# 3. ESTIMATING RISK PARITY PORTFOLIOS FOR RLDaR
# 3.1 Calculating the risk parity portfolio for RLDaR
b = None  # Risk contribution constraints vector
w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
print(w_rp.T)

# 3.2 Plotting portfolio composition
ax4 = rp.plot_pie(w=w_rp,
                  title='Risk Parity RDVaR',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=6,
                  width=10,
                  ax=None)
plt.show()

# 3.3 Plotting Risk Composition
ax5 = rp.plot_risk_con(w_rp,
                       cov=port.cov,
                       returns=port.returns,
                       rm=rm,
                       rf=0,
                       alpha=alpha,
                       kappa=kappa,
                       solver='MOSEK',
                       color="tab:blue", height=6, width=10, ax=None)
plt.show()

# Plotting the efficient frontier
ws = pd.concat([w, w_rp], axis=1)
ws.columns = ["Max Return/RLDaR", "Risk Parity RLDaR"]

mu = port.mu  # Expected returns
cov = port.cov  # Covariance matrix
returns = port.returns  # Returns of the assets

ax6 = rp.plot_frontier(w_frontier=frontier,
                       mu=mu,
                       cov=cov,
                       returns=returns,
                       rm=rm,
                       rf=rf,
                       alpha=alpha,
                       kappa=kappa,
                       solver='MOSEK',
                       cmap='viridis',
                       w=ws,
                       marker='*',
                       s=16,
                       height=6,
                       width=10,
                       ax=None)
plt.show()

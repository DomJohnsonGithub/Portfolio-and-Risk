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

# HIGHER L-MOMENTS OWA PORTFOLIO OPTIMIZATION #

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

# 2. ESTIMATING HIGHER-L OWA PORTFOLIOS
# 2.1 Calculating the portfolio that minimize a risk measure that includes Higher L-Moments
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimum portfolio
method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.
port.assets_stats(method_mu=method_mu,
                  method_cov=method_cov, d=0.94)

port.solvers = ['MOSEK']  # It is recommended to use mosek when optimizing
port.sol_params = {'MOSEK': {'mosek_params': {'MSK_IPAR_NUM_THREADS': 2}}}

# Estimate optimal portfolio:
obj = 'MinRisk'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
rf = 0  # Risk-free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

owa_w = rp.owa_l_moment_crm(len(Y), k=4)

w = port.owa_optimization(obj=obj,
                          owa_w=owa_w,
                          rf=rf,
                          l=l)

print(w.T)

# 2.2 Plotting portfolio composition
ax0 = rp.plot_pie(w=w,
                  title='Min Higher L-moment OWA Risk Measure',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=6,
                  width=10,
                  ax=None)
plt.show()

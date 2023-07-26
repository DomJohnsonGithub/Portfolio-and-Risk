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

# OWA PORTFOLIO OPTIMIZATION #

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

# 2. ESTIMATING OWA PORTFOLIOS
# 2.1 Comparing Classical formulations vs OWA formulations
port = rp.Portfolio(returns=Y)

# Calculating optimum portfolio
method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolios:

port.solvers = ['MOSEK']  # It is recommended to use mosek when optimizing GMD
port.sol_params = {'MOSEK': {'mosek_params': {mosek.iparam.num_threads: 2}}}
alpha = 0.05

port.alpha = alpha
model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rms = ['CVaR', 'WR']  # Risk measure used, this time will be Tail Gini Range
objs = ['MinRisk', 'Sharpe']  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk-free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

ws = pd.DataFrame([])
for rm in rms:
    for obj in objs:
        # Using Classical models
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
        # Using OWA model
        if rm == "CVaR":
            owa_w = rp.owa_cvar(len(Y), alpha=alpha)
        elif rm == 'WR':
            owa_w = rp.owa_wr(len(Y))
        w1 = port.owa_optimization(obj=obj, owa_w=owa_w, rf=rf, l=l)
        ws1 = pd.concat([w, w1], axis=1)
        ws1.columns = ['Classic ' + obj + ' ' + rm, 'OWA ' + obj + ' ' + rm]
        ws1['diff ' + obj + ' ' + rm] = ws1['Classic ' + obj + ' ' + rm] - ws1['OWA ' + obj + ' ' + rm]
        ws = pd.concat([ws, ws1], axis=1)

ws.style.format("{:.2%}").background_gradient(cmap='YlGn', vmin=0, vmax=1)
print(ws)

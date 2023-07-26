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

# PORTFOLIO OPTIMIZATION WITH RISK FACTORS AND PRINCIPAL COMPONENTS REGRESSION #

# 1. IMPORTING DATA

# Date range
start = '2010-01-01'
end = datetime.now() - timedelta(1)

# Tickers of assets
assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
          'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
          'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
assets.sort()

# Tickers of factors
factors = ['MTUM', 'QUAL', 'VLUE', 'SIZE', 'USMV']
factors.sort()

tickers = assets + factors
tickers.sort()

# Downloading data
data = yf.download(tickers, start=start, end=end)
data = data.loc[:, "Adj Close"].dropna()
data.columns = tickers

# Calculating returns
X = data[factors].pct_change().dropna()
Y = data[assets].pct_change().dropna()

# 2. ESTIMATING MEAN VARIANCE PORTFOLIOS
# 2.1 Estimating the loadings matrix
step = 'Forward'  # Could be Forward or Backward stepwise regression
loadings = rp.loadings_matrix(X=X, Y=Y, stepwise=step)
loadings.style.format("{:.4f}").background_gradient(cmap='RdYlGn')
print(loadings)

# 2.2 Calculating the portfolio that maximizes Sharpe ratio
port = rp.Portfolio(returns=Y)

# Calculating optimum portfolio
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

port.factors = X
port.factors_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:
model='FM' # Factor Model
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = False # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk-free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.T)

# 2.3 Plotting portfolio composition
ax0 = rp.plot_pie(w=w, title='Sharpe FM Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)
plt.show()

# 2.4 Plotting Risk Contribution
ax1 = rp.plot_risk_con(w, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.01,
                      color="tab:blue", height=6, width=10, ax=None)
plt.show()

# 3. ESTIMATING RISK PARITY PORTFOLIO USING RISK FACTORS AND OTHER RISK MEASURES
# 3.1 Calculating the risk parity portfolio for variance
b = None # Risk contribution constraints vector
w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
print(w_rp.T)

# 3.2 Plotting portfolio composition
ax2 = rp.plot_pie(w=w_rp, title='Risk Parity Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)
plt.show()

# 3.3 Plotting Risk Composition
ax3 = rp.plot_risk_con(w_rp, cov=port.cov_fm, returns=port.returns_fm, rm=rm, rf=0, alpha=0.01,
                      color="tab:blue", height=6, width=10, ax=None)
plt.show()

# 3.4 Calculate Optimal Portfolios for Several Risk Measures
rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM',
       'EVaR', 'CVaR', 'CDaR', 'UCI', 'EDaR']

w_s = pd.DataFrame([])
hist = False
for i in rms:
    w = port.rp_optimization(model=model, rm=i, rf=rf, b=b, hist=hist)
    w_s = pd.concat([w_s, w], axis=1)

w_s.columns = rms
print(w_s)

# Plotting a comparison of assets weights for each portfolio

fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)
w_s.plot.bar(ax=ax)
plt.show()
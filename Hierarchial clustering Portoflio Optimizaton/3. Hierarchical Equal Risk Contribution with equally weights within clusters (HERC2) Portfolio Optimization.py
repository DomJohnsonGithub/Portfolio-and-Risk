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

# HIERARCHICAL EQUAL RISK CONTRIBUTION WITH EQUAL WIGHTS WITHIN CLUSTERS #

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

# Plotting Assets Clusters
ax0 = rp.plot_clusters(returns=Y,
                      codependence='pearson',
                      linkage='ward',
                      k=None,
                      max_k=10,
                      leaf_order=True,
                      dendrogram=True,
                      #linecolor='tab:purple',
                      ax=None)
plt.show()

# 2. ESTIMATING HERC AND HERC2 PORTOFLIO
# The HERC2 portfolio is just a modification of HERC portfolio making the weights within clusters equal
# and no based on naive risk parity like original HERC portfolio
# 2.1 Calculating the HERC and HERC2 portfolio
port = rp.HCPortfolio(returns=Y)

# Estimate optimal portfolio:

codependence = 'pearson' # Correlation matrix used to group assets in clusters
rm = 'MV' # Risk measure used, this time will be variance
rf = 0 # Risk free rate
linkage = 'ward' # Linkage method used to build clusters
max_k = 10 # Max number of clusters used in two difference gap statistic
leaf_order = True # Consider optimal order of leafs in dendrogram

w1 = port.optimization(model='HERC',
                       codependence=codependence,
                       covariance='hist',
                       rm=rm,
                       rf=rf,
                       linkage=linkage,
                       max_k=max_k,
                       leaf_order=leaf_order)

w2 = port.optimization(model='HERC2',
                       codependence=codependence,
                       covariance='hist',
                       rm=rm,
                       rf=rf,
                       linkage=linkage,
                       max_k=max_k,
                       leaf_order=leaf_order)

w = pd.concat([w1, w2], axis=1)
w.columns = ['HERC', 'HERC2']
print(w.sort_values(by='HERC', ascending=False))

# 2.2 Plotting Risk Composition
mu = Y.mean()
cov = Y.cov() # Covariance matrix
returns = Y # Returns of the assets

fig, ax = plt.subplots(2,1, figsize=(12, 12))

ax = np.ravel(ax)
rp.plot_risk_con(w=w1,
                 cov=cov,
                 returns=returns,
                 rm=rm,
                 rf=0,
                 alpha=0.05,
                 color="tab:blue",
                 height=6,
                 width=10,
                 t_factor=252,
                 ax=ax[0])

rp.plot_risk_con(w=w2,
                 cov=cov,
                 returns=returns,
                 rm=rm,
                 rf=0,
                 alpha=0.05,
                 color="tab:blue",
                 height=6,
                 width=10,
                 t_factor=252,
                 ax=ax[1])
plt.show()

# 2.4 Calculate Optimal HERC and HERC2 PORTFOLIOS FOR SEVERAL COVARIANCE ESTIMATORS
# Covariance estimators available:

# 'hist': use historical estimates.
# 'ewma1'': use ewma with adjust=True, see `EWM <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows>`_ for more details.
# 'ewma2': use ewma with adjust=False, see `EWM <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows>`_ for more details.
# 'ledoit': use the Ledoit and Wolf Shrinkage method.
# 'oas': use the Oracle Approximation Shrinkage method.
# 'shrunk': use the basic Shrunk Covariance method.
models = ['HERC'] * 6 + ['HERC2'] * 6
covariances = ['hist', 'ewma1', 'ewma2', 'ledoit', 'oas', 'shrunk'] * 2

w_s = pd.DataFrame([])
for i, j in zip(models, covariances):
    w = port.optimization(model=i,
                          codependence=codependence,
                          covariance=j,
                          rm=rm,
                          rf=rf,
                          linkage=linkage,
                          max_k=max_k,
                          leaf_order=leaf_order)

    w_s = pd.concat([w_s, w], axis=1)

w_s.columns = zip(models, covariances)
print(w_s)

# Plotting a comparison of assets weights for each portfolio
fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)
w_s.plot.bar(ax=ax)
plt.show()
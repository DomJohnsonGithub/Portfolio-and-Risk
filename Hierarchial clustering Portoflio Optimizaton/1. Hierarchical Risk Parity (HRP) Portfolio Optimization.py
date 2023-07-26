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

# HIERARCHICAL RISK PARITY PORTFOLIO OPTIMIZATION #

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

ax = rp.plot_dendrogram(returns=Y,
                        codependence='pearson',
                        linkage='single',
                        k=None,
                        max_k=10,
                        leaf_order=True,
                        ax=None)

# 2. ESTIMATING HRP PORTOFLIO
# 2.1 Calculating the HRP portfolio
port = rp.HCPortfolio(returns=Y)

# Estimate optimal portfolio:
model = 'HRP'  # Could be HRP or HERC
codependence = 'pearson'  # Correlation matrix used to group assets in clusters
rm = 'MV'  # Risk measure used, this time will be variance
rf = 0  # Risk-free rate
linkage = 'single'  # Linkage method used to build clusters
max_k = 10  # Max number of clusters used in two difference gap statistic, only for HERC model
leaf_order = True  # Consider optimal order of leafs in dendrogram

w = port.optimization(model=model,
                      codependence=codependence,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order)

print(w.T)

# 2.2 Plotting portfolio composition
ax0 = rp.plot_pie(w=w,
                  title='HRP Naive Risk Parity',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=8,
                  width=10,
                  ax=None)
plt.show()

# 2.3 Plotting Risk Composition
mu = Y.mean()
cov = Y.cov()  # Covariance matrix
returns = Y  # Returns of the assets

ax1 = rp.plot_risk_con(w=w,
                       cov=cov,
                       returns=returns,
                       rm=rm,
                       rf=0,
                       alpha=0.05,
                       color="tab:blue",
                       height=6,
                       width=10,
                       t_factor=252,
                       ax=None)
plt.show()

# 2.4 Calculate Optimal HRP Portfolios for Several Risk Measures
rms = ['vol', 'MV', 'MAD', 'GMD', 'MSV', 'FLPM', 'SLPM', 'VaR',
       'CVaR', 'TG', 'EVaR', 'WR', 'RG', 'CVRG', 'TGRG', 'MDD',
       'ADD', 'DaR', 'CDaR', 'EDaR', 'UCI', 'MDD_Rel',
       'ADD_Rel', 'DaR_Rel', 'CDaR_Rel', 'EDaR_Rel', 'UCI_Rel']

w_s = pd.DataFrame([])
for i in rms:
    w = port.optimization(model=model,
                          codependence=codependence,
                          rm=i,
                          rf=rf,
                          linkage=linkage,
                          max_k=max_k,
                          leaf_order=leaf_order)

    w_s = pd.concat([w_s, w], axis=1)

w_s.columns = rms
print(w_s)

# Plotting a comparison of assets weights for each portfolio
fig = plt.gcf()
fig.set_figwidth(16)
fig.set_figheight(8)
ax5 = fig.subplots(nrows=1, ncols=1)
w_s.plot(kind='bar', width=0.8, ax=ax)
plt.show()

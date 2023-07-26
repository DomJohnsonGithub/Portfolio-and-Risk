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

# HIERARCHIAL PORTFOLIOS WITH CUSTOM COVARIANCE #

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

# Load our custom estimates of input parameters
custom_cov = pd.read_excel('custom_posterior_cov.xlsx', engine='openpyxl', index_col=0)

# Plotting Assets Clusters
ax0 = rp.plot_dendrogram(returns=Y,
                         custom_cov=custom_cov,
                         codependence='custom_cov',
                         linkage='ward',
                         k=None,
                         max_k=10,
                         leaf_order=True,
                         ax=None)
plt.show()

ax1 = rp.plot_clusters(returns=Y,
                       custom_cov=custom_cov,
                       codependence='custom_cov',
                       linkage='ward',
                       k=None,
                       max_k=10,
                       leaf_order=True,
                       dendrogram=True,
                       # linecolor='tab:purple',
                       ax=None)

plt.show()

# 2. ESTIMATING HERC PORTFOLIO
# 2.1 Calculating the HERC portfolio
port = rp.HCPortfolio(returns=Y)

# Estimate optimal portfolio:

model = 'HERC'  # Could be HRP or HERC
codependence = 'custom_cov'  # Correlation matrix used to group assets in clusters
covariance = 'custom_cov'
rm = 'MV'  # Risk measure used, this time will be variance
rf = 0  # Risk-free rate
linkage = 'ward'  # Linkage method used to build clusters
max_k = 10  # Max number of clusters used in two difference gap statistic
leaf_order = True  # Consider optimal order of leafs in dendrogram

w = port.optimization(model=model,
                      codependence=codependence,
                      covariance=covariance,
                      custom_cov=custom_cov,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order)

print(w.T)

# 2.2 Plotting portfolio composition
ax2 = rp.plot_pie(w=w,
                  title='HERC Naive Risk Parity with custom covariance',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=8,
                  width=10,
                  ax=None)
plt.show()

# 2.3 Plotting Risk Contribution
ax3 = rp.plot_risk_con(w=w,
                       cov=custom_cov,
                       returns=Y,
                       rm=rm,
                       rf=0,
                       alpha=0.05,
                       color="tab:blue",
                       height=6,
                       width=10,
                       t_factor=252,
                       ax=None)
plt.show()

# 3. ESTIMATING HERC PORTFOLIO FOR SEVERAL RISK MEASURES
rms = ['vol', 'MV', 'MAD', 'MSV', 'FLPM', 'SLPM',
       'VaR', 'CVaR', 'EVaR', 'WR', 'MDD', 'ADD',
       'DaR', 'CDaR', 'EDaR', 'UCI', 'MDD_Rel', 'ADD_Rel',
       'DaR_Rel', 'CDaR_Rel', 'EDaR_Rel', 'UCI_Rel']

w_s = pd.DataFrame([])
for i in rms:
    w = port.optimization(model=model,
                          codependence=codependence,
                          covariance=covariance,
                          custom_cov=custom_cov,
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
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)
w_s.plot.bar(ax=ax)
plt.legend(loc='lower right')
plt.show()

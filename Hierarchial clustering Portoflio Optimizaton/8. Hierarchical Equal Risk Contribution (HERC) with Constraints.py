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

# HIERARCHIAL EQUAL RISK CONTRIBUTION (HERC) PORTFOLIO OPTIMIZATION WITH CONSTRAINTS #

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
                       # linecolor='tab:purple',
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

# Estimate optimal portfolio:=
model = 'HERC'  # Could be HRP, HERC or NCO
codependence = 'pearson'  # Correlation matrix used to group assets in clusters
rm = 'MV'  # Risk measure used, this time will be variance
rf = 0  # Risk-free rate
linkage = 'ward'  # Linkage method used to build clusters
max_k = 10  # Max number of clusters used in two difference gap statistic
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
ax2 = rp.plot_pie(w=w,
                  title='HERC Naive Risk Parity',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=8,
                  width=10,
                  ax=None)
plt.show()

# 2.3 Plotting Risk Contribution
mu = Y.mean()
cov = Y.cov()  # Covariance matrix
returns = Y  # Returns of the assets

ax3 = rp.plot_risk_con(w=w,
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

# 3. ESTIMATING HERC PORTFOLIO WITH CONSTRAINTS
asset_classes = {'Assets': ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
                            'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
                            'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA'],
                 'Industry': ['Consumer Discretionary', 'Consumer Discretionary',
                              'Consumer Discretionary', 'Consumer Staples',
                              'Consumer Staples', 'Energy', 'Financials',
                              'Financials', 'Financials', 'Financials',
                              'Health Care', 'Health Care', 'Industrials', 'Industrials',
                              'Industrials', 'Health care', 'Industrials',
                              'Information Technology', 'Information Technology',
                              'Materials', 'Telecommunications Services', 'Utilities',
                              'Utilities', 'Telecommunications Services', 'Financials']}

asset_classes = pd.DataFrame(asset_classes)
asset_classes = asset_classes.sort_values(by=['Assets'])

constraints = {'Disabled': [False, False, False, False, False],
               'Type': ['Assets', 'Assets', 'All Assets',
                        'Each asset in a class', 'Each asset in a class'],
               'Set': ['', '', '', 'Industry', 'Industry'],
               'Position': ['HPQ', 'PSA', '', 'Financials', 'Information Technology'],
               'Sign': ['>=', '<=', '<=', '<=', '<='],
               'Weight': [0.01, 0.05, 0.06, 0.04, 0.02]}

constraints = pd.DataFrame(constraints)
print(constraints)

# 3.2 Calculating the HERC portfolio with constraints
w_max, w_min = rp.hrp_constraints(constraints, asset_classes)

port.w_max = w_max
port.w_min = w_min

w_1 = port.optimization(model=model,
                        codependence=codependence,
                        rm=rm,
                        rf=rf,
                        linkage=linkage,
                        max_k=max_k,
                        leaf_order=leaf_order)
print(w_1.T)

# 3.3 Plotting portfolio composition
ax4 = rp.plot_pie(w=w_1,
                  title='HERC Naive Risk Parity with Contraints',
                  others=0.05,
                  nrow=25,
                  cmap="tab20",
                  height=8,
                  width=10,
                  ax=None)
plt.show()

# 3.4 Plotting Risk Contribution
mu = Y.mean()
cov = Y.cov()  # Covariance matrix
returns = Y  # Returns of the assets

ax5 = rp.plot_risk_con(w=w_1,
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

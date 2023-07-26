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

# 2. COMPARING LINKAGE METHODS USING DENDOGRAMS
linkages = ['single', 'complete', 'average', 'weighted',
            'centroid', 'median', 'ward', 'DBHT']

fig, ax = plt.subplots(len(linkages), 1, figsize=(12, 30))
ax = np.ravel(ax)

j = 0
for i in linkages:
    ax[j] = rp.plot_dendrogram(returns=Y,
                               codependence='pearson',
                               linkage=i,
                               k=None,
                               max_k=10,
                               leaf_order=True,
                               ax=ax[j])
    j += 1

plt.show()

# 3. COMPARING LINKAGE METHODS USING NETWORKS
linkages = ['ward', 'DBHT']

fig, ax = plt.subplots(len(linkages), 1, figsize=(12, 15))
ax = np.ravel(ax)

j = 0
for i in linkages:
    ax[j] = rp.plot_network(returns=Y,
                            codependence="pearson",
                            linkage=i,
                            k=None,
                            max_k=10,
                            leaf_order=True,
                            kind='spring',
                            seed=0,
                            ax=ax[j])
    j += 1

plt.show()

# 4. Comparing Networks Layouts
kinds =['spring','kamada','planar','circular']

fig, ax = plt.subplots(len(kinds), 1, figsize=(12, 30))
ax = np.ravel(ax)

j = 0
for i in kinds:
    ax[j] = rp.plot_network(returns=Y,
                            codependence="pearson",
                            linkage="DBHT",
                            k=None,
                            max_k=10,
                            leaf_order=True,
                            kind=i,
                            seed=0,
                            ax=ax[j])
    j += 1
plt.show()

# 5. CLUSTERS COMPONENTS
clusters = rp.assets_clusters(returns=Y,
                              codependence='pearson',
                              linkage='DBHT',
                              k=None,
                              max_k=10,
                              leaf_order=True)

print(clusters.sort_values(by='Clusters'))

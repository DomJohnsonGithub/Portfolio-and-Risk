import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab
from scipy.stats import norm
import warnings

sns.set_style("darkgrid")
warnings.filterwarnings("ignore")

df = yf.download("AMZN", "2020-01-01", "2022-01-01")
df = df[['Close']]
df['returns'] = df.Close.pct_change().dropna()

# VAR VARIANCE_COVARIANCE APPROACH

# mean and standard deviation of the daily returns
# Plot the normal curve against the daily returns
mean = np.mean(df['returns'])
std_dev = np.std(df['returns'])

df['returns'].hist(bins="rice", density=True, histtype='stepfilled', alpha=0.5)

x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
plt.plot(x, scipy.stats.norm.pdf(x, mean, std_dev), "r")
plt.show()

# Use confidence level, mean and standard deviation
# Calculate VaR using point percentile function
VaR_95 = norm.ppf(0.05, mean, std_dev)
VaR_99 = norm.ppf(0.01, mean, std_dev)
print(tabulate([['95%', VaR_95], ['99%', VaR_99]], headers=['Confidence Level', 'Value at Risk']))

# VAR HISTORICAL SIMULATION APPROACH
plt.hist(df.returns, bins="rice")
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate the VaR
VaR_90 = df['returns'].quantile(0.1)
VaR_95 = df['returns'].quantile(0.05)
VaR_99 = df['returns'].quantile(0.01)

print(tabulate([['90%', VaR_90], ['95%', VaR_95], ['99%', VaR_99]], headers=['Confidence Level', 'Value at Risk']))

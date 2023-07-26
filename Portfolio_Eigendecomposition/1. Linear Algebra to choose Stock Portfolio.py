import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from itertools import chain

# Linear Algebra can help choose your Stock Portfolio

start, end = datetime(2012, 1, 1), datetime(2022, 12, 31)
tickers = ["AAPL", "KLAC", "GOOG", "MSFT", "WDC", "AMZN"]
prices = yf.download(tickers, start=start, end=end)["Adj Close"]
print(prices)

returns = prices.copy().pct_change().dropna()
print(returns)

# The Eigenvalues of the Covariance Matrix
# - extract an eigenportfolio. To avoid the dreaded Market-correlated portfolio
# (we aren’t trying to build an index fund here!), let’s take a portfolio using the next eigenvector.

training_period = 756
train = returns.iloc[:-training_period, :].copy()
test = returns.iloc[-training_period:, :].copy()

covariance_matrix = train.cov()
D, S = np.linalg.eigh(covariance_matrix)

eigenportfolio_1 = S[:, -1] / np.sum(S[:, -1])  # Normalize to sum to 1
eigenportfolio_2 = S[:, -2] / np.sum(S[:, -2])  # Normalize to sum to 1
print(eigenportfolio_1)
print(eigenportfolio_2)

# Set up porfolios
eigenportfolio = pd.DataFrame(eigenportfolio_1, columns=["Investment Weight"], index=tickers)
eigenportfolio2 = pd.DataFrame(eigenportfolio_2, columns=["Investment Weight"], index=tickers)

fig = plt.figure()
ax = plt.subplot(121)
eigenportfolio.plot(kind="bar", ax=ax, legend=False)
plt.title("Max E.V. Eigenportfolio")
ax = plt.subplot(122)
eigenportfolio2.plot(kind="bar", ax=ax, legend=False)
plt.title("2nd E.V. Eigenportfolio")
plt.show()


def cumulative_returns(sample, weights):
    return (((1 + sample).cumprod(axis=0)) - 1).dot(weights)


in_sample_ind = np.arange(0, (returns.shape[0] - training_period + 1))
out_sample_ind = np.arange((returns.shape[0] - training_period + 1), returns.shape[0])

cumulative_returns = cumulative_returns(returns, eigenportfolio).values
cumulative_returns = pd.Series(list(chain.from_iterable(cumulative_returns)))
print(cumulative_returns)

fig = plt.figure(figsize=(10, 4))
ax = plt.subplot(121)
ax.plot(cumulative_returns[:-training_period], c="black")
ax.plot(cumulative_returns[-training_period:], c="r")
plt.title("Eigenportfolio")

ax = plt.subplot(122)
plt.plot((((1 + returns.loc[:, "AAPL"]).cumprod(axis=0)) - 1))
plt.title("AAPL")

plt.tight_layout()
plt.grid(True)
plt.show()

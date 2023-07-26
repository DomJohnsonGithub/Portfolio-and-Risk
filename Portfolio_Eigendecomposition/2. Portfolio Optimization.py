import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from itertools import chain

# MARKOWITZ PORTFOLIO THEORY
# - optimization produces the Minimum Variance Portfolio (MVP)

# IMPORT DATA
start, end = datetime(2012, 1, 1), datetime(2013, 12, 31)
tickers = ["AAPL", "KLAC", "GOOG", "MSFT", "WDC", "AMZN"]
prices = yf.download(tickers, start=start, end=end)["Adj Close"]
# print(prices)

# Get Returns with pct_change method
returns = prices.copy().pct_change().dropna()
print(returns)

# GET IN SAMPLE AND OUT OF SAMPLE DATA
training_period = 21  # 21 days is 1 trading month
train = returns.iloc[:-training_period, :].copy();
test = returns.iloc[-training_period:, :].copy()
print(train.shape)
print(test.shape)

# Calculate the inverse of the covariance matrix using a pseudo-inverse (incase it is ill-conditioned or singular)
# and then construct the minimum variance strategy weights.

covariance_matrix = train.cov().values
inv_cov_mat = np.linalg.inv(covariance_matrix)

ones = np.ones(len(inv_cov_mat))
inv_dot_ones = np.dot(inv_cov_mat, ones)
min_var_weights = inv_dot_ones / np.dot(inv_dot_ones, ones)

min_var_portfolio = pd.DataFrame(min_var_weights, columns=["Investment Weight"], index=tickers)
print("Minimum Variance Portfolio:")
print(min_var_portfolio)

# COMPARE IT TO THE LARGEST EIGENVALUE'S EIGENPORTFOLIO
# largest eigenvalue eigenportfolio
D, S = np.linalg.eigh(covariance_matrix)
eigenportfolio_1 = S[:, -1] / np.sum(S[:, -1])  # noramalize to sum to 1
eigenportfolio_largest = pd.DataFrame(data=eigenportfolio_1, columns=["Investment Weight"], index=tickers)

print("Eigenportfolio Largest:")
print(eigenportfolio_largest)

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(121)
min_var_portfolio.plot(kind="bar", ax=ax, legend=False)
plt.title("Min Var Portfolio")
ax = plt.subplot(122)
eigenportfolio_largest.plot(kind="bar", ax=ax, legend=False)
plt.title("Max E.V. Eigenportfolio")
plt.show()

# - test the algo on returns data and compare it to the eigenprofile related to the largest eigenvalue
def cum_rets(sample, weights):
    return (((1 + sample.values).cumprod(axis=0)) - 1).dot(weights)


cumulative_returns = cum_rets(returns, min_var_portfolio)
cumulative_returns_largest = cum_rets(returns, eigenportfolio_largest)

cumulative_returns = pd.Series(list(chain(*cumulative_returns)))
cumulative_returns_largest = pd.Series(list(chain(*cumulative_returns_largest)))

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(cumulative_returns[:-training_period], c="b")
axes[0].plot(cumulative_returns[-training_period:], c="r")
axes[0].set_title("Minimum Variance Portfolio")
axes[0].grid(True)
axes[1].plot(cumulative_returns_largest[:-training_period], c="b")
axes[1].plot(cumulative_returns_largest[-training_period:], c="r")
axes[1].set_title("Eigenportfolio (largest)")
axes[1].grid(True)
plt.show()

# Variance = w^T Sigma w
largest_var = np.dot(eigenportfolio_1, np.dot(covariance_matrix, eigenportfolio_1))
min_var = np.dot(min_var_weights, np.dot(covariance_matrix, min_var_weights))
print("The eigenportfolio has a variance of %f.\nThe minimum variance portfolio has a variance of %f" %
      (largest_var, min_var))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
import statsmodels.api as sm
from statsmodels import regression


yf.pdr_override()

df1 = pdr.get_data_yahoo("GOOG", start="2017-01-01", end="2017-11-30")
df2 = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-11-30")

return_goog = df1.Close.pct_change()[1:]
return_spy = df2.Close.pct_change()[1:]

plt.figure(figsize=(20, 10))
return_goog.plot()
return_spy.plot()
plt.ylabel("Daily Return of GOOG and SPY")
plt.show()

X = return_spy.values
Y = return_goog.values


def linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()

    # We are removing the constant
    x = x[:, 1]
    return model.params[0], model.params[1]


alpha, beta = linreg(X, Y)
print('alpha: ' + str(alpha))
print('beta: ' + str(beta))

X2 = np.linspace(X.min(), X.max(), 100)
Y_hat = X2 * beta + alpha

plt.figure(figsize=(10, 7))
plt.scatter(X, Y, alpha=0.3)  # Plot the raw data
plt.xlabel("SPY Daily Return")
plt.ylabel("GOOG Daily Return")

plt.plot(X2, Y_hat, 'r', alpha=0.9)
plt.show()

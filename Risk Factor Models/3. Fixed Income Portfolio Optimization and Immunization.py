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

# BOND PORTFOLIO OPTIMIZATION AND IMMUNIZATION #

# 1. IMPORTING DATA

# Interest Rates Data
kr = pd.read_excel('KeyRates.xlsx', engine='openpyxl', index_col=0, header=0) / 100

# Prices  Data
assets = pd.read_excel('Assets.xlsx', engine='openpyxl', index_col=0, header=0)

# Find common dates
a = pd.merge(left=assets, right=kr, how='inner', on='Date')
dates = a.index

# Calculate interest rates returns
kr_returns = kr.loc[dates, :].sort_index().diff().dropna()
kr_returns.sort_index(ascending=False, inplace=True)

# List of instruments
equity = ['APA', 'CMCSA', 'CNP', 'HPQ', 'PSA', 'SEE', 'ZION']
bonds = ['PEP11900D031', 'PEP13000D012', 'PEP13000M088',
         'PEP23900M103', 'PEP70101M530', 'PEP70101M571',
         'PEP70310M156']

# Calculate assets returns
assets_returns = assets.loc[dates, equity + bonds]
assets_returns = assets_returns.sort_index().pct_change().dropna()
assets_returns.sort_index(ascending=False, inplace=True)

# Show tables
print(kr_returns.head().style.format("{:.4%}"))
print(assets_returns.head().style.format("{:.4%}"))

durations = pd.read_excel('durations.xlsx', index_col=0, header=0)
convexity = pd.read_excel('convexity.xlsx', index_col=0, header=0)

print('Durations Matrix')
print(durations.head().style.format("{:.4f}").background_gradient(cmap='YlGn'))
print('')
print('Convexity Matrix')
print(convexity.head().style.format("{:.4f}").background_gradient(cmap='YlGn'))

# 2. ESTIMATING MEAN VARIANCE PORTFOLIOS
# 2.1 Building the loadings matrix and risk factors returns
loadings = pd.concat([-1.0 * durations, 0.5 * convexity], axis=1)
loadings.style.format("{:.4f}").background_gradient(cmap='YlGn')
print(loadings)

# Building the risk factors returns matrix
kr_returns_2 = kr_returns ** 2
cols = loadings.columns

X = pd.concat([kr_returns, kr_returns_2], axis=1)
X.columns = cols
print(X.head().style.format("{:.4%}"))

# Building the asset returns matrix
Y = assets_returns[loadings.index]
print(Y.head())

# 2.2 Calculating the portfolio that maximizes Sharpe ratio
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Select method and estimate input parameters:
method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

port.factors = X
port.factors_stats(method_mu=method_mu, method_cov=method_cov, d=0.94, B=loadings)

# Estimate optimal portfolio:
model = 'FM'  # Factor Model
rm = 'MV'  # Risk measure used, this time will be variance
obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = False  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk-free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
w.style.format("{:.4%}").background_gradient(cmap='YlGn')
print(w)

# 3. OPTIMIZATION WITH KEY RATE DURATION CONSTRAINTS
# 3.1 Statistics of Risk Factors
table = pd.concat([loadings.min(), loadings.max()], axis=1)
table.columns = ['min', 'max']
print(table.iloc[:9,:].style.format("{:.4f}").background_gradient(cmap='YlGn'))
print(X.iloc[:,:9].corr().style.format("{:.4f}").background_gradient(cmap='YlGn'))

# 3.2 Creating Constraints on Key Rate Durations
# limit on the maximum duration that the portfolio can reach
# the key rate durations of portfolio for 1800, 3600 and 7200 days will be lower than -2, -2 and -3
constraints = {'Disabled': [False, False, False],
               'Factor': ['R 1800', 'R 3600', 'R 7200'],
               'Sign': ['<=', '<=', '<='],
               'Value': [-2, -2, -3],
               'Relative Factor': ['', '', '']}

constraints = pd.DataFrame(constraints)
print(constraints)

# 3.3 Estimating Optimum Portfolio with Key Rate Durations Constraints
C, D = rp.factors_constraints(constraints, loadings)

port.ainequality = C
port.binequality = D

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
w.style.format("{:.4%}").background_gradient(cmap='YlGn')
print(w)

# Calculating portfolio sensitivities for each risk factor
d_ = np.matrix(loadings).T * np.matrix(w)
d_ = pd.DataFrame(d_, index=loadings.columns, columns=['Values'])
print(d_.style.format("{:.4f}").background_gradient(cmap='YlGn'))

# 4. ESTIMATING MEAN VARIANCE PORTFOLIOS
# 4.1 Building the loadings matrix and risk factors returns
# Other approach for removing bond returns from factors matrix
cols = [col for col in assets_returns.columns if col not in loadings.index]

X = pd.concat([assets_returns[cols], X], axis=1)
print(X.head())

# Building the asset returns matrix
Y = pd.concat([assets_returns[cols], Y], axis=1)
print(Y.head())

# Building The Loadings Matrix
a = np.identity(len(cols))
a = pd.DataFrame(a, index=cols, columns=cols)
loadings = pd.concat([a, loadings], axis = 1)
loadings.fillna(0, inplace=True)
loadings.style.format("{:.4f}").background_gradient(cmap='YlGn')
print(loadings)

# 4.2 Calculating the portfolio that maximizes Sharpe ratio
port = rp.Portfolio(returns=Y)

# Select method and estimate input parameters:
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

port.factors = X
port.factors_stats(method_mu=method_mu, method_cov=method_cov, d=0.94, B=loadings)

# Estimate optimal portfolio:
model='FM' # Factor Model
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = False # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk-free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w)

# 5. OPTIMIZATION OF EQUITY AND BOND PORTFOLIO WITH KEY RATE DURATION CONSTRAINTS
# build immunized portfolios using duration matching (only in this example) and convexity matching

# 5.1 Statistics of Risk Factors
table = pd.concat([loadings.min(), loadings.max()], axis=1)
table.columns = ['min', 'max']
print(table.iloc[:16,:].style.format("{:.4f}").background_gradient(cmap='YlGn'))
print(X.iloc[:,:16].corr().style.format("{:.4f}").background_gradient(cmap='YlGn'))

# 5.2 Creating Constraints on Key Rate Durations
constraints = {'Disabled': [False, False, False],
               'Factor': ['R 1800', 'R 3600', 'R 7200'],
               'Sign': ['<=', '<=', '<='],
               'Value': [-2, -2, -3],
               'Relative Factor': ['', '', '']}

constraints = pd.DataFrame(constraints)
print(constraints)

# 5.3 Estimating Optimum Portfolio with Key Rate Durations Constraints
C, D = rp.factors_constraints(constraints, loadings)

port.ainequality = C
port.binequality = D

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
print(w.style.format("{:.4%}").background_gradient(cmap='YlGn'))

# Calculating portfolio sensitivities for each risk factor
d_ = np.matrix(loadings).T * np.matrix(w)
d_ = pd.DataFrame(d_, index=loadings.columns, columns=['Values'])
print(d_.style.format("{:.4f}").background_gradient(cmap='YlGn'))
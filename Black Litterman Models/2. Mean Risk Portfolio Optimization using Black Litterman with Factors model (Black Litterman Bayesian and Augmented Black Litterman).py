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

# BLACK LITTERMAN WITH FACTOR MODELS MEAN RISK OPTIMIZATION #

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
factors = ['MTUM', 'QUAL', 'SIZE', 'USMV', 'VLUE']

# Calculate assets returns
assets_returns = assets.loc[dates, equity + bonds]
assets_returns = assets_returns.sort_index().pct_change().dropna()
assets_returns.sort_index(ascending=False, inplace=True)

# Calculate factors returns
factors_returns = assets.loc[dates, factors]
factors_returns = factors_returns.sort_index().pct_change().dropna()
factors_returns.sort_index(ascending=False, inplace=True)

# Show tables
print(kr_returns.head().style.format("{:.4%}"))
print(assets_returns.head().style.format("{:.4%}"))

# Uploading Duration and Convexity Matrixes
durations = pd.read_excel('durations.xlsx', index_col=0, header=0)
convexity = pd.read_excel('convexity.xlsx', index_col=0, header=0)

print('Durations Matrix')
print(durations.head().style.format("{:.4f}").background_gradient(cmap='YlGn'))
print('')
print('Convexities Matrix')
print(convexity.head().style.format("{:.4f}").background_gradient(cmap='YlGn'))

# 2. ESTIMATING BLACK LITTERMAN WITH FACTORS FOR FIXED INCOME PORTFOLIOS
# 2.1 Building the loadings matrix and risk factors returns
loadings = pd.concat([-1.0 * durations, 0.5 * convexity], axis=1)
print(loadings)

# Building the risk factors returns matrix
kr_returns_2 = kr_returns ** 2
cols = loadings.columns

X = pd.concat([kr_returns, kr_returns_2], axis=1)
X.columns = cols
print(X.head().style.format("{:.4%}"))

# Building the asset returns matrix
Y = assets_returns[bonds]

# 2.2 Building views on risk factors
# Showing annualized returns of Fixed Income Risk Factors
print((X.mean()*252))

# Building views on some Risk Factors
views = {'Disabled': [False, False, False],
        'Factor': ['R 10800','R 1800','R 3600'],
        'Sign': ['>=', '<=', '<='],
        'Value': [0.001, -0.001, -0.003],
        'Relative Factor': ['R 7200', '', '']}

views = pd.DataFrame(views)
print(views)

# Building views matrixes P_f and Q_f
P_f, Q_f = rp.factors_views(views, loadings, const=False)

print('Matrix of factors views P_f')
print(P_f)
print('\nMatrix of returns of factors views Q_f')
print(Q_f)

# 2.3 Building Portfolios with mean vector and covariance matrix from Black Litterman with Factors
port = rp.Portfolio(returns=Y)

# Select method and estimate input parameters:
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

port.factors = X
port.factors_stats(method_mu=method_mu, method_cov=method_cov, d=0.94, B=loadings)

# Calculating optimum portfolios using Mean Vector and
# Covariance Matrix of Black Litterman with Factors
port.alpha = 0.05
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = False # False: BL covariance and risk factors scenarios
         # True: historical covariance and scenarios
         # 2: risk factors covariance and scenarios
rf = 0 # Risk free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

w_fm = port.optimization(model='FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

# Estimate Portfolio weights using Black Litterman Bayesian Model:
port.blfactors_stats(flavor='BLB',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f/252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=False,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

w_blb = port.optimization(model='BL_FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

# Estimate Portfolio weights using Augmented Black Litterman Model:
port.blfactors_stats(flavor='ABL',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f/252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=False,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

w_abl = port.optimization(model='BL_FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

ws = pd.concat([w_fm, w_blb, w_abl], axis=1)
ws.columns = ['Pure Factors', 'Bayesian BL', 'Augmented BL']
print(ws)

# ^^ problem here, covariance matrix is not a positive definite matrix
# the weights that we will get are highly concetrated in few assets
# Other approach is using the mean vector estimated with Black Litterman with Factors and the covariance matrix that we get from historical returns or a factor model

# 2.4 Building Portfolios with mean vector from Black Litterman with Factors
# Calculating optimum portfolios using only Mean Vector
# of Black Litterman with Factors and Factor Covariance Matrix
hist = 2 # False: BL covariance and risk factors scenarios
             # True: historical covariance and scenarios
             # 2: risk factors covariance and scenarios (Only in BL_FM)

# Estimate Portfolio weights using Black Litterman Bayesian Model:
port.blfactors_stats(flavor='BLB',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f/252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=False,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

w_blb = port.optimization(model='BL_FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

# Estimate Portfolio weights using Augmented Black Litterman Model:
port.blfactors_stats(flavor='ABL',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f/252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=False,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

w_abl = port.optimization(model='BL_FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

ws = pd.concat([w_fm, w_blb, w_abl], axis=1)
ws.columns = ['Pure Factors', 'Bayesian BL', 'Augmented BL']
print(ws)

# ^^ weights that we get are more diversified. Also, we can see that the
# Augmented Black Litterman creates more diversified portfolios than the Bayesian Black Litterman

# 3. ESTIMATING MEAN VARIANCE PORTFOLIO FOR EQUITY AND FIXED INCOME PORTFOLIO
# 3.1 Building the loadings matrix and risk factors returns
B = rp.loadings_matrix(factors_returns, assets_returns[equity])
print(B)

# Building the asset returns matrix
Y = pd.concat([assets_returns[equity], Y], axis=1)
print(Y)

X = pd.concat([factors_returns, X], axis=1)
print(X.head())

# Building The Loadings Matrix
loadings = pd.concat([B, loadings], axis = 1)
loadings.fillna(0, inplace=True)
print(loadings)

# 3.2 Building views on risk factors
# Showing annualized returns of Equity Risk Factors
print(factors_returns.mean()*252)

# Building views on some Risk Factors
views = {'Disabled': [False, False, False, False, False, False],
        'Factor': ['MTUM','USMV','SIZE','R 10800','R 1800','R 3600'],
        'Sign': ['>=', '>=', '>=', '>=', '<=', '<='],
        'Value': [0.02, 0.09, 0.12, 0.001, -0.001, -0.003],
        'Relative Factor': ['VLUE', '', '','R 90', '', '']}
views = pd.DataFrame(views)
print(views)

# Building views matrixes P_f and Q_f
P_f, Q_f = rp.factors_views(views, loadings, const=True)

print('Matrix of factors views P_f')
print(P_f)
print('\nMatrix of returns of factors views Q_f')
print(Q_f)

# 3.3 Building Portfolios with mean vector and covariance matrix from Black Litterman with Factors
port = rp.Portfolio(returns=Y)

# Calculating optimum portfolio
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu,
                  method_cov=method_cov,
                  d=0.94)

port.factors = X
port.factors_stats(method_mu=method_mu,
                   method_cov=method_cov,
                   d=0.94,
                   B=loadings,
                   dict_risk=dict(const=True))

port.alpha = 0.05
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = False # False: BL covariance and risk factors scenarios
             # True: historical covariance and scenarios
             # 2: risk factors covariance and scenarios
rf = 0 # Risk free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

w_fm = port.optimization(model='FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

# Estimate Portfolio weights using Black Litterman Bayesian Model:
port.blfactors_stats(flavor='BLB',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f/252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=True,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

w_blb = port.optimization(model='BL_FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

# Estimate Portfolio weights using Augmented Black Litterman Model:
port.blfactors_stats(flavor='ABL',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f/252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=True,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

w_abl = port.optimization(model='BL_FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

ws = pd.concat([w_fm, w_blb, w_abl], axis=1)
ws.columns = ['Pure Factors', 'Bayesian BL', 'Augmented BL']
print(ws)

# 3.4 Building Portfolios with mean vector from Black Litterman with Factors
# same prob as before with 3.3
hist = 2 # False: BL covariance and risk factors scenarios
         # True: historical covariance and scenarios
         # 2: risk factors covariance and scenarios (Only in BL_FM)

# Estimate Portfolio weights using Black Litterman Bayesian Model:
port.blfactors_stats(flavor='BLB',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f/252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=True,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

w_blb = port.optimization(model='BL_FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

# Estimate Portfolio weights using Augmented Black Litterman Model:
port.blfactors_stats(flavor='ABL',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f/252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=True,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

w_abl = port.optimization(model='BL_FM', rm=rm, obj=obj, rf=rf, l=l, hist=hist)

ws = pd.concat([w_fm, w_blb, w_abl], axis=1)
ws.columns = ['Pure Factors', 'Bayesian BL', 'Augmented BL']
print(ws)

# 4. ESTIMATING BLACK LITTERMAN WITH FACTORS MEAN RISK PORTFOLIOS
# 4.1 Calculate Black Litterman Bayesian Portfolios for Several Risk Measures
rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
       'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']

w_s = pd.DataFrame([])
port.alpha = 0.05

port.blfactors_stats(flavor='BLB',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f / 252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=True,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

model = 'BL_FM'
obj = 'Sharpe'
for i in rms:
    if i == 'MV':
        hist = 2
    else:
        hist = True
    w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
    w_s = pd.concat([w_s, w], axis=1)

w_s.columns = rms
print(w_s)

# Plotting a comparison of assets weights for each portfolio
fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)
w_s.plot.bar(ax=ax)
plt.show()

# 4.2 Calculate Augmented Black Litterman Portfolios for Several Risk Measures
rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
       'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']

w_s = pd.DataFrame([])
port.alpha = 0.05
port.blfactors_stats(flavor='ABL',
                     B=loadings,
                     P_f=P_f,
                     Q_f=Q_f / 252,
                     rf=0,
                     delta=None,
                     eq=True,
                     const=True,
                     diag=False,
                     method_mu=method_mu,
                     method_cov=method_cov)

model = 'BL_FM'
obj = 'Sharpe'
for i in rms:
    if i == 'MV':
        hist = 2
    else:
        hist = True
    w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
    w_s = pd.concat([w_s, w], axis=1)

w_s.columns = rms
print(w_s)

# Plotting a comparison of assets weights for each portfolio
fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)
w_s.plot.bar(ax=ax)
plt.show()


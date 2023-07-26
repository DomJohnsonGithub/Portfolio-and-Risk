import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import riskfolio as rp
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer

sns.set_style("darkgrid")
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

# LARGE SCALE APPLICATION #

# 1. IMPORTING DATA
data = pd.read_csv("assets_data.csv", index_col='Dates')
Y = data.pct_change().dropna()  # returns

# 2. ESTIMATING MAX MEAN/RISK PORTFOLIOS FOR ALL RISK MEASURES2
port = rp.Portfolio(returns=Y)

method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Input model parameters:
port.solvers = ['MOSEK']  # Setting MOSEK as the default solver
# if you want to set some MOSEK params use this code as an example
# import mosek
# port.sol_params = {'MOSEK': {'mosek_params': {mosek.iparam.intpnt_solve_form: mosek.solveform.dual}}}

port.alpha = 0.05  # Significance level for CVaR, EVaR y CDaR
model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk-free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

# 2.1 Optimizing Process by Risk Measure
rms = ["MV", "MAD", "MSV", "FLPM", "SLPM", "CVaR",
       "EVaR", "WR", "MDD", "ADD", "CDaR", "UCI", "EDaR"]

w = {}
for rm in rms:
    start = timer()
    w[rm] = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
    end = timer()
    print(rm + ' takes ', timedelta(seconds=end-start))

# 2.2 Portfolio Weights
w_s = pd.DataFrame([])
for rm in rms:
    w_s = pd.concat([w_s, w[rm]], axis=1)
w_s.columns = rms
print(w_s)

# 2.3 In sample CAGR by Portfolio
a1 = datetime.strptime(data.index[0], '%d-%m-%Y')
a2 = datetime.strptime(data.index[-1], '%d-%m-%Y')
days = (a2-a1).days

cagr = {}
for rm in rms:
    a = np.prod(1 + Y @ w_s[rm]) ** (360/days)-1
    cagr[rm] = [a]

cagr = pd.DataFrame(cagr).T
cagr.columns = ['CAGR']
cagr.style.format("{:.2%}").background_gradient(cmap='RdYlGn')
print(cagr)

# 3. ESTIMATING MIN RISK PORTfOLIOS FOR ALL RISK MEASURES
# 3.1 Optimizing Process by Risk Measure
rms = ["MV", "MAD", "MSV", "FLPM", "SLPM", "CVaR",
       "EVaR", "WR", "MDD", "ADD", "CDaR", "UCI", "EDaR"]

w_min = {}
obj = 'MinRisk'
for rm in rms:
    start = timer()
    w_min[rm] = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
    end = timer()
    print(rm + ' takes ', timedelta(seconds=end-start))

# 3.2 Portfolio Weights
w_min_s = pd.DataFrame([])
for rm in rms:
    w_min_s = pd.concat([w_min_s, w_min[rm]], axis=1)
w_min_s.columns = rms
print(w_min_s)

# 3.3 In sample CAGR by Portfolio
a1 = datetime.strptime(data.index[0], '%d-%m-%Y')
a2 = datetime.strptime(data.index[-1], '%d-%m-%Y')
days = (a2-a1).days

min_cagr = {}
for rm in rms:
    a = np.prod(1 + Y @ w_min_s[rm]) ** (360/days)-1
    min_cagr[rm] = [a]

min_cagr = pd.DataFrame(min_cagr).T
min_cagr.columns = ['CAGR']

min_cagr.style.format("{:.2%}").background_gradient(cmap='RdYlGn')
print(min_cagr)

# 4. ESTIMATING RISK PARITY PORTFOLIOS FOR ALL RISK MEASURESES
# 4.1 Optimizing Process by Risk Measure
rms = ["MV", "MAD", "MSV", "FLPM", "SLPM",
       "CVaR", "EVaR", "CDaR", "UCI", "EDaR"]

b = None  # Risk contribution constraints vector, when None is equally risk per asset

w_rp = {}
for rm in rms:
    start = timer()

    w_rp[rm] = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
    end = timer()
    print(rm + ' takes ', timedelta(seconds=end - start))

# 4.2 Portfolio Weights
w_rp_s = pd.DataFrame([])
for rm in rms:
    w_rp_s = pd.concat([w_rp_s, w_rp[rm]], axis=1)

w_rp_s.columns = rms
print(w_rp_s)

# 4.3 In sample CAGR
rp_cagr = {}
for rm in rms:
    a = np.prod(1 + Y @ w_rp_s[rm]) ** (360/days)-1
    rp_cagr[rm] = [a]

rp_cagr = pd.DataFrame(rp_cagr).T
rp_cagr.columns = ['CAGR']

rp_cagr.style.format("{:.2%}").background_gradient(cmap='RdYlGn')
print(rp_cagr)









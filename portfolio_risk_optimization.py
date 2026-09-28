"""
Bank Analytics Project 3: Financial Portfolio Risk Optimization & Stress-Testing Engine
Author: Vivek Kumar
Techniques: Modern Portfolio Theory (Markowitz Mean-Variance), Monte Carlo Simulation, Value at Risk (VaR), Sharpe Ratio
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def run_portfolio_risk_pipeline():
    print("=== BANK ANALYTICS: PORTFOLIO RISK & QUANTITATIVE ASSET ALLOCATION ===")
    
    # 1. Simulate Historical Daily Returns for 5 Core Financial Asset Classes
    np.random.seed(42)
    assets = ['Equities_Global', 'Emerging_Markets', 'Govt_Bonds', 'Corp_Credit', 'Commodities_Gold']
    n_days = 504 # 2 trading years
    
    daily_returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.00045, 0.00055, 0.00015, 0.00025, 0.00030],
            cov=[
                [0.00015, 0.00012, -0.00003, 0.00004, 0.00002],
                [0.00012, 0.00025, -0.00004, 0.00006, 0.00003],
                [-0.00003, -0.00004, 0.00004, 0.00002, 0.00001],
                [0.00004, 0.00006, 0.00002, 0.00005, 0.00002],
                [0.00002, 0.00003, 0.00001, 0.00002, 0.00012]
            ],
            size=n_days
        ),
        columns=assets
    )
    
    annual_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    risk_free_rate = 0.045
    
    # 2. Maximum Sharpe Ratio Optimization
    def neg_sharpe(weights):
        p_ret = np.dot(weights, annual_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - risk_free_rate) / p_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.05, 0.40) for _ in range(len(assets)))
    init_weights = np.array([0.2] * len(assets))
    
    opt_result = minimize(neg_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = opt_result.x
    
    opt_ret = np.dot(opt_weights, annual_returns)
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    opt_sharpe = (opt_ret - risk_free_rate) / opt_vol
    
    print("\nOptimal Asset Allocation (Max Sharpe Ratio):")
    for asset, w in zip(assets, opt_weights):
        print(f"  - {asset:18s}: {w * 100:.2f}%")
        
    print(f"\nExpected Annual Return : {opt_ret * 100:.2f}%")
    print(f"Portfolio Volatility   : {opt_vol * 100:.2f}%")
    print(f"Optimal Sharpe Ratio   : {opt_sharpe:.4f}")
    
    # 3. Monte Carlo Simulation for Value at Risk (VaR)
    n_simulations = 10000
    portfolio_daily_rets = np.dot(daily_returns, opt_weights)
    simulated_returns = np.random.normal(portfolio_daily_rets.mean(), portfolio_daily_rets.std(), n_simulations)
    
    var_95 = np.percentile(simulated_returns, 5)
    var_99 = np.percentile(simulated_returns, 1)
    cvar_95 = simulated_returns[simulated_returns <= var_95].mean()
    
    print(f"\n1-Day Value at Risk (VaR 95%): {var_95 * 100:.2f}%")
    print(f"1-Day Value at Risk (VaR 99%): {var_99 * 100:.2f}%")
    print(f"Conditional VaR (CVaR 95%)   : {cvar_95 * 100:.2f}%")
    print("\nPortfolio Risk Optimization pipeline executed successfully.")

if __name__ == '__main__':
    run_portfolio_risk_pipeline()

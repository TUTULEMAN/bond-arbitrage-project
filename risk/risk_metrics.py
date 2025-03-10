"""
This script computes risk metrics for the bond arbitrage strategy.
Metrics include:
  - Value at Risk (VaR) via historical simulation
  - Conditional VaR (CVaR) / Expected Shortfall
  - Maximum Drawdown from the equity curve

It also plots the equity curve along with the drawdown for visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, cornishfisher

class StrategyRiskAnalyzer:
    """Risk analysis tailored to bond arbitrage strategies"""
    
    def __init__(self, trades, initial_capital=1e6):
        """
        Parameters:
            trades (list): List of trade dictionaries from backtest_strategy
            initial_capital (float): Starting capital for risk calculations
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self._process_trades()
        
    def _process_trades(self):
        """Convert trades to DataFrame and calculate P&L metrics"""
        self.df = pd.DataFrame(self.trades)
        
        if not self.df.empty:
            self.df['cum_profit'] = self.df['profit'].cumsum()
            self.equity_curve = self.initial_capital + self.df['cum_profit']
            self.returns = self.df['profit'] / self.initial_capital
            
    def compute_strategy_var(self, method='historical', confidence_level=0.95):
        """
        Compute Value at Risk with multiple methods
        Methods: historical, gaussian, cornish-fisher
        """
        if method == 'historical':
            return self._historical_var(confidence_level)
        elif method == 'gaussian':
            return self._parametric_var(confidence_level)
        elif method == 'cornish-fisher':
            return self._cornish_fisher_var(confidence_level)
        else:
            raise ValueError("Invalid VaR method")

    def _historical_var(self, confidence_level):
        """Historical simulation VaR"""
        return np.percentile(self.returns, 100*(1 - confidence_level))

    def _parametric_var(self, confidence_level):
        """Gaussian parametric VaR"""
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        return mu + sigma * norm.ppf(1 - confidence_level)

    def _cornish_fisher_var(self, confidence_level):
        """Cornish-Fisher adjusted VaR"""
        skew = self._compute_skew()
        kurt = self._compute_kurtosis()
        z = norm.ppf(1 - confidence_level)
        z_cf = (z + (z**2 - 1)*skew/6 + (z**3 - 3*z)*kurt/24 -
                (2*z**3 - 5*z)*skew**2/36)
        return np.mean(self.returns) + np.std(self.returns) * z_cf

    def compute_strategy_cvar(self, confidence_level=0.95):
        """Conditional VaR for strategy returns"""
        var = self.compute_strategy_var('historical', confidence_level)
        return self.returns[self.returns <= var].mean()

    def compute_trade_drawdowns(self):
        """Drawdown analysis specific to trade sequence"""
        running_max = np.maximum.accumulate(self.equity_curve)
        drawdowns = (self.equity_curve - running_max) / running_max
        return {
            'max_drawdown': drawdowns.min(),
            'drawdown_duration': self._compute_drawdown_duration(drawdowns),
            'recovery_time': self._compute_recovery_time(drawdowns)
        }

    def _compute_drawdown_duration(self, drawdowns):
        """Average time spent in drawdown"""
        in_drawdown = drawdowns < 0
        return in_drawdown.mean() if len(drawdowns) > 0 else 0

    def _compute_recovery_time(self, drawdowns):
        """Average time to recover from drawdowns"""
        recoveries = np.diff((drawdowns < 0).astype(int)) == 1
        return recoveries.mean() if len(recoveries) > 0 else 0

    def plot_strategy_risk(self):
        """Enhanced visualization with trade context"""
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Equity curve with drawdowns
        ax[0].plot(self.equity_curve, label='Equity')
        ax[0].plot(self.equity_curve / np.maximum.accumulate(self.equity_curve) - 1, 
                 label='Drawdown', color='red')
        ax[0].set_ylabel('Equity / Drawdown')
        ax[0].legend()
        
        # Trade outcomes
        wins = self.df[self.df['profit'] > 0]
        losses = self.df[self.df['profit'] <= 0]
        ax[1].vlines(wins.index, 0, wins['profit'], color='green', alpha=0.3)
        ax[1].vlines(losses.index, 0, losses['profit'], color='red', alpha=0.3)
        ax[1].set_ylabel('Trade P&L')
        ax[1].set_xlabel('Trade Sequence')
        
        plt.suptitle('Strategy Risk Profile')
        plt.tight_layout()
        return fig

    def generate_risk_report(self):
        """Comprehensive risk summary"""
        return {
            'var_95_historical': self.compute_strategy_var('historical', 0.95),
            'var_95_cornishfisher': self.compute_strategy_var('cornish-fisher', 0.95),
            'cvar_95': self.compute_strategy_cvar(0.95),
            'max_drawdown': self.compute_trade_drawdowns()['max_drawdown'],
            'profit_factor': self.df['profit'].clip(lower=0).sum() / 
                            -self.df['profit'].clip(upper=0).sum(),
            'win_rate': (self.df['profit'] > 0).mean()
        }

def main():
    # Example integration with backtest results
    from src.strategies.bond_arb_strategy import backtest_strategy
    from src.utils.data_loader import load_processed_data
    
    # Load market data from project pipeline
    spread_data = load_processed_data('bond_spread_series')
    
    # Run backtest (using existing strategy)
    trades = backtest_strategy(spread=spread_data.values)
    
    # Compute risk metrics
    analyzer = StrategyRiskAnalyzer(trades)
    print("\nStrategy Risk Report:")
    print(pd.Series(analyzer.generate_risk_report()).to_string())
    
    # Visualize
    analyzer.plot_strategy_risk()
    plt.show()

if __name__ == "__main__":
    main()
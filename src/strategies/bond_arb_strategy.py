# src/strategies/bond_arb_strategy.py
import numpy as np

class BayesianBondArbStrategy:
    """
    Production-ready Bayesian bond arbitrage signal generator.
    Contains only live trading logic, no backtesting code.
    """
    
    def __init__(self, window_size=40, prior_window=120):
        self.window_size = window_size
        self.prior_window = prior_window
        self.position = None
        self.history = []  # For live state tracking
        
    def update_parameters(self, new_spread_value):
        """
        Core Bayesian update logic. Called in live trading.
        Returns: (z_score, mu_post, sigma_post)
        """
        self.history.append(new_spread_value)
        
        if len(self.history) < self.window_size + 2:
            return None, None, None  # Warm-up period
            
        # Adaptive prior calculation
        prior_data = self._get_prior_data()
        mu0 = np.mean(prior_data)
        tau2 = np.var(prior_data)
        
        # Posterior calculation
        window_data = self._get_window_data()
        n = len(window_data)
        X_bar = np.mean(window_data)
        window_var = max(np.var(window_data, ddof=1), 1e-8)
        
        sigma_post2 = 1.0 / (n/window_var + 1/tau2)
        mu_post = sigma_post2 * (n*X_bar/window_var + mu0/tau2)
        sigma_post = np.sqrt(sigma_post2)
        
        z_score = (new_spread_value - mu_post) / sigma_post
        
        return z_score, mu_post, sigma_post
    
    def _get_prior_data(self):
        """Get adaptive prior training window"""
        start = max(0, len(self.history) - self.prior_window - self.window_size)
        end = len(self.history) - self.window_size
        return self.history[start:end]
    
    def _get_window_data(self):
        """Get current observation window"""
        return self.history[-self.window_size:]
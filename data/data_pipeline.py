"""
This script simulates an Ornstein-Uhlenbeck process with jumps (OUJ) for the bond arbitrage strategy.
It generates the spread data, saves it to a CSV file in data/processed, and displays a plot.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from datetime import datetime
import yaml  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OUJSimulator:
    """
    Ornstein-Uhlenbeck Jump Diffusion simulator with enhanced features:
    - Multiple volatility regimes
    - Realistic trading calendar
    - Microstructure noise
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self._validate_config()

    def _load_config(self, config_path: str) -> Dict:
        """Load simulation parameters from YAML file"""
        default_config = {
            'process': {
                'T': 2000,
                'theta': 0.1,
                'mu': 0.0,
                'base_sigma': 0.1,
                'jump_intensity': 0.02,
                'jump_dist': {'type': 'normal', 'mean': 0.0, 'std': 0.25},
                'regimes': [
                    {'start': 500, 'end': 1500, 'sigma_multiplier': 2.0}
                ]
            },
            'output': {
                'directory': 'data/processed',
                'version_format': 'ouj_{date}_v{version}',
                'add_timestamp': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f)
        return default_config

    def _validate_config(self):
        """Ensure parameters are physically plausible"""
        params = self.config['process']
        if params['theta'] <= 0:
            raise ValueError("Mean reversion speed (theta) must be positive")
        if params['base_sigma'] <= 0:
            raise ValueError("Volatility (sigma) must be positive")
        if not 0 <= params['jump_intensity'] <= 1:
            raise ValueError("Jump intensity must be between 0 and 1")

    def simulate(self) -> Tuple[pd.DataFrame, dict]:
        """Generate OUJ process with regime switching and microstructure noise"""
        cfg = self.config['process']
        T = cfg['T']
        spread = np.zeros(T)
        spread[0] = cfg.get('initial_value', 0.0)
        
        # Vectorized regime volatility
        sigma = np.full(T, cfg['base_sigma'])
        for regime in cfg['regimes']:
            start, end = regime['start'], regime['end']
            sigma[start:end] *= regime['sigma_multiplier']
        
        # Vectorized simulation
        dt = 1.0 / 252  # Daily assuming 252 trading days
        rand = np.random.randn(T)
        jump_rand = np.random.rand(T)
        
        for t in range(1, T):
            # Core OU process
            drift = cfg['theta'] * (cfg['mu'] - spread[t-1]) * dt
            diffusion = sigma[t] * np.sqrt(dt) * rand[t]
            
            # Jump component
            if jump_rand[t] < cfg['jump_intensity']:
                if cfg['jump_dist']['type'] == 'normal':
                    jump = np.random.normal(cfg['jump_dist']['mean'], 
                                          cfg['jump_dist']['std'])
                else:
                    raise NotImplementedError("Other jump types not implemented")
            else:
                jump = 0
                
            # Add microstructure noise (bid-ask bounce)
            spread[t] = spread[t-1] + drift + diffusion + jump + np.random.normal(0, 0.01)
        
        # Create DataFrame with realistic timestamps
        dates = pd.date_range(end=datetime.today(), periods=T, freq='B')
        df = pd.DataFrame({
            'date': dates,
            'spread': spread,
            'regime_volatility': sigma
        })
        
        return df, self.config

    @staticmethod
    def save_output(df: pd.DataFrame, config: dict) -> str:
        """Save data with versioning and metadata"""
        output_dir = config['output']['directory']
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = config['output']['version_format'].format(
            date=datetime.now().strftime("%Y%m%d"),
            version=1
        )
        
        # Find next available version
        version = 1
        while os.path.exists(os.path.join(output_dir, f"{base_name}.csv")):
            version += 1
            base_name = base_name.split('_v')[0] + f"_v{version}"
        
        # Save data and config
        path = os.path.join(output_dir, f"{base_name}.csv")
        df.to_csv(path, index=False)
        
        with open(os.path.join(output_dir, f"{base_name}_config.yaml"), 'w') as f:
            yaml.dump(config, f)
            
        logger.info(f"Saved simulation data to {path}")
        return path

    @staticmethod
    def analyze(df: pd.DataFrame) -> plt.Figure:
        """Generate diagnostic plots"""
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        
        # Price series
        axs[0].plot(df['date'], df['spread'], lw=1)
        axs[0].set_title('Simulated Spread Series')
        
        # Volatility regimes
        axs[1].plot(df['date'], df['regime_volatility'], color='orange')
        axs[1].set_title('Volatility Regimes')
        
        # Statistical properties
        returns = df['spread'].diff().dropna()
        axs[2].hist(returns, bins=100, density=True, alpha=0.6)
        axs[2].set_title('Spread Returns Distribution')
        
        plt.tight_layout()
        return fig

def main():
    try:
        # Initialize simulator with config
        simulator = OUJSimulator(config_path="data/raw/simulation_config.yaml")
        
        # Generate and validate data
        df, config = simulator.simulate()
        
        # Save output
        output_path = simulator.save_output(df, config)
        
        # Generate analysis report
        fig = simulator.analyze(df)
        fig.savefig(os.path.join(config['output']['directory'], 
                               'simulation_report.png'))
        
        logger.info("Data pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
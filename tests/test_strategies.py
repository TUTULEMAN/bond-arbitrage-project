"""
These tests check for:
  - No trade signals on a flat spread.
  - A valid long trade signal.
  - A valid short trade signal.
"""

import numpy as np
import pytest

# Import the strategy backtest function.
# This assumes that you have a function backtest_strategy defined in your production code.
from src.strategies.bond_arb_strategy import backtest_strategy

# Test parameters (these should match your strategy's expected defaults)
WINDOW_SIZE = 5
ENTRY_THRESHOLD = 2.0
LONG_EXIT_THRESHOLD = 2.0
SHORT_EXIT_THRESHOLD = -2.0
TRANSACTION_COST = 0.004

def test_no_trades():
    """
    When the spread is flat, the strategy should not trigger any trade.
    """
    # Flat spread: no variation, hence no entry signals.
    spread = np.zeros(50)
    
    trades = backtest_strategy(
        spread=spread,
        window_size=WINDOW_SIZE,
        entry_threshold=ENTRY_THRESHOLD,
        long_exit_threshold=LONG_EXIT_THRESHOLD,
        short_exit_threshold=SHORT_EXIT_THRESHOLD,
        transaction_cost=TRANSACTION_COST
    )
    # Expect no trades when there is no signal.
    assert trades == [] or len(trades) == 0

def test_long_trade():
    """
    Simulate a scenario that should trigger a long trade.
    Create a synthetic spread where after a period of near-zero values,
    the spread drops sharply triggering a long entry and later recovers to exit.
    """
    # Build a spread array:
    # - First 10 periods: flat at 0.
    # - One period: a sharp drop to -3 (should trigger long entry if Bayesian update yields near 0).
    # - Followed by a recovery to 1 for 10 periods (triggering long exit).
    spread = np.concatenate([np.zeros(10), np.array([-3.0]), np.ones(10)])
    
    trades = backtest_strategy(
        spread=spread,
        window_size=WINDOW_SIZE,
        entry_threshold=ENTRY_THRESHOLD,
        long_exit_threshold=LONG_EXIT_THRESHOLD,
        short_exit_threshold=SHORT_EXIT_THRESHOLD,
        transaction_cost=TRANSACTION_COST
    )
    
    # Check that at least one trade was executed.
    assert len(trades) >= 1, "Expected at least one long trade to be executed."
    
    # Verify that the trade(s) are long and have a computed profit.
    for trade in trades:
        assert trade['type'] == 'long', "Expected trade type to be 'long'."
        assert 'profit' in trade, "Trade dictionary should contain a 'profit' key."

def test_short_trade():
    """
    Simulate a scenario that should trigger a short trade.
    Create a synthetic spread where after a period of near-zero values,
    the spread spikes sharply triggering a short entry and later recovers to exit.
    """
    # Build a spread array:
    # - First 10 periods: flat at 0.
    # - One period: a sharp spike to 3 (should trigger short entry if Bayesian update yields near 0).
    # - Followed by a recovery to -1 for 10 periods (triggering short exit).
    spread = np.concatenate([np.zeros(10), np.array([3.0]), -np.ones(10)])
    
    trades = backtest_strategy(
        spread=spread,
        window_size=WINDOW_SIZE,
        entry_threshold=ENTRY_THRESHOLD,
        long_exit_threshold=LONG_EXIT_THRESHOLD,
        short_exit_threshold=SHORT_EXIT_THRESHOLD,
        transaction_cost=TRANSACTION_COST
    )
    
    # Check that at least one trade was executed.
    assert len(trades) >= 1, "Expected at least one short trade to be executed."
    
    # Verify that the trade(s) are short and have a computed profit.
    for trade in trades:
        assert trade['type'] == 'short', "Expected trade type to be 'short'."
        assert 'profit' in trade, "Trade dictionary should contain a 'profit' key."

if __name__ == "__main__":
    pytest.main()

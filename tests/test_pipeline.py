"""
Unit tests for data/pipeline.py, including:
  - Simulation of the OUJ process.
  - Saving simulated data to a CSV file.
  - Plotting the data (ensuring plt.show() is called).
  - Verifying the main() function execution.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure the parent directory is on sys.path so that we can import the data module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from data/data_pipeline.py
from data.data_pipeline import simulate_ouj, save_data, plot_data, main as pipeline_main

def test_simulate_ouj():
    """Test that simulate_ouj returns a numpy array of the correct length with the correct initial value."""
    T = 100
    spread = simulate_ouj(T=T)
    assert isinstance(spread, np.ndarray), "simulate_ouj should return a numpy array."
    assert len(spread) == T, f"Expected length {T}, got {len(spread)}."
    assert spread[0] == 0.0, "The initial spread value should be 0."

def test_save_data(tmp_path):
    """Test that save_data writes a CSV file with the expected 'time' and 'spread' columns and values."""
    # Create a sample spread array.
    spread = np.array([0.0, 0.5, 1.0, 1.5])
    output_file = tmp_path / "ouj_simulation_test.csv"
    
    # Save the data.
    save_data(spread, str(output_file))
    
    # Verify that the file exists.
    assert output_file.exists(), f"File {output_file} was not created."
    
    # Load the file and verify its contents.
    df = pd.read_csv(output_file)
    expected_time = list(range(len(spread)))
    assert "time" in df.columns, "CSV file should have a 'time' column."
    assert "spread" in df.columns, "CSV file should have a 'spread' column."
    assert list(df["time"]) == expected_time, "Time column values are not sequential as expected."
    np.testing.assert_allclose(df["spread"].values, spread, err_msg="Spread values do not match the input.")

def test_plot_data(monkeypatch):
    """Test that plot_data calls plt.show() without actually opening a window."""
    from matplotlib import pyplot as plt
    called = False

    def dummy_show():
        nonlocal called
        called = True

    monkeypatch.setattr(plt, "show", dummy_show)
    spread = np.linspace(-1, 1, 100)
    plot_data(spread)
    assert called, "plot_data should call plt.show() to display the plot."

def test_pipeline_main(monkeypatch):
    """
    Test the main() function in pipeline.py.
    Instead of saving data to disk, we override save_data to capture the output file path.
    Also, disable plot display.
    """
    # Import the pipeline module.
    import data.data_pipeline as pipeline

    # Override plt.show to avoid displaying the plot.
    from matplotlib import pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    # Create a dummy save_data to capture the output path.
    def dummy_save_data(spread, output_path):
        dummy_save_data.called = True
        dummy_save_data.output_path = output_path
    dummy_save_data.called = False

    monkeypatch.setattr(pipeline, "save_data", dummy_save_data)
    
    # Run main() from the pipeline.
    pipeline.main()
    
    # Confirm that save_data was called.
    assert dummy_save_data.called, "Expected main() to call save_data."
    
    # Check that the output path ends with the expected directory and filename.
    expected_suffix = os.path.join("processed", "ouj_simulation.csv")
    assert dummy_save_data.output_path.endswith(expected_suffix), \
        f"Output path is unexpected: {dummy_save_data.output_path}"

if __name__ == "__main__":
    pytest.main()

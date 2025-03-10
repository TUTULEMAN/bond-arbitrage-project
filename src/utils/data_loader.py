#!/usr/bin/env python3
"""
src/utils/data_loader.py

Utility functions for loading processed data for the bond arbitrage strategy.
"""

import os
import pandas as pd

def load_processed_data(filename="ouj_simulation.csv", base_path=None):
    """
    Load processed data from the data/processed directory.

    Parameters:
        filename (str): Name of the CSV file to load (default: "ouj_simulation.csv").
        base_path (str, optional): Base directory path for the processed data. 
                                   If not provided, the repository root is assumed.
    
    Returns:
        pd.DataFrame: DataFrame containing the processed data.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # If base_path is not provided, set it relative to this file.
    if base_path is None:
        # Assumes repository structure: src/utils/ -> two levels up to data/processed/
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
    
    file_path = os.path.join(base_path, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    # Quick test to load and display processed data.
    try:
        data = load_processed_data()
        print("Loaded data with shape:", data.shape)
        print(data.head())
    except Exception as e:
        print(f"Error loading processed data: {e}")

#!/usr/bin/env python3
"""Minimal test to isolate PyTorch tensor issue."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from tbs.envs.trading_env import TradingEnv
from tbs.data.loader import DataLoader

def test_environment():
    """Test environment creation and reset."""
    print("Testing environment creation...")
    
    # Load some data
    data_loader = DataLoader()
    data = data_loader.load_data(
        source="yfinance",
        ticker="BTC-USD",
        start="2023-01-01",
        end="2023-01-10",
        interval="1d",
        cache=True,
    )
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    
    # Create environment
    env = TradingEnv(data=data, window_size=5)
    print("Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation space dtype: {env.observation_space.dtype}")
    
    # Test step
    obs, reward, terminated, truncated, info = env.step(1)  # Buy action
    print(f"Step completed successfully")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    
    print("✅ Environment test passed!")

if __name__ == "__main__":
    test_environment()

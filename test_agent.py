#!/usr/bin/env python3
"""Test agent creation with environment."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from tbs.envs.trading_env import TradingEnv
from tbs.data.loader import DataLoader
from tbs.agents.registry import registry

def test_agent_creation():
    """Test agent creation."""
    print("Testing agent creation...")
    
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
    
    # Create environment
    env = TradingEnv(data=data, window_size=5)
    print("Environment created successfully")
    
    # Test agent creation
    try:
        agent = registry.create_agent(
            algo="ppo",
            env=env,
            learning_rate=0.0003,
            batch_size=64,
            n_steps=2048,
            gamma=0.99,
            verbose=1
        )
        print("✅ Agent created successfully!")
        
        # Test a few learning steps
        agent.learn(total_timesteps=100, progress_bar=False)
        print("✅ Agent learning test passed!")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_creation()

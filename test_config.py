#!/usr/bin/env python3
"""Test configuration loading and merging."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omegaconf import OmegaConf

def test_config():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    try:
        # Load configurations
        base_config = OmegaConf.load("configs/default.yaml")
        algo_config = OmegaConf.load("configs/ppo.yaml")
        config = OmegaConf.merge(base_config, algo_config)
        
        print("✅ Configuration loaded successfully!")
        print(f"Config keys: {list(config.keys())}")
        print(f"Train config: {config.train}")
        print(f"Env config: {config.env}")
        print(f"Portfolio config: {config.portfolio}")
        
        # Test environment creation with this config
        from tbs.data.loader import DataLoader
        from tbs.envs.trading_env import TradingEnv
        
        data_loader = DataLoader()
        data = data_loader.load_data(
            source="yfinance",
            ticker="BTC-USD",
            start="2023-01-01",
            end="2023-03-01",  # More data for larger window
            interval="1d",
            cache=True,
        )
        
        print(f"Data shape: {data.shape}")
        print(f"Window size: {config.env.window_size}")
        print(f"Required rows: {config.env.window_size + 1}")
        
        env = TradingEnv(
            data=data,
            window_size=10,  # Smaller window for testing
            fee_bps=config.env.fee_bps,
            slippage_bps=config.env.slippage_bps,
            reward_scheme=config.env.reward,
            max_drawdown=config.env.max_drawdown,
            starting_cash=config.portfolio.starting_cash,
            leverage=config.portfolio.leverage,
            position_sizing=config.portfolio.position_sizing,
            seed=config.train.seed,
        )
        
        print("✅ Environment created with config successfully!")
        
        # Test agent creation
        from tbs.agents.registry import registry
        
        # Filter training parameters
        training_params = {"total_timesteps", "algo", "seed", "train_split", "test_split", "eval_freq", "save_freq", "tensorboard_log"}
        agent_kwargs = {k: v for k, v in config.train.items() 
                       if k not in training_params}
        
        print(f"Agent kwargs: {agent_kwargs}")
        
        agent = registry.create_agent(
            algo=config.train.algo,
            env=env,
            policy_kwargs={},  # Empty policy kwargs to avoid issues
            learning_rate=0.0003,
            batch_size=64,
            n_steps=2048,
            gamma=0.99,
            verbose=1
        )
        
        print("✅ Agent created with config successfully!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config()

#!/usr/bin/env python3
"""Example script demonstrating Trading Bot Simulator usage."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tbs.data.loader import DataLoader
from tbs.envs.trading_env import TradingEnv
from tbs.portfolio.wallet import Wallet


def main():
    """Run a simple example."""
    print("🤖 Trading Bot Simulator - Example")
    print("=" * 50)
    
    # Load sample data
    print("📊 Loading sample data...")
    data_loader = DataLoader()
    
    # Use sample CSV data
    sample_data = data_loader.load_data(
        source="csv",
        csv_path="data/sample/BTC-USD_sample.csv"
    )
    
    print(f"✅ Loaded {len(sample_data)} data points")
    print(f"📅 Date range: {sample_data['Datetime'].min()} to {sample_data['Datetime'].max()}")
    print(f"💰 Price range: ${sample_data['Close'].min():.2f} to ${sample_data['Close'].max():.2f}")
    
    # Create environment
    print("\n🎮 Creating trading environment...")
    env = TradingEnv(
        data=sample_data,
        window_size=10,
        fee_bps=10.0,
        slippage_bps=5.0,
        reward_scheme="profit_increment",
        max_drawdown=0.3,
        starting_cash=10000.0,
        seed=42,
    )
    
    print(f"✅ Environment created with {env.action_space.n} actions")
    print(f"📊 Observation space: {env.observation_space.shape}")
    
    # Run a simple simulation
    print("\n🎯 Running simulation...")
    obs, info = env.reset()
    
    total_reward = 0.0
    step_count = 0
    max_steps = 50
    
    while step_count < max_steps:
        # Simple random policy
        import random
        action = random.randint(0, 2)  # 0=Hold, 1=Buy, 2=Sell
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"Step {step_count}: Portfolio Value = ${info['portfolio_value']:.2f}, "
                  f"Cash = ${info['cash']:.2f}, Position = {info['position']:.2f}")
        
        if done:
            break
    
    print(f"\n📈 Simulation completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
    print(f"Total return: {((info['portfolio_value'] - 10000) / 10000) * 100:.2f}%")
    print(f"Number of trades: {info['actions_taken']}")
    
    print("\n🎉 Example completed successfully!")
    print("💡 Try running 'tbs --help' to see all available commands")


if __name__ == "__main__":
    main()

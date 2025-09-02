"""Trading environment implementation."""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Union

from ..data.loader import DataLoader
from ..portfolio.wallet import Wallet
from .features import FeatureEngineer
from .reward_schemes import RewardScheme
from .utils import apply_slippage, calculate_fees


class TradingEnv(gym.Env):
    """A trading environment for reinforcement learning agents.
    
    This environment simulates trading with mock currency while using real historical
    market data. The agent can perform Buy, Sell, or Hold actions to maximize profit.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 64,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        reward_scheme: str = "profit_increment",
        max_drawdown: float = 0.3,
        stop_loss: float = 0.2,
        allow_short: bool = False,
        starting_cash: float = 10000.0,
        leverage: float = 1.0,
        position_sizing: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the trading environment.
        
        Args:
            data: OHLCV price data
            window_size: Number of historical bars to include in observation
            fee_bps: Trading fee in basis points (0.1% = 10 bps)
            slippage_bps: Slippage in basis points (0.05% = 5 bps)
            reward_scheme: Reward function to use
            max_drawdown: Maximum allowed drawdown before episode termination
            stop_loss: Stop loss threshold as fraction of portfolio value
            allow_short: Whether to allow short selling
            starting_cash: Initial cash balance
            leverage: Leverage multiplier (1.0 = no leverage)
            position_sizing: Fraction of available cash to use per trade
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.data = data.copy()
        self.window_size = window_size
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.allow_short = allow_short
        self.starting_cash = starting_cash
        self.leverage = leverage
        self.position_sizing = position_sizing
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.reward_scheme = RewardScheme(reward_scheme)
        self.wallet = Wallet(
            starting_cash=starting_cash,
            allow_short=allow_short,
            leverage=leverage,
        )
        
        # State tracking
        self.current_step = 0
        self.initial_portfolio_value = starting_cash
        self.max_portfolio_value = starting_cash
        self.episode_actions = []
        self.episode_rewards = []
        
        # Set random seed
        if seed is not None:
            self._seed = seed
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        
        # Observation space: features + portfolio state
        feature_dim = self.feature_engineer.get_feature_dim()
        portfolio_dim = 4  # cash, position, unrealized_pnl, portfolio_value
        obs_dim = feature_dim + portfolio_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=float
        )
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate the input data format."""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        if len(self.data) < self.window_size + 1:
            raise ValueError(f"Data must have at least {self.window_size + 1} rows")
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = self.window_size
        self.wallet.reset()
        self.initial_portfolio_value = self.starting_cash
        self.max_portfolio_value = self.starting_cash
        self.episode_actions = []
        self.episode_rewards = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Execute action
        reward = self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Check termination conditions
        terminated = self._check_termination()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _execute_action(self, action: int) -> float:
        """Execute the given action and return reward.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            Reward for this action
        """
        current_price = self.data.iloc[self.current_step]["Close"]
        
        if action == 0:  # Hold
            return 0.0
        elif action == 1:  # Buy
            # Calculate position size
            available_cash = self.wallet.cash * self.position_sizing
            if available_cash <= 0:
                return 0.0
            
            # Apply slippage
            execution_price = apply_slippage(current_price, self.slippage_bps, "buy")
            
            # Calculate units to buy
            units = available_cash / execution_price
            
            # Execute trade
            trade_value = units * execution_price
            fees = calculate_fees(trade_value, self.fee_bps)
            total_cost = trade_value + fees
            
            if total_cost <= self.wallet.cash:
                self.wallet.buy(units, execution_price, fees)
                self.episode_actions.append(("buy", units, execution_price))
            else:
                return 0.0
                
        elif action == 2:  # Sell
            if self.wallet.position <= 0:
                return 0.0
            
            # Apply slippage
            execution_price = apply_slippage(current_price, self.slippage_bps, "sell")
            
            # Execute trade
            trade_value = self.wallet.position * execution_price
            fees = calculate_fees(trade_value, self.fee_bps)
            total_proceeds = trade_value - fees
            
            self.wallet.sell(self.wallet.position, execution_price, fees)
            self.episode_actions.append(("sell", self.wallet.position, execution_price))
        
        # Calculate reward
        reward = self.reward_scheme.calculate_reward(
            self.wallet, self.episode_actions, self.current_step
        )
        
        return reward
    
    def _update_portfolio_value(self) -> None:
        """Update current portfolio value."""
        current_price = self.data.iloc[self.current_step]["Close"]
        self.wallet.update_unrealized_pnl(current_price)
        
        portfolio_value = self.wallet.get_portfolio_value(current_price)
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        current_price = self.data.iloc[self.current_step]["Close"]
        portfolio_value = self.wallet.get_portfolio_value(current_price)
        
        # Check stop loss
        if portfolio_value <= self.initial_portfolio_value * (1 - self.stop_loss):
            return True
        
        # Check max drawdown
        current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        if current_drawdown >= self.max_drawdown:
            return True
        
        # Check if we've reached the end of data
        if self.current_step >= len(self.data) - 1:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get price data window
        window_data = self.data.iloc[self.current_step - self.window_size + 1:self.current_step + 1]
        
        # Calculate features
        features = self.feature_engineer.calculate_features(window_data)
        
        # Get portfolio state
        current_price = self.data.iloc[self.current_step]["Close"]
        portfolio_state = np.array([
            self.wallet.cash,
            self.wallet.position,
            self.wallet.unrealized_pnl,
            self.wallet.get_portfolio_value(current_price)
        ], dtype=np.float32)
        
        # Combine features and portfolio state
        observation = np.concatenate([features, portfolio_state])
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current environment info."""
        current_price = self.data.iloc[self.current_step]["Close"]
        portfolio_value = self.wallet.get_portfolio_value(current_price)
        
        return {
            "step": self.current_step,
            "portfolio_value": portfolio_value,
            "cash": self.wallet.cash,
            "position": self.wallet.position,
            "unrealized_pnl": self.wallet.unrealized_pnl,
            "realized_pnl": self.wallet.realized_pnl,
            "total_pnl": self.wallet.realized_pnl + self.wallet.unrealized_pnl,
            "current_price": current_price,
            "max_portfolio_value": self.max_portfolio_value,
            "drawdown": (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value,
            "actions_taken": len(self.episode_actions),
        }
    
    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass
    
    def close(self) -> None:
        """Close the environment."""
        pass

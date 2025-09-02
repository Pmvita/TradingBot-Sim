"""Reward schemes for the trading environment."""

import numpy as np
from typing import List, Tuple

from ..portfolio.wallet import Wallet


class RewardScheme:
    """Base class for reward schemes."""
    
    def __init__(self, scheme_name: str) -> None:
        """Initialize reward scheme.
        
        Args:
            scheme_name: Name of the reward scheme to use
        """
        self.scheme_name = scheme_name
    
    def calculate_reward(
        self, wallet: Wallet, actions: List[Tuple[str, float, float]], step: int
    ) -> float:
        """Calculate reward for the current state.
        
        Args:
            wallet: Current wallet state
            actions: List of actions taken in this episode
            step: Current step number
            
        Returns:
            Reward value
        """
        if self.scheme_name == "profit_increment":
            return self._profit_increment(wallet, actions, step)
        elif self.scheme_name == "risk_adjusted":
            return self._risk_adjusted(wallet, actions, step)
        elif self.scheme_name == "trade_penalty":
            return self._trade_penalty(wallet, actions, step)
        else:
            raise ValueError(f"Unknown reward scheme: {self.scheme_name}")
    
    def _profit_increment(
        self, wallet: Wallet, actions: List[Tuple[str, float, float]], step: int
    ) -> float:
        """Simple profit increment reward.
        
        Returns the change in portfolio value since the last step.
        """
        if step == 0:
            return 0.0
        
        # Calculate current portfolio value
        current_price = wallet.last_price if hasattr(wallet, 'last_price') else 0.0
        current_value = wallet.get_portfolio_value(current_price)
        
        # Calculate previous portfolio value (simplified)
        # In a real implementation, you'd track the previous value
        return current_value - wallet.starting_cash
    
    def _risk_adjusted(
        self, wallet: Wallet, actions: List[Tuple[str, float, float]], step: int
    ) -> float:
        """Risk-adjusted reward that penalizes volatility.
        
        Combines profit with a penalty for high volatility.
        """
        profit_reward = self._profit_increment(wallet, actions, step)
        
        # Calculate volatility penalty (simplified)
        if len(actions) > 1:
            # Penalize frequent trading
            volatility_penalty = -0.001 * len(actions)
        else:
            volatility_penalty = 0.0
        
        return profit_reward + volatility_penalty
    
    def _trade_penalty(
        self, wallet: Wallet, actions: List[Tuple[str, float, float]], step: int
    ) -> float:
        """Reward with small penalty per trade to reduce churn."""
        profit_reward = self._profit_increment(wallet, actions, step)
        
        # Small penalty per trade
        trade_penalty = -0.0001 * len(actions)
        
        return profit_reward + trade_penalty

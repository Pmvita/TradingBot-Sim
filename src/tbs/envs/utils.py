"""Utility functions for trading environment."""

import numpy as np
from typing import Literal, Tuple


def apply_slippage(price: float, slippage_bps: float, direction: Literal["buy", "sell"]) -> float:
    """Apply slippage to trade execution price.
    
    Args:
        price: Base price
        slippage_bps: Slippage in basis points
        direction: Trade direction ("buy" or "sell")
        
    Returns:
        Execution price with slippage applied
    """
    slippage_multiplier = slippage_bps / 10000.0  # Convert bps to decimal
    
    if direction == "buy":
        # Buy orders typically execute at a higher price (worse for buyer)
        return price * (1 + slippage_multiplier)
    elif direction == "sell":
        # Sell orders typically execute at a lower price (worse for seller)
        return price * (1 - slippage_multiplier)
    else:
        raise ValueError(f"Invalid direction: {direction}")


def calculate_fees(trade_value: float, fee_bps: float) -> float:
    """Calculate trading fees.
    
    Args:
        trade_value: Value of the trade
        fee_bps: Fee rate in basis points
        
    Returns:
        Fee amount
    """
    fee_rate = fee_bps / 10000.0  # Convert bps to decimal
    return trade_value * fee_rate


def calculate_position_size(
    available_cash: float,
    current_price: float,
    position_sizing: float = 1.0,
    max_leverage: float = 1.0,
) -> float:
    """Calculate position size based on available cash and constraints.
    
    Args:
        available_cash: Available cash for trading
        current_price: Current asset price
        position_sizing: Fraction of cash to use (0.0 to 1.0)
        max_leverage: Maximum leverage multiplier
        
    Returns:
        Number of units to trade
    """
    if current_price <= 0:
        return 0.0
    
    # Calculate base position size
    cash_to_use = available_cash * position_sizing
    base_units = cash_to_use / current_price
    
    # Apply leverage constraint
    max_units = (available_cash * max_leverage) / current_price
    
    return min(base_units, max_units)


def normalize_price_data(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize price data for feature engineering.
    
    Args:
        data: Price data array
        method: Normalization method ("minmax", "zscore", "returns")
        
    Returns:
        Normalized data
    """
    if method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    elif method == "zscore":
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return np.zeros_like(data)
        return (data - mean_val) / std_val
    
    elif method == "returns":
        # Calculate percentage returns
        returns = np.diff(data) / data[:-1]
        # Pad with zero for the first element
        return np.concatenate([[0.0], returns])
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_volatility(prices: np.ndarray, window: int = 20) -> float:
    """Calculate price volatility.
    
    Args:
        prices: Price array
        window: Rolling window size
        
    Returns:
        Volatility measure
    """
    if len(prices) < window:
        return 0.0
    
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns[-window:]) if len(returns) >= window else np.std(returns)


def calculate_drawdown(portfolio_values: np.ndarray) -> Tuple[float, float]:
    """Calculate maximum drawdown and current drawdown.
    
    Args:
        portfolio_values: Array of portfolio values over time
        
    Returns:
        Tuple of (max_drawdown, current_drawdown)
    """
    if len(portfolio_values) == 0:
        return 0.0, 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Calculate drawdowns
    drawdowns = (running_max - portfolio_values) / running_max
    
    max_drawdown = np.max(drawdowns)
    current_drawdown = drawdowns[-1] if len(drawdowns) > 0 else 0.0
    
    return max_drawdown, current_drawdown


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    if std_excess_return == 0:
        return 0.0
    
    # Annualize (assuming daily returns)
    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)
    
    return sharpe_ratio


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    mean_excess_return = np.mean(excess_returns)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    # Annualize (assuming daily returns)
    sortino_ratio = (mean_excess_return / downside_deviation) * np.sqrt(252)
    
    return sortino_ratio

"""Mock portfolio wallet implementation."""

from typing import List, Optional


class Wallet:
    """Mock portfolio wallet for tracking positions and PnL."""
    
    def __init__(
        self,
        starting_cash: float = 10000.0,
        allow_short: bool = False,
        leverage: float = 1.0,
    ) -> None:
        """Initialize wallet.
        
        Args:
            starting_cash: Initial cash balance
            allow_short: Whether to allow short selling
            leverage: Leverage multiplier (1.0 = no leverage)
        """
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.position = 0.0  # Number of units held
        self.avg_price = 0.0  # Average price of position
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fees = 0.0
        self.allow_short = allow_short
        self.leverage = leverage
        self.last_price = 0.0
        
        # Trade history
        self.trades: List[dict] = []
    
    def reset(self) -> None:
        """Reset wallet to initial state."""
        self.cash = self.starting_cash
        self.position = 0.0
        self.avg_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fees = 0.0
        self.last_price = 0.0
        self.trades.clear()
    
    def buy(self, units: float, price: float, fees: float) -> bool:
        """Buy units at given price.
        
        Args:
            units: Number of units to buy
            price: Price per unit
            fees: Trading fees
            
        Returns:
            True if trade was successful
        """
        if units <= 0:
            return False
        
        total_cost = units * price + fees
        
        if total_cost > self.cash:
            return False
        
        # Update position
        if self.position == 0:
            self.avg_price = price
        else:
            # Calculate new average price
            total_value = self.position * self.avg_price + units * price
            self.avg_price = total_value / (self.position + units)
        
        self.position += units
        self.cash -= total_cost
        self.total_fees += fees
        
        # Record trade
        self.trades.append({
            "action": "buy",
            "units": units,
            "price": price,
            "fees": fees,
            "timestamp": len(self.trades)
        })
        
        return True
    
    def sell(self, units: float, price: float, fees: float) -> bool:
        """Sell units at given price.
        
        Args:
            units: Number of units to sell
            price: Price per unit
            fees: Trading fees
            
        Returns:
            True if trade was successful
        """
        if units <= 0 or self.position < units:
            return False
        
        total_proceeds = units * price - fees
        
        # Calculate realized PnL
        realized_pnl = (price - self.avg_price) * units - fees
        self.realized_pnl += realized_pnl
        
        # Update position
        self.position -= units
        self.cash += total_proceeds
        self.total_fees += fees
        
        # Update average price if position remains
        if self.position == 0:
            self.avg_price = 0.0
        
        # Record trade
        self.trades.append({
            "action": "sell",
            "units": units,
            "price": price,
            "fees": fees,
            "realized_pnl": realized_pnl,
            "timestamp": len(self.trades)
        })
        
        return True
    
    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized PnL based on current price.
        
        Args:
            current_price: Current market price
        """
        self.last_price = current_price
        
        if self.position != 0:
            self.unrealized_pnl = (current_price - self.avg_price) * self.position
        else:
            self.unrealized_pnl = 0.0
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Get total portfolio value.
        
        Args:
            current_price: Current market price
            
        Returns:
            Total portfolio value
        """
        return self.cash + (self.position * current_price)
    
    def get_total_pnl(self, current_price: float) -> float:
        """Get total PnL (realized + unrealized).
        
        Args:
            current_price: Current market price
            
        Returns:
            Total PnL
        """
        self.update_unrealized_pnl(current_price)
        return self.realized_pnl + self.unrealized_pnl
    
    def get_return(self, current_price: float) -> float:
        """Get total return as percentage.
        
        Args:
            current_price: Current market price
            
        Returns:
            Total return as percentage
        """
        total_value = self.get_portfolio_value(current_price)
        return ((total_value - self.starting_cash) / self.starting_cash) * 100
    
    def get_position_value(self, current_price: float) -> float:
        """Get current position value.
        
        Args:
            current_price: Current market price
            
        Returns:
            Position value
        """
        return self.position * current_price
    
    def get_cash_ratio(self, current_price: float) -> float:
        """Get cash as ratio of total portfolio value.
        
        Args:
            current_price: Current market price
            
        Returns:
            Cash ratio (0.0 to 1.0)
        """
        total_value = self.get_portfolio_value(current_price)
        return self.cash / total_value if total_value > 0 else 1.0
    
    def get_position_ratio(self, current_price: float) -> float:
        """Get position as ratio of total portfolio value.
        
        Args:
            current_price: Current market price
            
        Returns:
            Position ratio (0.0 to 1.0)
        """
        total_value = self.get_portfolio_value(current_price)
        position_value = self.get_position_value(current_price)
        return position_value / total_value if total_value > 0 else 0.0
    
    def can_buy(self, units: float, price: float, fees: float) -> bool:
        """Check if we can afford to buy.
        
        Args:
            units: Number of units to buy
            price: Price per unit
            fees: Trading fees
            
        Returns:
            True if we can afford the trade
        """
        total_cost = units * price + fees
        return total_cost <= self.cash
    
    def can_sell(self, units: float) -> bool:
        """Check if we can sell the given units.
        
        Args:
            units: Number of units to sell
            
        Returns:
            True if we have enough units to sell
        """
        return self.position >= units
    
    def get_trade_count(self) -> int:
        """Get total number of trades.
        
        Returns:
            Number of trades
        """
        return len(self.trades)
    
    def get_buy_trades(self) -> List[dict]:
        """Get list of buy trades.
        
        Returns:
            List of buy trade dictionaries
        """
        return [trade for trade in self.trades if trade["action"] == "buy"]
    
    def get_sell_trades(self) -> List[dict]:
        """Get list of sell trades.
        
        Returns:
            List of sell trade dictionaries
        """
        return [trade for trade in self.trades if trade["action"] == "sell"]
    
    def get_trade_history(self) -> List[dict]:
        """Get complete trade history.
        
        Returns:
            List of all trade dictionaries
        """
        return self.trades.copy()

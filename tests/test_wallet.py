"""Tests for portfolio wallet functionality."""

import pytest
import numpy as np

from tbs.portfolio.wallet import Wallet


@pytest.fixture
def wallet():
    """Create wallet for testing."""
    return Wallet(starting_cash=10000.0, allow_short=False, leverage=1.0)


class TestWallet:
    """Test wallet functionality."""
    
    def test_initialization(self, wallet):
        """Test wallet initialization."""
        assert wallet.starting_cash == 10000.0
        assert wallet.cash == 10000.0
        assert wallet.position == 0.0
        assert wallet.avg_price == 0.0
        assert wallet.realized_pnl == 0.0
        assert wallet.unrealized_pnl == 0.0
        assert wallet.total_fees == 0.0
        assert wallet.allow_short is False
        assert wallet.leverage == 1.0
    
    def test_reset(self, wallet):
        """Test wallet reset."""
        # Make some changes
        wallet.cash = 5000.0
        wallet.position = 10.0
        wallet.realized_pnl = 100.0
        
        # Reset
        wallet.reset()
        
        assert wallet.cash == wallet.starting_cash
        assert wallet.position == 0.0
        assert wallet.realized_pnl == 0.0
        assert len(wallet.trades) == 0
    
    def test_buy_success(self, wallet):
        """Test successful buy."""
        units = 10.0
        price = 100.0
        fees = 5.0
        
        success = wallet.buy(units, price, fees)
        
        assert success is True
        assert wallet.position == units
        assert wallet.avg_price == price
        assert wallet.cash == wallet.starting_cash - (units * price + fees)
        assert wallet.total_fees == fees
        assert len(wallet.trades) == 1
        assert wallet.trades[0]["action"] == "buy"
    
    def test_buy_insufficient_cash(self, wallet):
        """Test buy with insufficient cash."""
        units = 1000.0  # Too many units
        price = 100.0
        fees = 5.0
        
        success = wallet.buy(units, price, fees)
        
        assert success is False
        assert wallet.position == 0.0
        assert wallet.cash == wallet.starting_cash
        assert len(wallet.trades) == 0
    
    def test_buy_invalid_units(self, wallet):
        """Test buy with invalid units."""
        success = wallet.buy(-1.0, 100.0, 5.0)
        
        assert success is False
        assert wallet.position == 0.0
        assert wallet.cash == wallet.starting_cash
    
    def test_sell_success(self, wallet):
        """Test successful sell."""
        # First buy
        wallet.buy(10.0, 100.0, 5.0)
        
        # Then sell
        sell_price = 110.0
        sell_fees = 5.0
        success = wallet.sell(10.0, sell_price, sell_fees)
        
        assert success is True
        assert wallet.position == 0.0
        assert wallet.avg_price == 0.0
        assert wallet.realized_pnl > 0  # Should have profit
        assert len(wallet.trades) == 2
        assert wallet.trades[1]["action"] == "sell"
    
    def test_sell_insufficient_position(self, wallet):
        """Test sell with insufficient position."""
        success = wallet.sell(10.0, 100.0, 5.0)
        
        assert success is False
        assert wallet.position == 0.0
        assert wallet.cash == wallet.starting_cash
    
    def test_sell_partial(self, wallet):
        """Test partial sell."""
        # Buy 20 units
        wallet.buy(20.0, 100.0, 5.0)
        
        # Sell 10 units
        success = wallet.sell(10.0, 110.0, 5.0)
        
        assert success is True
        assert wallet.position == 10.0
        assert wallet.avg_price == 100.0  # Should remain the same
        assert wallet.realized_pnl > 0
    
    def test_update_unrealized_pnl(self, wallet):
        """Test unrealized PnL update."""
        # Buy some units
        wallet.buy(10.0, 100.0, 5.0)
        
        # Update with higher price
        wallet.update_unrealized_pnl(110.0)
        
        assert wallet.unrealized_pnl > 0  # Should have unrealized profit
        assert wallet.last_price == 110.0
    
    def test_get_portfolio_value(self, wallet):
        """Test portfolio value calculation."""
        # Buy some units
        wallet.buy(10.0, 100.0, 5.0)
        current_price = 110.0
        
        portfolio_value = wallet.get_portfolio_value(current_price)
        
        expected_value = wallet.cash + (wallet.position * current_price)
        assert portfolio_value == expected_value
    
    def test_get_total_pnl(self, wallet):
        """Test total PnL calculation."""
        # Buy and sell for profit
        wallet.buy(10.0, 100.0, 5.0)
        wallet.sell(10.0, 110.0, 5.0)
        
        total_pnl = wallet.get_total_pnl(110.0)
        
        assert total_pnl == wallet.realized_pnl + wallet.unrealized_pnl
    
    def test_get_return(self, wallet):
        """Test return calculation."""
        # Buy some units
        wallet.buy(10.0, 100.0, 5.0)
        current_price = 110.0
        
        return_pct = wallet.get_return(current_price)
        
        expected_return = ((wallet.get_portfolio_value(current_price) - wallet.starting_cash) / wallet.starting_cash) * 100
        assert return_pct == expected_return
    
    def test_get_position_value(self, wallet):
        """Test position value calculation."""
        # Buy some units
        wallet.buy(10.0, 100.0, 5.0)
        current_price = 110.0
        
        position_value = wallet.get_position_value(current_price)
        
        assert position_value == wallet.position * current_price
    
    def test_get_cash_ratio(self, wallet):
        """Test cash ratio calculation."""
        # Buy some units
        wallet.buy(10.0, 100.0, 5.0)
        current_price = 110.0
        
        cash_ratio = wallet.get_cash_ratio(current_price)
        
        total_value = wallet.get_portfolio_value(current_price)
        expected_ratio = wallet.cash / total_value
        assert cash_ratio == expected_ratio
    
    def test_get_position_ratio(self, wallet):
        """Test position ratio calculation."""
        # Buy some units
        wallet.buy(10.0, 100.0, 5.0)
        current_price = 110.0
        
        position_ratio = wallet.get_position_ratio(current_price)
        
        total_value = wallet.get_portfolio_value(current_price)
        position_value = wallet.get_position_value(current_price)
        expected_ratio = position_value / total_value
        assert position_ratio == expected_ratio
    
    def test_can_buy(self, wallet):
        """Test can buy check."""
        # Should be able to buy
        assert wallet.can_buy(10.0, 100.0, 5.0) is True
        
        # Should not be able to buy
        assert wallet.can_buy(1000.0, 100.0, 5.0) is False
    
    def test_can_sell(self, wallet):
        """Test can sell check."""
        # Should not be able to sell (no position)
        assert wallet.can_sell(10.0) is False
        
        # Buy some units
        wallet.buy(10.0, 100.0, 5.0)
        
        # Should be able to sell
        assert wallet.can_sell(10.0) is True
        assert wallet.can_sell(5.0) is True
        
        # Should not be able to sell more than position
        assert wallet.can_sell(15.0) is False
    
    def test_get_trade_count(self, wallet):
        """Test trade count."""
        assert wallet.get_trade_count() == 0
        
        wallet.buy(10.0, 100.0, 5.0)
        assert wallet.get_trade_count() == 1
        
        wallet.sell(10.0, 110.0, 5.0)
        assert wallet.get_trade_count() == 2
    
    def test_get_buy_trades(self, wallet):
        """Test get buy trades."""
        wallet.buy(10.0, 100.0, 5.0)
        wallet.sell(5.0, 110.0, 5.0)
        wallet.buy(5.0, 105.0, 5.0)
        
        buy_trades = wallet.get_buy_trades()
        assert len(buy_trades) == 2
        assert all(trade["action"] == "buy" for trade in buy_trades)
    
    def test_get_sell_trades(self, wallet):
        """Test get sell trades."""
        wallet.buy(10.0, 100.0, 5.0)
        wallet.sell(5.0, 110.0, 5.0)
        wallet.buy(5.0, 105.0, 5.0)
        
        sell_trades = wallet.get_sell_trades()
        assert len(sell_trades) == 1
        assert all(trade["action"] == "sell" for trade in sell_trades)
    
    def test_get_trade_history(self, wallet):
        """Test get trade history."""
        wallet.buy(10.0, 100.0, 5.0)
        wallet.sell(5.0, 110.0, 5.0)
        
        history = wallet.get_trade_history()
        assert len(history) == 2
        assert history[0]["action"] == "buy"
        assert history[1]["action"] == "sell"

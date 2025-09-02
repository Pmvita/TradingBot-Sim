"""Tests for trading environment."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from tbs.envs.trading_env import TradingEnv
from tbs.envs.features import FeatureEngineer
from tbs.envs.reward_schemes import RewardScheme
from tbs.envs.utils import apply_slippage, calculate_fees


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Datetime": dates,
        "Open": np.random.uniform(100, 200, 100),
        "High": np.random.uniform(150, 250, 100),
        "Low": np.random.uniform(50, 150, 100),
        "Close": np.random.uniform(100, 200, 100),
        "Volume": np.random.uniform(1000000, 10000000, 100),
    })
    return data


@pytest.fixture
def trading_env(sample_data):
    """Create trading environment for testing."""
    return TradingEnv(
        data=sample_data,
        window_size=10,
        fee_bps=10.0,
        slippage_bps=5.0,
        reward_scheme="profit_increment",
        max_drawdown=0.3,
        starting_cash=10000.0,
        seed=42,
    )


class TestTradingEnv:
    """Test trading environment functionality."""
    
    def test_initialization(self, trading_env):
        """Test environment initialization."""
        assert trading_env.window_size == 10
        assert trading_env.fee_bps == 10.0
        assert trading_env.slippage_bps == 5.0
        assert trading_env.starting_cash == 10000.0
        assert trading_env.action_space.n == 3  # Buy, Sell, Hold
    
    def test_reset(self, trading_env):
        """Test environment reset."""
        obs, info = trading_env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) > 0
        
        assert isinstance(info, dict)
        assert "portfolio_value" in info
        assert "cash" in info
        assert "position" in info
    
    def test_step(self, trading_env):
        """Test environment step."""
        obs, _ = trading_env.reset()
        
        # Test hold action
        obs, reward, done, truncated, info = trading_env.step(0)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_buy_action(self, trading_env):
        """Test buy action."""
        obs, _ = trading_env.reset()
        
        # Buy action
        obs, reward, done, truncated, info = trading_env.step(1)
        
        assert info["position"] > 0
        assert info["cash"] < trading_env.starting_cash
    
    def test_sell_action(self, trading_env):
        """Test sell action."""
        obs, _ = trading_env.reset()
        
        # First buy
        trading_env.step(1)
        
        # Then sell
        obs, reward, done, truncated, info = trading_env.step(2)
        
        assert info["position"] == 0
        assert info["cash"] > 0
    
    def test_termination_conditions(self, trading_env):
        """Test episode termination conditions."""
        obs, _ = trading_env.reset()
        
        # Run until termination
        done = False
        step_count = 0
        max_steps = 1000
        
        while not done and step_count < max_steps:
            obs, reward, done, truncated, info = trading_env.step(0)
            step_count += 1
        
        assert step_count < max_steps  # Should terminate before max steps


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    def test_initialization(self):
        """Test feature engineer initialization."""
        fe = FeatureEngineer()
        assert fe.sma_periods == [10, 20, 50]
        assert fe.ema_periods == [12, 26]
        assert fe.rsi_length == 14
    
    def test_get_feature_dim(self):
        """Test feature dimension calculation."""
        fe = FeatureEngineer()
        dim = fe.get_feature_dim()
        assert dim > 0
        assert isinstance(dim, int)
    
    def test_calculate_features(self, sample_data):
        """Test feature calculation."""
        fe = FeatureEngineer()
        features = fe.calculate_features(sample_data)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) == fe.get_feature_dim()
    
    def test_price_features(self, sample_data):
        """Test price feature calculation."""
        fe = FeatureEngineer()
        features = fe._calculate_price_features(sample_data)
        
        assert len(features) == 3
        assert all(isinstance(f, float) for f in features)
    
    def test_sma_features(self, sample_data):
        """Test SMA feature calculation."""
        fe = FeatureEngineer()
        features = fe._calculate_sma_features(sample_data)
        
        assert len(features) == len(fe.sma_periods)
        assert all(isinstance(f, float) for f in features)


class TestRewardScheme:
    """Test reward scheme functionality."""
    
    def test_initialization(self):
        """Test reward scheme initialization."""
        rs = RewardScheme("profit_increment")
        assert rs.scheme_name == "profit_increment"
    
    def test_profit_increment(self):
        """Test profit increment reward."""
        rs = RewardScheme("profit_increment")
        
        # Mock wallet
        wallet = Mock()
        wallet.last_price = 100.0
        wallet.get_portfolio_value.return_value = 11000.0
        wallet.starting_cash = 10000.0
        
        reward = rs._profit_increment(wallet, [], 1)
        assert isinstance(reward, float)
    
    def test_risk_adjusted(self):
        """Test risk adjusted reward."""
        rs = RewardScheme("risk_adjusted")
        
        wallet = Mock()
        wallet.last_price = 100.0
        wallet.get_portfolio_value.return_value = 11000.0
        wallet.starting_cash = 10000.0
        
        reward = rs._risk_adjusted(wallet, [("buy", 1.0, 100.0)], 1)
        assert isinstance(reward, float)
    
    def test_trade_penalty(self):
        """Test trade penalty reward."""
        rs = RewardScheme("trade_penalty")
        
        wallet = Mock()
        wallet.last_price = 100.0
        wallet.get_portfolio_value.return_value = 11000.0
        wallet.starting_cash = 10000.0
        
        reward = rs._trade_penalty(wallet, [("buy", 1.0, 100.0)], 1)
        assert isinstance(reward, float)
    
    def test_unknown_scheme(self):
        """Test unknown reward scheme."""
        rs = RewardScheme("unknown")
        
        with pytest.raises(ValueError):
            rs.calculate_reward(Mock(), [], 0)


class TestUtils:
    """Test utility functions."""
    
    def test_apply_slippage(self):
        """Test slippage application."""
        price = 100.0
        slippage_bps = 10.0
        
        buy_price = apply_slippage(price, slippage_bps, "buy")
        sell_price = apply_slippage(price, slippage_bps, "sell")
        
        assert buy_price > price  # Buy price should be higher
        assert sell_price < price  # Sell price should be lower
        assert buy_price == price * 1.001  # 10 bps = 0.1%
        assert sell_price == price * 0.999
    
    def test_calculate_fees(self):
        """Test fee calculation."""
        trade_value = 1000.0
        fee_bps = 10.0
        
        fees = calculate_fees(trade_value, fee_bps)
        
        assert fees == 1.0  # 10 bps = 0.1% = 1.0
        assert isinstance(fees, float)
    
    def test_invalid_direction(self):
        """Test invalid slippage direction."""
        with pytest.raises(ValueError):
            apply_slippage(100.0, 10.0, "invalid")

"""Tests for data loading functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from tbs.data.loader import DataLoader


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "Datetime": dates,
        "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "High": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        "Low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "Close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Volume": [1000000] * 10,
    })
    return data


@pytest.fixture
def temp_csv_file(tmp_path, sample_csv_data):
    """Create temporary CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    sample_csv_data.to_csv(csv_file, index=False)
    return csv_file


class TestDataLoader:
    """Test data loader functionality."""
    
    def test_initialization(self, tmp_path):
        """Test data loader initialization."""
        loader = DataLoader(cache_dir=str(tmp_path))
        assert loader.cache_dir == tmp_path
        assert loader.cache_dir.exists()
    
    def test_load_csv_data(self, temp_csv_file):
        """Test loading CSV data."""
        loader = DataLoader()
        data = loader.load_data(source="csv", csv_path=str(temp_csv_file))
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert all(col in data.columns for col in ["Datetime", "Open", "High", "Low", "Close", "Volume"])
        assert pd.api.types.is_datetime64_any_dtype(data["Datetime"])
    
    def test_load_csv_missing_columns(self, tmp_path):
        """Test loading CSV with missing columns."""
        # Create CSV with missing columns
        bad_data = pd.DataFrame({
            "Date": pd.date_range("2022-01-01", periods=5),
            "Price": [100, 101, 102, 103, 104],
        })
        bad_csv = tmp_path / "bad_data.csv"
        bad_data.to_csv(bad_csv, index=False)
        
        loader = DataLoader()
        
        with pytest.raises(ValueError):
            loader.load_data(source="csv", csv_path=str(bad_csv))
    
    def test_validate_data(self, sample_csv_data):
        """Test data validation."""
        loader = DataLoader()
        
        # Valid data
        assert loader.validate_data(sample_csv_data) is True
        
        # Invalid data - missing columns
        invalid_data = sample_csv_data.drop(columns=["Close"])
        assert loader.validate_data(invalid_data) is False
        
        # Invalid data - negative prices
        invalid_data = sample_csv_data.copy()
        invalid_data.loc[0, "Close"] = -1
        assert loader.validate_data(invalid_data) is False
        
        # Invalid data - negative volume
        invalid_data = sample_csv_data.copy()
        invalid_data.loc[0, "Volume"] = -1
        assert loader.validate_data(invalid_data) is False
    
    @patch("tbs.data.loader.yf.Ticker")
    def test_load_yfinance_data(self, mock_ticker, sample_csv_data):
        """Test loading yfinance data."""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_csv_data.set_index("Datetime")
        mock_ticker.return_value = mock_ticker_instance
        
        loader = DataLoader()
        data = loader.load_data(
            source="yfinance",
            ticker="BTC-USD",
            start="2022-01-01",
            end="2022-01-10",
            interval="1d",
            cache=False,
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert "Datetime" in data.columns
    
    @patch("tbs.data.loader.yf.Ticker")
    def test_load_yfinance_empty_data(self, mock_ticker):
        """Test loading yfinance data with empty response."""
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        loader = DataLoader()
        
        with pytest.raises(ValueError):
            loader.load_data(
                source="yfinance",
                ticker="INVALID-TICKER",
                start="2022-01-01",
                end="2022-01-10",
                interval="1d",
                cache=False,
            )
    
    def test_get_cache_file_path(self):
        """Test cache file path generation."""
        loader = DataLoader()
        
        path = loader._get_cache_file_path("BTC-USD", "2022-01-01", "2022-01-31", "1d")
        
        assert isinstance(path, Path)
        assert "BTC-USD" in str(path)
        assert "1d" in str(path)
        assert "2022-01-01" in str(path)
        assert "2022-01-31" in str(path)
    
    def test_clear_cache(self, tmp_path):
        """Test cache clearing."""
        # Create some dummy cache files
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        (cache_dir / "BTC-USD_1d_2022-01-01_2022-01-31.parquet").touch()
        (cache_dir / "AAPL_1d_2022-01-01_2022-01-31.parquet").touch()
        
        loader = DataLoader(cache_dir=str(cache_dir))
        
        # Clear specific ticker
        loader.clear_cache("BTC-USD")
        assert not (cache_dir / "BTC-USD_1d_2022-01-01_2022-01-31.parquet").exists()
        assert (cache_dir / "AAPL_1d_2022-01-01_2022-01-31.parquet").exists()
        
        # Clear all
        loader.clear_cache()
        assert not any(cache_dir.glob("*.parquet"))
    
    def test_get_cache_info(self, tmp_path):
        """Test cache info retrieval."""
        # Create some dummy cache files
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        (cache_dir / "test1.parquet").touch()
        (cache_dir / "test2.parquet").touch()
        
        loader = DataLoader(cache_dir=str(cache_dir))
        info = loader.get_cache_info()
        
        assert info["total_files"] == 2
        assert info["cache_dir"] == str(cache_dir)
        assert "test1.parquet" in info["files"]
        assert "test2.parquet" in info["files"]
    
    def test_unknown_source(self):
        """Test unknown data source."""
        loader = DataLoader()
        
        with pytest.raises(ValueError):
            loader.load_data(source="unknown")

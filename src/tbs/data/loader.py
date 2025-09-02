"""Data loading and caching functionality."""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional, Union
import structlog

logger = structlog.get_logger(__name__)


class DataLoader:
    """Load and cache financial data from various sources."""
    
    def __init__(self, cache_dir: str = "data/cache") -> None:
        """Initialize data loader.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(
        self,
        source: str = "yfinance",
        ticker: Optional[str] = None,
        csv_path: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        cache: bool = True,
    ) -> pd.DataFrame:
        """Load financial data from specified source.
        
        Args:
            source: Data source ("yfinance" or "csv")
            ticker: Ticker symbol for yfinance
            csv_path: Path to CSV file
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
            cache: Whether to use caching
            
        Returns:
            DataFrame with OHLCV data
        """
        if source == "yfinance":
            return self._load_yfinance_data(ticker, start, end, interval, cache)
        elif source == "csv":
            return self._load_csv_data(csv_path)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def _load_yfinance_data(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        cache: bool = True,
    ) -> pd.DataFrame:
        """Load data from yfinance with optional caching.
        
        Args:
            ticker: Ticker symbol
            start: Start date
            end: End date
            interval: Data interval
            cache: Whether to use caching
            
        Returns:
            DataFrame with OHLCV data
        """
        if cache:
            cache_file = self._get_cache_file_path(ticker, start, end, interval)
            if cache_file.exists():
                logger.info("Loading cached data", ticker=ticker, cache_file=str(cache_file))
                return pd.read_parquet(cache_file)
        
        logger.info("Fetching data from yfinance", ticker=ticker, start=start, end=end, interval=interval)
        
        try:
            # Download data
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start, end=end, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Reset index to make Datetime a column
            data = data.reset_index()
            
            # Rename columns to match expected format
            data = data.rename(columns={
                'Date': 'Datetime',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Ensure Datetime is datetime type
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            # Sort by datetime
            data = data.sort_values('Datetime').reset_index(drop=True)
            
            # Cache the data
            if cache:
                cache_file = self._get_cache_file_path(ticker, start, end, interval)
                data.to_parquet(cache_file)
                logger.info("Cached data", cache_file=str(cache_file))
            
            return data
            
        except Exception as e:
            logger.error("Failed to fetch data from yfinance", ticker=ticker, error=str(e))
            raise
    
    def _load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info("Loading data from CSV", csv_path=csv_path)
        
        try:
            data = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert Datetime to datetime type
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            # Sort by datetime
            data = data.sort_values('Datetime').reset_index(drop=True)
            
            return data
            
        except Exception as e:
            logger.error("Failed to load CSV data", csv_path=csv_path, error=str(e))
            raise
    
    def _get_cache_file_path(
        self, ticker: str, start: Optional[str], end: Optional[str], interval: str
    ) -> Path:
        """Generate cache file path.
        
        Args:
            ticker: Ticker symbol
            start: Start date
            end: End date
            interval: Data interval
            
        Returns:
            Path to cache file
        """
        # Create filename from parameters
        start_str = start or "start"
        end_str = end or "end"
        filename = f"{ticker}_{interval}_{start_str}_{end_str}.parquet"
        
        return self.cache_dir / filename
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cached data.
        
        Args:
            ticker: Specific ticker to clear (if None, clear all)
        """
        if ticker:
            # Clear specific ticker cache
            pattern = f"{ticker}_*.parquet"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info("Cleared cache file", file=str(cache_file))
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
                logger.info("Cleared cache file", file=str(cache_file))
    
    def get_cache_info(self) -> dict:
        """Get information about cached data.
        
        Returns:
            Dictionary with cache information
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))
        
        info = {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            "files": [f.name for f in cache_files]
        }
        
        return info
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data format and quality.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
        """
        # Check required columns
        required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data['Datetime']):
            return False
        
        # Check for missing values
        if data[required_columns].isnull().any().any():
            return False
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        if (data[price_columns] <= 0).any().any():
            return False
        
        # Check for negative volume
        if (data['Volume'] < 0).any():
            return False
        
        # Check that High >= Low
        if (data['High'] < data['Low']).any():
            return False
        
        return True

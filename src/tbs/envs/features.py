"""Technical indicators and feature engineering."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class FeatureEngineer:
    """Engineer features from OHLCV data."""
    
    def __init__(
        self,
        sma_periods: Optional[List[int]] = None,
        ema_periods: Optional[List[int]] = None,
        rsi_length: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_length: int = 20,
        bollinger_std: float = 2.0,
    ) -> None:
        """Initialize feature engineer.
        
        Args:
            sma_periods: Periods for Simple Moving Averages
            ema_periods: Periods for Exponential Moving Averages
            rsi_length: Length for RSI calculation
            macd_fast: Fast period for MACD
            macd_slow: Slow period for MACD
            macd_signal: Signal period for MACD
            bollinger_length: Length for Bollinger Bands
            bollinger_std: Standard deviation multiplier for Bollinger Bands
        """
        self.sma_periods = sma_periods or [10, 20, 50]
        self.ema_periods = ema_periods or [12, 26]
        self.rsi_length = rsi_length
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_length = bollinger_length
        self.bollinger_std = bollinger_std
    
    def get_feature_dim(self) -> int:
        """Get the dimension of the feature vector."""
        # Price features: returns, high_low_ratio, volume_ratio
        price_features = 3
        
        # SMA features
        sma_features = len(self.sma_periods)
        
        # EMA features
        ema_features = len(self.ema_periods)
        
        # RSI
        rsi_features = 1
        
        # MACD
        macd_features = 3  # MACD line, signal line, histogram
        
        # Bollinger Bands
        bb_features = 3  # upper, middle, lower bands
        
        # Volume features
        volume_features = 2  # volume_sma_ratio, volume_ema_ratio
        
        return (price_features + sma_features + ema_features + rsi_features + 
                macd_features + bb_features + volume_features)
    
    def calculate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate all features for the given data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Feature array
        """
        features = []
        
        # Price features
        features.extend(self._calculate_price_features(data))
        
        # Moving averages
        features.extend(self._calculate_sma_features(data))
        features.extend(self._calculate_ema_features(data))
        
        # RSI
        features.extend(self._calculate_rsi_features(data))
        
        # MACD
        features.extend(self._calculate_macd_features(data))
        
        # Bollinger Bands
        features.extend(self._calculate_bollinger_features(data))
        
        # Volume features
        features.extend(self._calculate_volume_features(data))
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_price_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate price-based features."""
        if len(data) < 2:
            return [0.0, 0.0, 0.0]
        
        # Price returns
        returns = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
        
        # High-Low ratio
        high_low_ratio = data['High'].iloc[-1] / data['Low'].iloc[-1]
        
        # Volume ratio (current volume / average volume)
        avg_volume = data['Volume'].mean()
        volume_ratio = data['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1.0
        
        return [returns, high_low_ratio, volume_ratio]
    
    def _calculate_sma_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate Simple Moving Average features."""
        features = []
        
        for period in self.sma_periods:
            if len(data) >= period:
                sma = data['Close'].rolling(window=period).mean().iloc[-1]
                current_price = data['Close'].iloc[-1]
                sma_ratio = current_price / sma if sma > 0 else 1.0
            else:
                sma_ratio = 1.0
            
            features.append(sma_ratio)
        
        return features
    
    def _calculate_ema_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate Exponential Moving Average features."""
        features = []
        
        for period in self.ema_periods:
            if len(data) >= period:
                ema = data['Close'].ewm(span=period).mean().iloc[-1]
                current_price = data['Close'].iloc[-1]
                ema_ratio = current_price / ema if ema > 0 else 1.0
            else:
                ema_ratio = 1.0
            
            features.append(ema_ratio)
        
        return features
    
    def _calculate_rsi_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate RSI features."""
        if len(data) < self.rsi_length + 1:
            return [50.0]  # Neutral RSI
        
        # Calculate price changes
        delta = data['Close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.rsi_length).mean().iloc[-1]
        avg_losses = losses.rolling(window=self.rsi_length).mean().iloc[-1]
        
        # Calculate RSI
        if avg_losses == 0:
            rsi = 100.0
        else:
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
        
        return [rsi]
    
    def _calculate_macd_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate MACD features."""
        if len(data) < self.macd_slow:
            return [0.0, 0.0, 0.0]
        
        # Calculate EMAs
        ema_fast = data['Close'].ewm(span=self.macd_fast).mean()
        ema_slow = data['Close'].ewm(span=self.macd_slow).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        # Normalize by current price
        current_price = data['Close'].iloc[-1]
        if current_price > 0:
            macd_norm = macd_line.iloc[-1] / current_price
            signal_norm = signal_line.iloc[-1] / current_price
            histogram_norm = histogram.iloc[-1] / current_price
        else:
            macd_norm = signal_norm = histogram_norm = 0.0
        
        return [macd_norm, signal_norm, histogram_norm]
    
    def _calculate_bollinger_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate Bollinger Bands features."""
        if len(data) < self.bollinger_length:
            return [1.0, 1.0, 1.0]
        
        # Calculate SMA and standard deviation
        sma = data['Close'].rolling(window=self.bollinger_length).mean()
        std = data['Close'].rolling(window=self.bollinger_length).std()
        
        # Calculate bands
        upper_band = sma + (std * self.bollinger_std)
        lower_band = sma - (std * self.bollinger_std)
        
        # Normalize by current price
        current_price = data['Close'].iloc[-1]
        if current_price > 0:
            upper_norm = upper_band.iloc[-1] / current_price
            middle_norm = sma.iloc[-1] / current_price
            lower_norm = lower_band.iloc[-1] / current_price
        else:
            upper_norm = middle_norm = lower_norm = 1.0
        
        return [upper_norm, middle_norm, lower_norm]
    
    def _calculate_volume_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate volume-based features."""
        if len(data) < max(self.sma_periods):
            return [1.0, 1.0]
        
        current_volume = data['Volume'].iloc[-1]
        
        # Volume SMA ratio
        volume_sma = data['Volume'].rolling(window=min(self.sma_periods)).mean().iloc[-1]
        volume_sma_ratio = current_volume / volume_sma if volume_sma > 0 else 1.0
        
        # Volume EMA ratio
        volume_ema = data['Volume'].ewm(span=min(self.ema_periods)).mean().iloc[-1]
        volume_ema_ratio = current_volume / volume_ema if volume_ema > 0 else 1.0
        
        return [volume_sma_ratio, volume_ema_ratio]

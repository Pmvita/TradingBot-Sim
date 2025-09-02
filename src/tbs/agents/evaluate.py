"""Evaluation functionality for trained agents."""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import structlog
from omegaconf import OmegaConf

from ..envs.trading_env import TradingEnv
from ..data.loader import DataLoader
from ..portfolio.wallet import Wallet
from ..viz.plots import create_evaluation_plots
from .registry import registry

logger = structlog.get_logger(__name__)


class Evaluator:
    """Evaluator for trained RL agents."""
    
    def __init__(self, run_path: str) -> None:
        """Initialize evaluator.
        
        Args:
            run_path: Path to training run directory
        """
        self.run_path = Path(run_path)
        self.config = OmegaConf.load(self.run_path / "config.yaml")
        self.data_loader = DataLoader()
        self.model = None
        
        # Load trained model
        algo = self.config["train"]["algo"]
        agent_class = registry.get_agent_class(algo)
        self.model = agent_class.load(str(self.run_path / "model.zip"))
    
    def evaluate(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        baseline_strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate trained agent against baselines.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            baseline_strategies: List of baseline strategies to compare
            
        Returns:
            Evaluation results
        """
        # Load test data
        logger.info("Loading evaluation data", ticker=ticker, start=start_date, end=end_date)
        data = self.data_loader.load_data(
            source="yfinance",
            ticker=ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            cache=True,
        )
        
        # Create environment
        env = self._create_environment(data)
        
        # Evaluate agent
        agent_results = self._evaluate_agent(env, data)
        
        # Evaluate baselines
        baseline_results = {}
        if baseline_strategies:
            for strategy in baseline_strategies:
                baseline_results[strategy] = self._evaluate_baseline(strategy, data)
        
        # Combine results
        results = {
            "agent": agent_results,
            "baselines": baseline_results,
            "metadata": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
                "algorithm": self.config["train"]["algo"],
            }
        }
        
        # Save results
        self._save_evaluation_results(results)
        
        # Create plots
        self._create_evaluation_plots(results, data)
        
        return results
    
    def _create_environment(self, data: pd.DataFrame) -> TradingEnv:
        """Create trading environment.
        
        Args:
            data: OHLCV data
            
        Returns:
            Trading environment
        """
        env_config = self.config["env"]
        
        return TradingEnv(
            data=data,
            window_size=env_config["window_size"],
            fee_bps=env_config["fee_bps"],
            slippage_bps=env_config["slippage_bps"],
            reward_scheme=env_config["reward"],
            max_drawdown=env_config["max_drawdown"],
            stop_loss=env_config.get("stop_loss", 0.2),
            allow_short=env_config.get("allow_short", False),
            starting_cash=self.config["portfolio"]["starting_cash"],
            leverage=self.config["portfolio"]["leverage"],
            position_sizing=self.config["portfolio"]["position_sizing"],
            seed=self.config["train"]["seed"],
        )
    
    def _evaluate_agent(self, env: TradingEnv, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate trained agent.
        
        Args:
            env: Trading environment
            data: OHLCV data
            
        Returns:
            Agent evaluation results
        """
        logger.info("Evaluating trained agent")
        
        # Run evaluation
        obs, _ = env.reset()
        done = False
        portfolio_values = []
        actions = []
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            portfolio_values.append(info["portfolio_value"])
            actions.append(action)
        
        # Calculate metrics
        returns = self._calculate_returns(portfolio_values)
        metrics = self._calculate_metrics(returns, portfolio_values, actions)
        
        return {
            "portfolio_values": portfolio_values,
            "actions": actions,
            "metrics": metrics,
        }
    
    def _evaluate_baseline(self, strategy: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate baseline strategy.
        
        Args:
            strategy: Baseline strategy name
            data: OHLCV data
            
        Returns:
            Baseline evaluation results
        """
        logger.info("Evaluating baseline", strategy=strategy)
        
        if strategy == "buy_and_hold":
            return self._evaluate_buy_and_hold(data)
        elif strategy == "sma_crossover":
            return self._evaluate_sma_crossover(data)
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")
    
    def _evaluate_buy_and_hold(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate buy-and-hold strategy.
        
        Args:
            data: OHLCV data
            
        Returns:
            Buy-and-hold results
        """
        starting_cash = self.config["portfolio"]["starting_cash"]
        initial_price = data["Close"].iloc[0]
        final_price = data["Close"].iloc[-1]
        
        # Buy at start
        units = starting_cash / initial_price
        final_value = units * final_price
        
        # Calculate returns
        returns = [(final_value - starting_cash) / starting_cash]
        portfolio_values = [starting_cash, final_value]
        
        metrics = self._calculate_metrics(returns, portfolio_values, [])
        
        return {
            "portfolio_values": portfolio_values,
            "actions": [],
            "metrics": metrics,
        }
    
    def _evaluate_sma_crossover(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate SMA crossover strategy.
        
        Args:
            data: OHLCV data
            
        Returns:
            SMA crossover results
        """
        short_period = self.config["evaluation"]["sma_short"]
        long_period = self.config["evaluation"]["sma_long"]
        
        # Calculate SMAs
        data = data.copy()
        data["SMA_short"] = data["Close"].rolling(window=short_period).mean()
        data["SMA_long"] = data["Close"].rolling(window=long_period).mean()
        
        # Generate signals
        data["signal"] = 0
        data.loc[data["SMA_short"] > data["SMA_long"], "signal"] = 1  # Buy
        data.loc[data["SMA_short"] < data["SMA_long"], "signal"] = -1  # Sell
        
        # Simulate trading
        starting_cash = self.config["portfolio"]["starting_cash"]
        cash = starting_cash
        position = 0.0
        portfolio_values = [starting_cash]
        actions = []
        
        for i in range(1, len(data)):
            signal = data["signal"].iloc[i]
            price = data["Close"].iloc[i]
            
            if signal == 1 and position == 0:  # Buy signal
                position = cash / price
                cash = 0.0
                actions.append(1)  # Buy
            elif signal == -1 and position > 0:  # Sell signal
                cash = position * price
                position = 0.0
                actions.append(2)  # Sell
            else:
                actions.append(0)  # Hold
            
            portfolio_value = cash + (position * price)
            portfolio_values.append(portfolio_value)
        
        # Calculate metrics
        returns = self._calculate_returns(portfolio_values)
        metrics = self._calculate_metrics(returns, portfolio_values, actions)
        
        return {
            "portfolio_values": portfolio_values,
            "actions": actions,
            "metrics": metrics,
        }
    
    def _calculate_returns(self, portfolio_values: List[float]) -> List[float]:
        """Calculate returns from portfolio values.
        
        Args:
            portfolio_values: List of portfolio values
            
        Returns:
            List of returns
        """
        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
        return returns
    
    def _calculate_metrics(
        self, returns: List[float], portfolio_values: List[float], actions: List[int]
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            returns: List of returns
            portfolio_values: List of portfolio values
            actions: List of actions
            
        Returns:
            Dictionary of metrics
        """
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        portfolio_array = np.array(portfolio_values)
        
        # Basic metrics
        total_return = (portfolio_array[-1] - portfolio_array[0]) / portfolio_array[0]
        
        # Volatility
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        risk_free_rate = self.config["evaluation"]["risk_free_rate"]
        excess_returns = returns_array - (risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(portfolio_array)
        drawdowns = (running_max - portfolio_array) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Win rate (for trading strategies)
        if actions:
            buy_actions = [i for i, a in enumerate(actions) if a == 1]
            sell_actions = [i for i, a in enumerate(actions) if a == 2]
            
            if len(buy_actions) > 0 and len(sell_actions) > 0:
                trades = []
                for buy_idx in buy_actions:
                    for sell_idx in sell_actions:
                        if sell_idx > buy_idx:
                            trade_return = (portfolio_values[sell_idx] - portfolio_values[buy_idx]) / portfolio_values[buy_idx]
                            trades.append(trade_return)
                            break
                
                if trades:
                    win_rate = sum(1 for t in trades if t > 0) / len(trades)
                    avg_trade_return = np.mean(trades)
                else:
                    win_rate = 0.0
                    avg_trade_return = 0.0
            else:
                win_rate = 0.0
                avg_trade_return = 0.0
        else:
            win_rate = 0.0
            avg_trade_return = 0.0
        
        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_trade_return": avg_trade_return,
            "num_trades": len([a for a in actions if a in [1, 2]]),
        }
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results.
        
        Args:
            results: Evaluation results
        """
        eval_dir = self.run_path / "eval"
        eval_dir.mkdir(exist_ok=True)
        
        # Save results as JSON
        results_path = eval_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics as CSV
        metrics_data = []
        for strategy, data in results.items():
            if strategy == "metadata":
                continue
            metrics = data["metrics"]
            metrics["strategy"] = strategy
            metrics_data.append(metrics)
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = eval_dir / "metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
        
        logger.info("Saved evaluation results", eval_dir=str(eval_dir))
    
    def _create_evaluation_plots(self, results: Dict[str, Any], data: pd.DataFrame) -> None:
        """Create evaluation plots.
        
        Args:
            results: Evaluation results
            data: OHLCV data
        """
        eval_dir = self.run_path / "eval"
        plots_dir = eval_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        create_evaluation_plots(results, data, plots_dir)
        
        logger.info("Created evaluation plots", plots_dir=str(plots_dir))


def evaluate_agent(
    run_path: str,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    baseline_strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience function to evaluate an agent.
    
    Args:
        run_path: Path to training run directory
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval
        baseline_strategies: List of baseline strategies to compare
        
    Returns:
        Evaluation results
    """
    evaluator = Evaluator(run_path)
    results = evaluator.evaluate(ticker, start_date, end_date, interval, baseline_strategies)
    
    logger.info("Evaluation completed successfully")
    return results

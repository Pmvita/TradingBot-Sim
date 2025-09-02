"""Training functionality for RL agents."""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import structlog
from omegaconf import OmegaConf

from ..envs.trading_env import TradingEnv
from ..data.loader import DataLoader
from .registry import registry

logger = structlog.get_logger(__name__)


class Trainer:
    """Trainer for reinforcement learning agents."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.data_loader = DataLoader()
        self.run_dir = None
        self.model = None
        
    def train(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        run_name: Optional[str] = None,
    ) -> str:
        """Train an RL agent.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            run_name: Custom run name
            
        Returns:
            Path to saved model
        """
        # Load data
        logger.info("Loading training data", ticker=ticker, start=start_date, end=end_date)
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
        
        # Create agent
        algo = self.config["train"]["algo"]
        
        # Filter out training-specific parameters that shouldn't be passed to agent constructor
        training_params = {"total_timesteps", "algo", "seed", "train_split", "test_split", "eval_freq", "save_freq", "tensorboard_log"}
        agent_kwargs = {k: v for k, v in self.config["train"].items() 
                       if k not in training_params}
        
        agent = registry.create_agent(
            algo=algo,
            env=env,
            policy_kwargs=agent_kwargs.get("policy_kwargs", {}),
            **{k: v for k, v in agent_kwargs.items() if k != "policy_kwargs"}
        )
        
        # Setup run directory
        self.run_dir = self._setup_run_dir(algo, run_name)
        
        # Train agent
        logger.info("Starting training", algo=algo, timesteps=self.config["train"]["total_timesteps"])
        start_time = time.time()
        
        agent.learn(
            total_timesteps=self.config["train"]["total_timesteps"],
            progress_bar=True,
        )
        
        training_time = time.time() - start_time
        logger.info("Training completed", training_time=training_time)
        
        # Save model and config
        model_path = self.run_dir / "model.zip"
        agent.save(str(model_path))
        
        # Save configuration
        config_path = self.run_dir / "config.yaml"
        OmegaConf.save(self.config, config_path)
        
        # Save training metrics
        self._save_training_metrics(agent, training_time)
        
        self.model = agent
        return str(model_path)
    
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
    
    def _setup_run_dir(self, algo: str, run_name: Optional[str] = None) -> Path:
        """Setup run directory.
        
        Args:
            algo: Algorithm name
            run_name: Custom run name
            
        Returns:
            Run directory path
        """
        timestamp = int(time.time())
        if run_name:
            dir_name = f"{algo}_{run_name}_{timestamp}"
        else:
            dir_name = f"{algo}_{timestamp}"
        
        run_dir = Path("runs") / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlink to latest
        latest_link = Path("runs") / f"{algo}_latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
        
        return run_dir
    
    def _save_training_metrics(self, agent, training_time: float) -> None:
        """Save training metrics.
        
        Args:
            agent: Trained agent
            training_time: Training duration
        """
        metrics = {
            "training_time": training_time,
            "total_timesteps": self.config["train"]["total_timesteps"],
            "algorithm": self.config["train"]["algo"],
            "timestamp": int(time.time()),
        }
        
        # Add agent-specific metrics if available
        if hasattr(agent, "logger") and hasattr(agent.logger, "name_to_value"):
            for key, value in agent.logger.name_to_value.items():
                if isinstance(value, (int, float)):
                    metrics[f"train_{key}"] = value
        
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Saved training metrics", metrics_path=str(metrics_path))


def train_agent(
    config_path: str,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience function to train an agent.
    
    Args:
        config_path: Path to configuration file
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval
        run_name: Custom run name
        
    Returns:
        Path to saved model
    """
    # Load configuration
    if config is None:
        config = OmegaConf.load(config_path)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train agent
    model_path = trainer.train(ticker, start_date, end_date, interval, run_name)
    
    logger.info("Training completed successfully", model_path=model_path)
    return model_path

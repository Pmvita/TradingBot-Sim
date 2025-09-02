"""Command-line interface for Trading Bot Simulator."""

import os
import sys
from pathlib import Path
from typing import Optional

import structlog
import typer
from omegaconf import OmegaConf

from .agents.evaluate import evaluate_agent
from .agents.train import train_agent
from .data.loader import DataLoader

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

app = typer.Typer(help="Trading Bot Simulator - RL Trading Environment")


@app.command()
def fetch(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Data interval"),
    csv: Optional[str] = typer.Option(
        None, "--csv", help="Use CSV file instead of yfinance"
    ),
    cache: bool = typer.Option(
        True, "--cache/--no-cache", help="Enable/disable caching"
    ),
) -> None:
    """Fetch historical market data."""
    try:
        data_loader = DataLoader()

        if csv:
            data = data_loader.load_data(source="csv", csv_path=csv)
            logger.info("Loaded CSV data", csv_path=csv, rows=len(data))
        else:
            data = data_loader.load_data(
                source="yfinance",
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                cache=cache,
            )
            logger.info("Fetched data", ticker=ticker, rows=len(data))

        print(f"✅ Successfully loaded {len(data)} data points")
        print(f"📊 Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
        print(
            f"💰 Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}"
        )

    except Exception as e:
        logger.error("Failed to fetch data", error=str(e))
        print(f"❌ Error: {e}")
        sys.exit(1)


@app.command()
def train(
    algo: str = typer.Option(..., "--algo", "-a", help="RL algorithm (ppo, dqn, a2c)"),
    config: str = typer.Option(..., "--config", "-c", help="Configuration file path"),
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Data interval"),
    total_timesteps: Optional[int] = typer.Option(
        None, "--total-timesteps", help="Override total timesteps"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Custom run name"),
) -> None:
    """Train a reinforcement learning agent."""
    try:
        # Load configuration
        if not os.path.exists(config):
            print(f"❌ Configuration file not found: {config}")
            sys.exit(1)

        # Load configuration with defaults
        base_config = OmegaConf.load("configs/default.yaml")
        config_overrides = OmegaConf.load(config)
        cfg = OmegaConf.merge(base_config, config_overrides)
        if total_timesteps:
            cfg.train.total_timesteps = total_timesteps
        if seed:
            cfg.train.seed = seed

        # Train agent
        model_path = train_agent(
            config_path=config,
            ticker=ticker,
            start_date=start,
            end_date=end,
            interval=interval,
            run_name=run_name,
            config=cfg,
        )

        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"🤖 Algorithm: {algo.upper()}")
        print(f"📊 Ticker: {ticker}")
        print(f"📅 Period: {start} to {end}")

    except Exception as e:
        logger.error("Training failed", error=str(e))
        print(f"❌ Training failed: {e}")
        sys.exit(1)


@app.command()
def eval(
    run: str = typer.Option(..., "--run", "-r", help="Path to training run directory"),
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Data interval"),
    baseline: str = typer.Option(
        "all",
        "--baseline",
        "-b",
        help="Baseline strategies (buy_and_hold, sma_crossover, all)",
    ),
) -> None:
    """Evaluate a trained agent against baselines."""
    try:
        # Determine baseline strategies
        if baseline == "all":
            baseline_strategies = ["buy_and_hold", "sma_crossover"]
        else:
            baseline_strategies = [baseline]

        # Evaluate agent
        results = evaluate_agent(
            run_path=run,
            ticker=ticker,
            start_date=start,
            end_date=end,
            interval=interval,
            baseline_strategies=baseline_strategies,
        )

        # Print summary
        print("✅ Evaluation completed successfully!")
        print(f"📊 Results saved to: {run}/eval/")
        print(f"📈 Plots saved to: {run}/eval/plots/")

        # Print metrics summary
        print("\n📋 Performance Summary:")
        print("-" * 50)

        for strategy, data in results.items():
            if strategy == "metadata":
                continue

            if "metrics" in data:  # Add safety check
                metrics = data["metrics"]
                print(f"\n{strategy.upper()}:")
                print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
                print(f"  Num Trades: {metrics.get('num_trades', 0)}")
            else:
                print(f"\n{strategy.upper()}: No metrics available")

    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


@app.command()
def backtest_baselines(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    strategy: str = typer.Option(
        ..., "--strategy", help="Strategy (buy_and_hold, sma_crossover)"
    ),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Data interval"),
    short: Optional[int] = typer.Option(10, "--short", help="Short SMA period"),
    long: Optional[int] = typer.Option(50, "--long", help="Long SMA period"),
) -> None:
    """Backtest baseline strategies."""
    try:
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_data(
            source="yfinance",
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            cache=True,
        )

        # Create dummy config for evaluation
        config = {
            "env": {
                "window_size": 64,
                "fee_bps": 10,
                "slippage_bps": 5,
                "reward": "profit_increment",
                "max_drawdown": 0.3,
                "stop_loss": 0.2,
                "allow_short": False,
            },
            "portfolio": {
                "starting_cash": 10000,
                "allow_short": False,
                "leverage": 1.0,
                "position_sizing": 1.0,
            },
            "evaluation": {
                "sma_short": short,
                "sma_long": long,
                "risk_free_rate": 0.02,
            },
            "train": {
                "algo": "ppo",
                "seed": 42,
            },
        }

        # Create temporary evaluator
        class TempEvaluator:
            def __init__(self, config):
                self.config = config
                self.data_loader = DataLoader()

        evaluator = TempEvaluator(config)

        # Evaluate baseline
        if strategy == "buy_and_hold":
            results = evaluator._evaluate_buy_and_hold(data)
        elif strategy == "sma_crossover":
            results = evaluator._evaluate_sma_crossover(data)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Print results
        print(f"✅ Backtest completed for {strategy}!")
        print(f"📊 Ticker: {ticker}")
        print(f"📅 Period: {start} to {end}")
        print(f"💰 Initial Value: ${config['portfolio']['starting_cash']:,.2f}")
        print(f"💰 Final Value: ${results['portfolio_values'][-1]:,.2f}")
        print(f"📈 Total Return: {results['metrics']['total_return']:.2%}")
        print(f"📊 Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {results['metrics']['max_drawdown']:.2%}")

    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        print(f"❌ Backtest failed: {e}")
        sys.exit(1)


@app.command()
def gui(
    interface: str = typer.Option(
        "streamlit", "--interface", "-i", help="GUI interface (streamlit, dash)"
    ),
    port: int = typer.Option(8501, "--port", "-p", help="Port to run the GUI on"),
    host: str = typer.Option(
        "localhost", "--host", "-h", help="Host to run the GUI on"
    ),
) -> None:
    """Launch the Trading Bot Simulator GUI."""
    try:
        if interface == "streamlit":
            import sys

            import streamlit.web.cli as stcli

            # Set up streamlit arguments
            sys.argv = [
                "streamlit",
                "run",
                str(Path(__file__).parent / "gui" / "streamlit_app.py"),
                "--server.port",
                str(port),
                "--server.address",
                host,
                "--server.headless",
                "true",
            ]

            # Run streamlit
            stcli.main()

        elif interface == "dash":
            # Import dash app dynamically
            from .gui.dash_app import app as dash_app

            # Run dash app
            dash_app.run_server(debug=False, host=host, port=port)

        else:
            print(f"❌ Unknown interface: {interface}. Available: streamlit, dash")
            sys.exit(1)

    except Exception as e:
        logger.error("GUI failed to start", error=str(e))
        print(f"❌ GUI failed to start: {e}")
        sys.exit(1)


@app.command()
def info() -> None:
    """Show system information."""
    print("🤖 Trading Bot Simulator")
    print("=" * 40)
    print("📦 Version: 0.1.0")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print("📊 Available Algorithms: PPO, DQN, A2C")
    print("📈 Data Sources: yfinance, CSV")
    print("📋 Baseline Strategies: buy_and_hold, sma_crossover")
    print("\n📚 Quick Start:")
    print("  1. tbs fetch --ticker BTC-USD --start 2022-01-01 --end 2022-12-31")
    print(
        "  2. tbs train --algo ppo --config configs/ppo.yaml --ticker BTC-USD --start 2022-01-01 --end 2022-06-30"
    )
    print(
        "  3. tbs eval --run runs/ppo/latest --ticker BTC-USD --start 2022-07-01 --end 2022-12-31"
    )


if __name__ == "__main__":
    app()

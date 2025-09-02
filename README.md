# Trading Bot Simulator

A reinforcement learning trading bot simulator that trades with mock currency while consuming real historical market data for charts. The agent learns Buy, Sell, Hold actions to maximize profit and optionally the Sharpe ratio.

![Trading Bot Simulator](docs/images/equity_curve.png)

## 🚀 Features

- **🤖 Reinforcement Learning Agents**: PPO, DQN, and A2C algorithms from Stable Baselines3
- **📈 Real Market Data**: Historical OHLCV data via yfinance or CSV files
- **💰 Mock Trading**: No real money, no exchange keys, safe simulation environment
- **📊 Comprehensive Evaluation**: Compare against buy-and-hold and SMA crossover baselines
- **📈 Rich Visualizations**: Equity curves, drawdown analysis, action frequency, rolling Sharpe ratio
- **🎨 Complete GUI**: Both Streamlit and Dash interfaces for easy interaction
- **🏭 Production Ready**: Complete CI/CD, Docker support, comprehensive testing
- **🔬 Reproducible**: Deterministic training with fixed seeds

## 🎯 Quick Start

### Option 1: GUI (Recommended for Beginners)
```bash
# 1. Install dependencies
make setup

# 2. Launch the GUI
make gui

# 3. Open your browser to http://localhost:8501
# 4. Use the GUI to fetch data, train models, and evaluate performance
```

### Option 2: Command Line
```bash
# 1. Install dependencies
make setup

# 2. Fetch sample data
poetry run tbs fetch --ticker BTC-USD --start 2022-01-01 --end 2022-12-31 --interval 1d

# 3. Train a PPO agent
poetry run tbs train --algo ppo --config configs/ppo.yaml --ticker BTC-USD

# 4. Evaluate against baselines
poetry run tbs eval --run runs/ppo/latest --baseline buy_and_hold

# 5. View results
open runs/ppo/latest/eval/plots/equity_curve.png
```

## Installation

### Prerequisites

- Python 3.9+
- Poetry (recommended) or pip

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone <repository-url>
cd trading-bot-simulator
poetry install
poetry run tbs --help
```

### Using pip

```bash
pip install -e .
tbs --help
```

## 🎨 GUI Interface

The Trading Bot Simulator comes with two complete GUI interfaces for easy interaction:

### Streamlit GUI (Recommended)
```bash
# Launch Streamlit GUI
make gui
# or
python3 gui_launcher.py streamlit 8501
```

**Features:**
- 📊 **Dashboard**: Real-time metrics and portfolio overview
- 📈 **Data Management**: Fetch and visualize market data
- 🎯 **Training**: Configure and train RL agents with visual progress
- 📋 **Evaluation**: Compare models against baselines
- 💰 **Live Trading**: Real-time trading simulation
- ⚙️ **Settings**: Configuration management

### Dash GUI (Advanced)
```bash
# Launch Dash GUI
make gui-dash
# or
python3 gui_launcher.py dash 8050
```

**Features:**
- 🎨 **Professional UI**: Bootstrap-based responsive design
- 🔄 **Real-time Updates**: Live data updates and callbacks
- 📱 **Mobile Support**: Responsive design for all devices
- 🔧 **Advanced Controls**: More granular parameter control

### GUI Screenshots

![Dashboard](docs/images/gui_dashboard.png)
![Training](docs/images/gui_training.png)
![Evaluation](docs/images/gui_evaluation.png)

## 📖 Usage Examples

### Fetch Historical Data

```bash
# Fetch Bitcoin data from yfinance
poetry run tbs fetch --ticker BTC-USD --start 2021-01-01 --end 2023-12-31 --interval 1d

# Use local CSV file
poetry run tbs fetch --csv data/my_data.csv
```

### Train Different Agents

```bash
# Train PPO agent
poetry run tbs train --algo ppo --config configs/ppo.yaml --ticker BTC-USD

# Train DQN agent
poetry run tbs train --algo dqn --config configs/dqn.yaml --ticker AAPL

# Train A2C agent with custom parameters
poetry run tbs train --algo a2c --config configs/a2c.yaml --ticker ETH-USD --total-timesteps 100000
```

### Evaluate Performance

```bash
# Evaluate against buy-and-hold baseline
poetry run tbs eval --run runs/ppo/latest --baseline buy_and_hold

# Evaluate against SMA crossover
poetry run tbs eval --run runs/dqn/latest --baseline sma_crossover --short 10 --long 50

# Compare multiple baselines
poetry run tbs eval --run runs/a2c/latest --baseline all
```

### Backtest Baselines

```bash
# Test SMA crossover strategy
poetry run tbs backtest-baselines --ticker AAPL --strategy sma_crossover --short 10 --long 50

# Test buy-and-hold
poetry run tbs backtest-baselines --ticker BTC-USD --strategy buy_and_hold
```

## Configuration

The simulator uses YAML configuration files with environment variable overrides. Key configuration options:

```yaml
# configs/default.yaml
data:
  source: yfinance  # or csv
  ticker: BTC-USD
  interval: 1d
  cache: true

env:
  window_size: 64
  fee_bps: 10  # 0.1% fee
  slippage_bps: 5  # 0.05% slippage
  reward: profit_increment
  max_drawdown: 0.3

train:
  algo: ppo
  total_timesteps: 100000
  seed: 42
  train_split: 0.8
  test_split: 0.2

portfolio:
  starting_cash: 10000
  allow_short: false
  leverage: 1.0
```

## Project Structure

```
trading-bot-simulator/
├── src/tbs/                    # Main package
│   ├── envs/                   # Trading environment
│   ├── agents/                 # RL agents and training
│   ├── data/                   # Data loading and caching
│   ├── portfolio/              # Mock portfolio management
│   ├── gui/                    # GUI interfaces (Streamlit & Dash)
│   └── viz/                    # Visualization utilities
├── configs/                    # Configuration files
├── data/                       # Data storage
├── runs/                       # Training outputs
├── tests/                      # Test suite
├── docs/                       # Documentation
├── docker/                     # Docker configuration
└── gui_launcher.py            # GUI launcher script
```

## Environment Details

### Observation Space
- Price returns (normalized)
- Volume indicators
- Technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands)
- Current position, cash, holdings
- Unrealized PnL

### Action Space
- **Buy**: Purchase asset units
- **Sell**: Sell asset units  
- **Hold**: No action

### Reward Functions
- **profit_increment**: Change in portfolio value
- **risk_adjusted**: Profit minus volatility penalty
- **trade_penalty**: Small negative reward per trade

## Evaluation Metrics

The simulator evaluates performance using:
- **Total Return**: Overall portfolio growth
- **CAGR**: Compound Annual Growth Rate
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Average Trade PnL**: Mean profit per trade

## Docker Support

```bash
# Build and run with Docker
docker compose build
docker compose run trainer tbs train --algo ppo --config configs/ppo.yaml
```

## Development

```bash
# Setup development environment
make setup

# Launch GUI for development
make gui

# Format code
make fmt

# Run linting
make lint

# Run type checking
make typecheck

# Run tests
make test

# Run smoke test
make smoke

# Build documentation
make docs
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **IMPORTANT**: This is a simulation tool for educational and research purposes only. 

- No real money is traded
- No live market orders are placed
- Results are not indicative of real trading performance
- Past performance does not guarantee future results
- Always do your own research before making investment decisions

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/your-username/trading-bot-simulator/issues)
- 💬 [Discussions](https://github.com/your-username/trading-bot-simulator/discussions)

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{trading_bot_simulator,
  title={Trading Bot Simulator: A Reinforcement Learning Trading Environment},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/trading-bot-simulator}
}
```

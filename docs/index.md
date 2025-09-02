# Trading Bot Simulator Documentation

Welcome to the comprehensive documentation for the Trading Bot Simulator - a complete reinforcement learning trading environment with both CLI and GUI interfaces.

## 📚 Quick Navigation

### 🚀 Getting Started
- [Setup Guide](setup.md) - Installation and initial configuration
- [Quick Start](usage.md) - Your first trading bot in 5 minutes
- [GUI Guide](gui.md) - Complete GUI interface documentation

### 🎯 Core Features
- [Environment Details](env.md) - Trading environment specifications
- [Algorithms](algorithms.md) - PPO, DQN, and A2C implementation details
- [Data Management](data.md) - Market data fetching and processing
- [Evaluation](evaluation.md) - Performance metrics and baseline comparisons

### 🛠️ Development
- [API Reference](api.md) - Complete API documentation
- [Configuration](config.md) - YAML configuration guide
- [Docker](docker.md) - Container deployment
- [Contributing](CONTRIBUTING.md) - Development guidelines

## 🎨 GUI Interfaces

The Trading Bot Simulator provides two complete GUI interfaces:

### Streamlit GUI (Recommended)
- **Best for**: Beginners, data scientists, quick prototyping
- **Features**: Interactive widgets, real-time charts, easy parameter tuning
- **Launch**: `make gui` or `python3 gui_launcher.py streamlit 8501`

### Dash GUI (Advanced)
- **Best for**: Professional users, custom dashboards, real-time trading
- **Features**: Bootstrap UI, advanced callbacks, mobile responsive
- **Launch**: `make gui-dash` or `python3 gui_launcher.py dash 8050`

## 🎯 Key Features

### 🤖 Reinforcement Learning
- **PPO**: Proximal Policy Optimization for continuous action spaces
- **DQN**: Deep Q-Network for discrete action spaces
- **A2C**: Advantage Actor-Critic for policy gradient methods

### 📈 Market Data
- **yfinance**: Real-time and historical data from Yahoo Finance
- **CSV Support**: Import custom datasets
- **Caching**: Fast data access with intelligent caching
- **Multiple Intervals**: 1m, 5m, 15m, 1h, 1d

### 💰 Trading Environment
- **Mock Portfolio**: Safe simulation with no real money
- **Transaction Costs**: Realistic fees and slippage modeling
- **Risk Management**: Position sizing, drawdown limits
- **Multiple Assets**: Support for any tradable asset

### 📊 Evaluation & Analysis
- **Baseline Comparison**: Buy-and-hold, SMA crossover strategies
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate
- **Visualizations**: Equity curves, drawdown analysis, trade logs
- **Statistical Analysis**: Rolling metrics, correlation analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Layer     │    │   CLI Layer     │    │   Core Engine   │
│                 │    │                 │    │                 │
│ • Streamlit     │    │ • Typer CLI     │    │ • Trading Env   │
│ • Dash          │    │ • Commands      │    │ • RL Agents     │
│ • Real-time     │    │ • Scripts       │    │ • Portfolio     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Data Layer    │
                    │                 │
                    │ • yfinance      │
                    │ • CSV loader    │
                    │ • Caching       │
                    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: GUI (Recommended)
```bash
# Install and setup
make setup

# Launch GUI
make gui

# Open browser to http://localhost:8501
```

### Option 2: Command Line
```bash
# Install and setup
make setup

# Fetch data
poetry run tbs fetch --ticker BTC-USD --start 2022-01-01 --end 2022-12-31

# Train agent
poetry run tbs train --algo ppo --config configs/ppo.yaml --ticker BTC-USD

# Evaluate
poetry run tbs eval --run runs/ppo/latest --baseline all
```

## 📊 Example Workflow

1. **Data Collection**: Fetch historical market data
2. **Environment Setup**: Configure trading parameters
3. **Agent Training**: Train RL agent on historical data
4. **Evaluation**: Compare against baseline strategies
5. **Analysis**: Review performance metrics and visualizations
6. **Optimization**: Iterate on hyperparameters and strategies

## 🔧 Configuration

The simulator uses YAML configuration files with environment variable overrides:

```yaml
# configs/default.yaml
data:
  source: yfinance
  ticker: BTC-USD
  interval: 1d
  cache: true

env:
  window_size: 64
  fee_bps: 10
  slippage_bps: 5
  reward: profit_increment
  max_drawdown: 0.3

train:
  algo: ppo
  total_timesteps: 100000
  seed: 42

portfolio:
  starting_cash: 10000
  allow_short: false
  leverage: 1.0
```

## 🐳 Docker Support

```bash
# Build image
docker build -t trading-bot-simulator .

# Run with GUI
docker run -p 8501:8501 trading-bot-simulator make gui

# Run training
docker run trading-bot-simulator poetry run tbs train --algo ppo
```

## 📈 Performance Metrics

The simulator tracks comprehensive performance metrics:

- **Returns**: Total return, CAGR, annualized return
- **Risk**: Volatility, max drawdown, VaR
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trading**: Win rate, profit factor, average trade
- **Drawdown**: Recovery time, underwater periods

## 🔒 Security & Safety

- **No Live Trading**: All trading is simulation only
- **Local Data**: Data is cached locally, no external dependencies
- **No API Keys**: Uses public data sources only
- **Isolated Environment**: Each session is completely isolated

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code style guidelines
- Testing requirements
- Pull request process
- Development setup

## 📞 Support

- 📖 **Documentation**: This site and [README.md](../README.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/trading-bot-simulator/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/trading-bot-simulator/discussions)
- 📧 **Email**: your-email@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Happy Trading! 🚀📈**

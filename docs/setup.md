# Setup Guide

This guide will help you set up the Trading Bot Simulator on your system with both CLI and GUI interfaces.

## Prerequisites

- Python 3.11 or higher
- Poetry (recommended) or pip
- Git

## Installation

### Using Poetry (Recommended)

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/trading-bot-simulator.git
   cd trading-bot-simulator
   ```

3. **Install dependencies**:
   ```bash
   make setup
   # or manually:
   poetry install
   ```

4. **Verify installation**:
   ```bash
   poetry run tbs --help
   ```

### Using pip

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/trading-bot-simulator.git
   cd trading-bot-simulator
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   tbs --help
   ```

## Quick Start

### Option 1: GUI (Recommended for Beginners)

1. **Launch the GUI**:
   ```bash
   make gui
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Use the GUI** to:
   - Fetch market data
   - Train RL agents
   - Evaluate performance
   - View results

### Option 2: Command Line

1. **Fetch sample data**:
   ```bash
   poetry run tbs fetch --ticker BTC-USD --start 2022-01-01 --end 2022-01-31 --interval 1d
   ```

2. **Train a PPO agent**:
   ```bash
   poetry run tbs train --algo ppo --config configs/ppo.yaml --ticker BTC-USD --start 2022-01-01 --end 2022-01-15 --total-timesteps 1000
   ```

3. **Evaluate the agent**:
   ```bash
   poetry run tbs eval --run runs/ppo/latest --ticker BTC-USD --start 2022-01-16 --end 2022-01-31 --baseline all
   ```

## GUI Setup

The Trading Bot Simulator includes two GUI interfaces:

### Streamlit GUI (Recommended)
```bash
# Launch Streamlit GUI
make gui
# or
python3 gui_launcher.py streamlit 8501
```

**Features:**
- Interactive data visualization
- Real-time training progress
- Easy parameter tuning
- Portfolio tracking
- Performance analysis

### Dash GUI (Advanced)
```bash
# Launch Dash GUI
make gui-dash
# or
python3 gui_launcher.py dash 8050
```

**Features:**
- Professional Bootstrap UI
- Real-time callbacks
- Mobile responsive design
- Advanced controls

## Development Setup

For contributors and developers:

1. **Install development dependencies**:
   ```bash
   poetry install --with dev
   ```

2. **Install pre-commit hooks**:
   ```bash
   poetry run pre-commit install
   ```

3. **Run tests**:
   ```bash
   make test
   ```

4. **Run linting**:
   ```bash
   make lint
   ```

5. **Launch GUI for development**:
   ```bash
   make gui
   ```

## Configuration

The simulator uses YAML configuration files. Key configuration options:

- **Data settings**: Source, ticker, date range, interval
- **Environment settings**: Window size, fees, slippage, reward scheme
- **Training settings**: Algorithm, timesteps, hyperparameters
- **Portfolio settings**: Starting cash, leverage, position sizing

See `configs/default.yaml` for all available options.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the correct directory and have installed dependencies
2. **yfinance errors**: Check your internet connection and ticker symbol
3. **Memory issues**: Reduce window size or use smaller datasets
4. **CUDA errors**: Set `CUDA_VISIBLE_DEVICES=""` to use CPU only
5. **GUI not loading**: Check if ports 8501 or 8050 are available

### Getting Help

- Check the [documentation](docs/)
- Search [existing issues](https://github.com/your-username/trading-bot-simulator/issues)
- Create a new issue with detailed information

## Platform-Specific Notes

### Windows

- Install Visual C++ Build Tools for some dependencies
- Use WSL for better compatibility

### macOS

- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies

### Linux

- Install build essentials: `sudo apt-get install build-essential`
- Install Python development headers: `sudo apt-get install python3-dev`

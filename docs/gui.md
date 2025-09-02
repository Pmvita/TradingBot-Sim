# 🤖 Trading Bot Simulator - GUI

A complete, production-ready GUI for the Trading Bot Simulator with both Streamlit and Dash interfaces.

## 🚀 Quick Start

### Option 1: Using Make Commands
```bash
# Launch Streamlit GUI (recommended)
make gui

# Launch Dash GUI
make gui-dash
```

### Option 2: Using CLI Commands
```bash
# Launch Streamlit GUI
poetry run tbs gui --interface streamlit --port 8501

# Launch Dash GUI
poetry run tbs gui --interface dash --port 8050
```

### Option 3: Direct Python Script
```bash
# Launch Streamlit GUI
python gui_launcher.py streamlit 8501

# Launch Dash GUI
python gui_launcher.py dash 8050
```

## 🎯 Features

### 📊 Dashboard
- **Real-time metrics**: Data points, trained models, live trading status, portfolio value
- **Interactive charts**: Price charts, portfolio performance, equity curves
- **Recent activity**: Latest actions and performance overview

### 📈 Data Management
- **Market data fetching**: Support for yfinance and CSV uploads
- **Data preview**: Interactive tables and charts
- **Cache management**: Clear and manage cached data
- **Multiple intervals**: 1d, 1h, 15m, 5m, 1m

### 🎯 Training
- **Algorithm selection**: PPO, DQN, A2C with visual configuration
- **Parameter tuning**: Interactive sliders and inputs for all hyperparameters
- **Real-time progress**: Training progress bars and status updates
- **Model management**: View, delete, and manage trained models

### 📋 Evaluation
- **Model selection**: Choose from trained models
- **Baseline comparison**: Buy & Hold, SMA Crossover, All strategies
- **Performance metrics**: Total return, Sharpe ratio, max drawdown, win rate
- **Visual results**: Equity curves and performance charts

### 💰 Live Trading Simulation
- **Real-time trading**: Start/stop live trading simulation
- **Portfolio tracking**: Real-time portfolio value and PnL
- **Trade logging**: Complete trade history and analysis
- **Risk management**: Position sizing and drawdown limits

### ⚙️ Settings
- **Configuration management**: Save and load settings
- **Default values**: Set default tickers, intervals, cash amounts
- **Logging control**: Adjust log levels and output
- **Chart customization**: Theme selection and update frequencies

## 🎨 Interface Comparison

### Streamlit Interface
- **Pros**: 
  - Faster development and iteration
  - Built-in caching and session state
  - Excellent for data science workflows
  - Simple deployment
- **Cons**: 
  - Less customizable styling
  - Limited real-time updates
  - Single-threaded

### Dash Interface
- **Pros**: 
  - Highly customizable with Bootstrap
  - Real-time updates with callbacks
  - Better for complex interactions
  - Professional appearance
- **Cons**: 
  - More complex development
  - Steeper learning curve
  - Requires more setup

## 🔧 Configuration

### Environment Variables
```bash
# GUI Settings
export TBS_GUI_PORT=8501
export TBS_GUI_HOST=localhost
export TBS_GUI_INTERFACE=streamlit

# Data Settings
export TBS_DEFAULT_TICKER=BTC-USD
export TBS_DEFAULT_INTERVAL=1d
export TBS_CACHE_ENABLED=true
```

### Settings File
The GUI automatically creates a `settings.json` file:
```json
{
  "default_ticker": "BTC-USD",
  "default_interval": "1d",
  "cache_enabled": true,
  "default_start_cash": 10000,
  "default_window_size": 64,
  "auto_save": true,
  "log_level": "INFO",
  "max_cache_size": 1000,
  "update_frequency": "5s",
  "chart_theme": "plotly"
}
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
make setup

# Launch GUI
make gui
```

### Docker Deployment
```bash
# Build Docker image
docker build -t trading-bot-simulator .

# Run with GUI
docker run -p 8501:8501 trading-bot-simulator make gui
```

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run src/tbs/gui/streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create trading-bot-simulator
git push heroku main
```

## 📱 Mobile Support

Both interfaces are responsive and work on mobile devices:
- **Streamlit**: Automatic mobile optimization
- **Dash**: Bootstrap responsive design

## 🔒 Security

- **No live trading**: All trading is simulation only
- **Local data**: Data is cached locally
- **No API keys**: Uses public data sources only
- **Session isolation**: Each user session is isolated

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Use different port
   make gui --port 8502
   ```

2. **Import errors**
   ```bash
   # Reinstall dependencies
   poetry install
   ```

3. **Data loading issues**
   ```bash
   # Clear cache
   poetry run tbs fetch --clear-cache
   ```

### Debug Mode
```bash
# Enable debug logging
export TBS_LOG_LEVEL=DEBUG
make gui
```

## 📈 Performance Tips

1. **Use caching**: Enable data caching for faster loading
2. **Limit data range**: Use smaller date ranges for faster processing
3. **Reduce update frequency**: Lower update frequency for better performance
4. **Close unused tabs**: Close unused browser tabs to reduce memory usage

## 🔮 Future Enhancements

- [ ] Real-time market data integration
- [ ] Advanced charting with TradingView integration
- [ ] Multi-asset portfolio management
- [ ] Social trading features
- [ ] Mobile app development
- [ ] Cloud-based model training
- [ ] API for external integrations

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the main project documentation
- Open an issue on GitHub
- Join the community Discord

---

**Happy Trading! 🚀📈**

#!/usr/bin/env python3
"""Modern Streamlit GUI for Trading Bot Simulator."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tbs.data.loader import DataLoader
from tbs.agents.train import train_agent
from tbs.agents.evaluate import evaluate_agent
from tbs.envs.trading_env import TradingEnv
from tbs.portfolio.wallet import Wallet
from tbs.envs.features import FeatureEngineer
from tbs.envs.reward_schemes import RewardScheme
from omegaconf import OmegaConf

# Page configuration
st.set_page_config(
    page_title="Trading Bot Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-username/trading-bot-simulator',
        'Report a bug': 'https://github.com/your-username/trading-bot-simulator/issues',
        'About': '# Trading Bot Simulator\nA complete RL trading environment with GUI'
    }
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #9467bd;
        --light-bg: #f8f9fa;
        --dark-bg: #2c3e50;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: var(--success-color);
        box-shadow: 0 0 8px var(--success-color);
    }
    
    .status-inactive {
        background-color: var(--warning-color);
        box-shadow: 0 0 8px var(--warning-color);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--light-bg);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* Success/Error messages */
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--success-color);
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--warning-color);
        margin: 1rem 0;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background-color: var(--light-bg);
        border: none;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Data tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Loading animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'live_trading' not in st.session_state:
    st.session_state.live_trading = False
if 'portfolio_history' not in st.session_state:
    st.session_state.portfolio_history = []
if 'current_portfolio' not in st.session_state:
    st.session_state.current_portfolio = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

def main():
    """Main application with modern UI."""
    # Modern header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Trading Bot Simulator</h1>
        <p>Advanced Reinforcement Learning Trading Environment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        # Page selection with icons
        pages = {
            "📊 Dashboard": "Dashboard",
            "📈 Data Management": "Data",
            "🎯 Training": "Training", 
            "📋 Evaluation": "Evaluation",
            "💰 Live Trading": "Live Trading",
            "⚙️ Settings": "Settings"
        }
        
        selected_page = st.selectbox(
            "Choose a page",
            list(pages.keys()),
            index=list(pages.values()).index(st.session_state.current_page)
        )
        st.session_state.current_page = pages[selected_page]
        
        # Quick stats in sidebar
        st.markdown("---")
        st.markdown("## 📊 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Points", len(st.session_state.data) if st.session_state.data is not None else 0)
        with col2:
            st.metric("Models", len(st.session_state.trained_models))
        
        # Live trading status
        if st.session_state.live_trading:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                        padding: 0.5rem; border-radius: 8px; text-align: center;">
                <span class="status-indicator status-active"></span>
                <strong>Live Trading Active</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
                        padding: 0.5rem; border-radius: 8px; text-align: center;">
                <span class="status-indicator status-inactive"></span>
                <strong>Live Trading Inactive</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("---")
        st.markdown("## ⚡ Quick Actions")
        
        if st.button("🚀 Fetch Sample Data", use_container_width=True):
            fetch_sample_data()
        
        if st.button("🎯 Quick Train", use_container_width=True):
            quick_train()
        
        if st.button("📊 Quick Eval", use_container_width=True):
            quick_eval()
    
    # Page routing
    if st.session_state.current_page == "Dashboard":
        dashboard_page()
    elif st.session_state.current_page == "Data":
        data_management_page()
    elif st.session_state.current_page == "Training":
        training_page()
    elif st.session_state.current_page == "Evaluation":
        evaluation_page()
    elif st.session_state.current_page == "Live Trading":
        live_trading_page()
    elif st.session_state.current_page == "Settings":
        settings_page()

def dashboard_page():
    """Modern dashboard with enhanced visuals."""
    st.markdown("## 📊 Dashboard Overview")
    
    # Overview metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Data Points</h3>
            <h2>{}</h2>
        </div>
        """.format(len(st.session_state.data) if st.session_state.data is not None else 0), 
        unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Trained Models</h3>
            <h2>{}</h2>
        </div>
        """.format(len(st.session_state.trained_models)), 
        unsafe_allow_html=True)
    
    with col3:
        live_status = "🟢 Active" if st.session_state.live_trading else "🔴 Inactive"
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Live Trading</h3>
            <h2>{}</h2>
        </div>
        """.format(live_status), 
        unsafe_allow_html=True)
    
    with col4:
        portfolio_value = st.session_state.current_portfolio.get_portfolio_value(0) if st.session_state.current_portfolio else 0
        st.markdown("""
        <div class="metric-card">
            <h3>💵 Portfolio Value</h3>
            <h2>${:,.2f}</h2>
        </div>
        """.format(portfolio_value), 
        unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        if st.session_state.data is not None:
            st.markdown("### 📈 Market Data")
            fig = create_enhanced_price_chart(st.session_state.data)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: var(--light-bg); border-radius: 12px;">
                <h3>📊 No Data Available</h3>
                <p>Fetch some market data to get started!</p>
                <button onclick="window.location.href='#data-management'">Fetch Data</button>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Recent activity
        st.markdown("### 📋 Recent Activity")
        
        if st.session_state.trained_models:
            for model in st.session_state.trained_models[-3:]:  # Show last 3
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <strong>{model['algo'].upper()}</strong><br>
                    <small>{model['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No training activity yet")
    
    # Portfolio performance
    if st.session_state.portfolio_history:
        st.markdown("### 💰 Portfolio Performance")
        fig = create_enhanced_portfolio_chart(st.session_state.portfolio_history)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def data_management_page():
    """Enhanced data management with modern UI."""
    st.markdown("## 📈 Data Management")
    
    # Tabs for different data operations
    tab1, tab2, tab3 = st.tabs(["🚀 Fetch Data", "📊 Data Preview", "🗑️ Cache Management"])
    
    with tab1:
        st.markdown("### Fetch Market Data")
        
        # Modern form layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Data Source")
            source = st.selectbox(
                "Select data source",
                ["yfinance", "CSV Upload"],
                help="Choose between live market data or local CSV file"
            )
            
            if source == "yfinance":
                ticker = st.text_input(
                    "Ticker Symbol",
                    value="BTC-USD",
                    help="Enter stock/crypto ticker (e.g., BTC-USD, AAPL, TSLA)"
                )
                
                interval = st.selectbox(
                    "Time Interval",
                    ["1d", "1h", "15m", "5m", "1m"],
                    help="Select data frequency"
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload CSV file",
                    type=['csv'],
                    help="Upload your own market data CSV file"
                )
        
        with col2:
            st.markdown("#### 📅 Date Range")
            start_date = st.date_input(
                "Start Date",
                value=datetime(2022, 1, 1),
                help="Select start date for data"
            )
            
            end_date = st.date_input(
                "End Date", 
                value=datetime(2022, 12, 31),
                help="Select end date for data"
            )
            
            cache_data = st.checkbox(
                "Cache Data",
                value=True,
                help="Cache data for faster future access"
            )
        
        # Fetch button with loading state
        if st.button("🚀 Fetch Data", type="primary", use_container_width=True):
            with st.spinner("Fetching market data..."):
                try:
                    if source == "yfinance":
                        data_loader = DataLoader()
                        data = data_loader.load_data(
                            source="yfinance",
                            ticker=ticker,
                            start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"),
                            interval=interval,
                            cache=cache_data
                        )
                    else:
                        if uploaded_file is not None:
                            data_loader = DataLoader()
                            data = data_loader.load_data(source="csv", csv_path=uploaded_file)
                        else:
                            st.error("Please upload a CSV file")
                            return
                    
                    st.session_state.data = data
                    
                    st.success(f"✅ Successfully loaded {len(data):,} data points")
                    
                    # Show data summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Date Range", f"{data['Datetime'].min().date()} to {data['Datetime'].max().date()}")
                    with col2:
                        st.metric("Price Range", f"${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
                    with col3:
                        st.metric("Volume", f"{data['Volume'].sum():,.0f}")
                        
                except Exception as e:
                    st.error(f"❌ Error fetching data: {str(e)}")
    
    with tab2:
        st.markdown("### Data Preview")
        
        if st.session_state.data is not None:
            # Data statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Data Statistics")
                stats_df = st.session_state.data.describe()
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Price Distribution")
                fig = px.histogram(
                    st.session_state.data, 
                    x='Close', 
                    nbins=50,
                    title="Price Distribution"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Raw data preview
            st.markdown("#### 📋 Raw Data")
            st.dataframe(
                st.session_state.data.head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No data available. Please fetch data first.")
    
    with tab3:
        st.markdown("### Cache Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                try:
                    data_loader = DataLoader()
                    data_loader.clear_cache()
                    st.success("✅ Cache cleared successfully")
                except Exception as e:
                    st.error(f"❌ Error clearing cache: {str(e)}")
        
        with col2:
            if st.button("📊 Cache Info", use_container_width=True):
                try:
                    data_loader = DataLoader()
                    cache_info = data_loader.get_cache_info()
                    st.json(cache_info)
                except Exception as e:
                    st.error(f"❌ Error getting cache info: {str(e)}")

def training_page():
    """Enhanced training page with modern UI."""
    st.markdown("## 🎯 Model Training")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please fetch data first in the Data Management page.")
        return
    
    # Training configuration in tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "🚀 Training", "📋 Models"])
    
    with tab1:
        st.markdown("### Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 Algorithm Selection")
            algo = st.selectbox(
                "Algorithm",
                ["ppo", "dqn", "a2c"],
                help="Choose reinforcement learning algorithm"
            )
            
            config_file = st.selectbox(
                "Config File",
                [f"configs/{algo}.yaml"],
                help="Select configuration file"
            )
            
            total_timesteps = st.number_input(
                "Total Timesteps",
                min_value=1000,
                value=10000,
                step=1000,
                help="Number of training steps"
            )
            
            seed = st.number_input(
                "Random Seed",
                value=42,
                help="Random seed for reproducibility"
            )
        
        with col2:
            st.markdown("#### 🎯 Environment Settings")
            window_size = st.slider(
                "Window Size",
                min_value=10,
                max_value=128,
                value=64,
                help="Number of historical data points to include"
            )
            
            fee_bps = st.slider(
                "Fee (basis points)",
                min_value=0,
                max_value=100,
                value=10,
                help="Trading fee in basis points"
            )
            
            slippage_bps = st.slider(
                "Slippage (basis points)",
                min_value=0,
                max_value=50,
                value=5,
                help="Market slippage in basis points"
            )
            
            reward_scheme = st.selectbox(
                "Reward Scheme",
                ["profit_increment", "risk_adjusted", "trade_penalty"],
                help="Choose reward function"
            )
        
        # Advanced settings in expander
        with st.expander("🔧 Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.0003,
                    format="%.4f",
                    help="Learning rate for the algorithm"
                )
                
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=16,
                    max_value=256,
                    value=64,
                    help="Training batch size"
                )
                
                gamma = st.slider(
                    "Gamma",
                    min_value=0.8,
                    max_value=0.99,
                    value=0.99,
                    format="%.2f",
                    help="Discount factor"
                )
            
            with col2:
                max_drawdown = st.slider(
                    "Max Drawdown",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.3,
                    format="%.1f",
                    help="Maximum allowed drawdown"
                )
                
                starting_cash = st.number_input(
                    "Starting Cash",
                    min_value=1000,
                    max_value=100000,
                    value=10000,
                    help="Initial portfolio value"
                )
                
                allow_short = st.checkbox(
                    "Allow Short Selling",
                    value=False,
                    help="Enable short selling"
                )
    
    with tab2:
        st.markdown("### Start Training")
        
        # Training progress
        if st.button("🎯 Start Training", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training thread
            def train_thread():
                try:
                    # Create custom config
                    config = OmegaConf.load(config_file)
                    config.train.total_timesteps = total_timesteps
                    config.train.seed = seed
                    config.env.window_size = window_size
                    config.env.fee_bps = fee_bps
                    config.env.slippage_bps = slippage_bps
                    config.env.reward = reward_scheme
                    config.env.max_drawdown = max_drawdown
                    config.portfolio.starting_cash = starting_cash
                    config.portfolio.allow_short = allow_short
                    
                    # Save config
                    config_path = f"configs/custom_{algo}_{int(time.time())}.yaml"
                    OmegaConf.save(config, config_path)
                    
                    # Train agent
                    model_path = train_agent(
                        config_path=config_path,
                        ticker="BTC-USD",
                        start_date=st.session_state.data['Datetime'].min().strftime("%Y-%m-%d"),
                        end_date=st.session_state.data['Datetime'].max().strftime("%Y-%m-%d"),
                        interval="1d"
                    )
                    
                    # Add to trained models
                    st.session_state.trained_models.append({
                        'algo': algo,
                        'model_path': model_path,
                        'config_path': config_path,
                        'timestamp': datetime.now(),
                        'total_timesteps': total_timesteps
                    })
                    
                    status_text.success("✅ Training completed successfully!")
                    
                except Exception as e:
                    status_text.error(f"❌ Training failed: {str(e)}")
            
            # Start training in thread
            thread = threading.Thread(target=train_thread)
            thread.start()
            
            # Simulate progress
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)
                status_text.text(f"Training... {i+1}%")
    
    with tab3:
        st.markdown("### Trained Models")
        
        if st.session_state.trained_models:
            for i, model in enumerate(st.session_state.trained_models):
                with st.expander(f"🤖 {model['algo'].upper()} - {model['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Algorithm:** {model['algo'].upper()}")
                        st.markdown(f"**Timesteps:** {model['total_timesteps']:,}")
                    
                    with col2:
                        st.markdown(f"**Path:** {model['model_path']}")
                        st.markdown(f"**Config:** {model['config_path']}")
                    
                    with col3:
                        if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                            st.session_state.trained_models.pop(i)
                            st.rerun()
                        
                        if st.button(f"📊 Evaluate", key=f"eval_{i}"):
                            st.session_state.current_page = "Evaluation"
                            st.rerun()
        else:
            st.info("No trained models available. Start training to see models here.")

def evaluation_page():
    """Enhanced evaluation page with modern UI."""
    st.markdown("## 📋 Model Evaluation")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train a model first.")
        return
    
    # Model selection
    st.markdown("### Model Selection")
    
    model_options = [f"{m['algo'].upper()} - {m['timestamp'].strftime('%Y-%m-%d %H:%M')}" for m in st.session_state.trained_models]
    selected_model_idx = st.selectbox(
        "Select Model",
        range(len(model_options)),
        format_func=lambda x: model_options[x]
    )
    selected_model = st.session_state.trained_models[selected_model_idx]
    
    # Evaluation configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Evaluation Settings")
        baseline = st.selectbox(
            "Baseline Strategy",
            ["buy_and_hold", "sma_crossover", "all"],
            help="Choose baseline strategy for comparison"
        )
        
        test_start = st.date_input(
            "Test Start Date",
            value=datetime(2023, 1, 1),
            help="Start date for evaluation period"
        )
        
        test_end = st.date_input(
            "Test End Date",
            value=datetime(2023, 12, 31),
            help="End date for evaluation period"
        )
    
    with col2:
        st.markdown("#### ⚙️ Baseline Parameters")
        sma_short = st.number_input(
            "SMA Short Period",
            min_value=5,
            max_value=50,
            value=10,
            help="Short period for SMA crossover"
        )
        
        sma_long = st.number_input(
            "SMA Long Period",
            min_value=10,
            max_value=200,
            value=50,
            help="Long period for SMA crossover"
        )
    
    # Run evaluation
    if st.button("📊 Run Evaluation", type="primary", use_container_width=True):
        with st.spinner("Running evaluation..."):
            try:
                # Run evaluation
                results = evaluate_agent(
                    run_path=selected_model['model_path'],
                    ticker="BTC-USD",
                    start_date=test_start.strftime("%Y-%m-%d"),
                    end_date=test_end.strftime("%Y-%m-%d"),
                    interval="1d",
                    baseline_strategies=[baseline] if baseline != "all" else ["buy_and_hold", "sma_crossover"]
                )
                
                st.success("✅ Evaluation completed successfully!")
                
                # Display results in tabs
                tab1, tab2 = st.tabs(["📊 Metrics", "📈 Charts"])
                
                with tab1:
                    st.markdown("### Performance Metrics")
                    
                    # Create metrics dataframe
                    metrics_data = []
                    for strategy, data in results.items():
                        if strategy == "metadata":
                            continue
                        metrics = data.get("metrics", {})
                        metrics_data.append({
                            "Strategy": strategy.upper(),
                            "Total Return": f"{metrics.get('total_return', 0):.2%}",
                            "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
                            "Max Drawdown": f"{metrics.get('max_drawdown', 0):.2%}",
                            "Win Rate": f"{metrics.get('win_rate', 0):.2%}",
                            "Num Trades": metrics.get('num_trades', 0)
                        })
                    
                    if metrics_data:
                        df_metrics = pd.DataFrame(metrics_data)
                        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                
                with tab2:
                    st.markdown("### Performance Charts")
                    
                    if "equity_curves" in results:
                        fig = create_enhanced_equity_chart(results["equity_curves"])
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")

def live_trading_page():
    """Enhanced live trading page with modern UI."""
    st.markdown("## 💰 Live Trading Simulation")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please fetch data first in the Data Management page.")
        return
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train a model first.")
        return
    
    # Model selection
    st.markdown("### Model Selection")
    
    model_options = [f"{m['algo'].upper()} - {m['timestamp'].strftime('%Y-%m-%d %H:%M')}" for m in st.session_state.trained_models]
    selected_model_idx = st.selectbox(
        "Select Model for Live Trading",
        range(len(model_options)),
        format_func=lambda x: model_options[x]
    )
    selected_model = st.session_state.trained_models[selected_model_idx]
    
    # Live trading controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 Start Live Trading", type="primary", use_container_width=True, disabled=st.session_state.live_trading):
            st.session_state.live_trading = True
            st.session_state.current_portfolio = Wallet(starting_cash=10000.0)
            st.session_state.portfolio_history = []
            st.rerun()
    
    with col2:
        if st.button("🔴 Stop Live Trading", type="secondary", use_container_width=True, disabled=not st.session_state.live_trading):
            st.session_state.live_trading = False
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Portfolio", type="secondary", use_container_width=True):
            st.session_state.current_portfolio = Wallet(starting_cash=10000.0)
            st.session_state.portfolio_history = []
            st.rerun()
    
    # Live trading status
    if st.session_state.live_trading:
        st.success("🟢 Live trading is active")
        
        # Portfolio overview
        st.markdown("### Portfolio Overview")
        
        if st.session_state.current_portfolio:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cash", f"${st.session_state.current_portfolio.cash:,.2f}")
            
            with col2:
                st.metric("Position", f"{st.session_state.current_portfolio.position:.4f}")
            
            with col3:
                st.metric("Unrealized PnL", f"${st.session_state.current_portfolio.unrealized_pnl:,.2f}")
            
            with col4:
                portfolio_value = st.session_state.current_portfolio.get_portfolio_value(0)
                st.metric("Total Value", f"${portfolio_value:,.2f}")
        
        # Live chart
        st.markdown("### Live Trading Chart")
        
        if st.session_state.portfolio_history:
            fig = create_enhanced_live_trading_chart(st.session_state.data, st.session_state.portfolio_history)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Trading log
        st.markdown("### Trading Log")
        if st.session_state.current_portfolio and st.session_state.current_portfolio.trades:
            trades_df = pd.DataFrame(st.session_state.current_portfolio.trades)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("🔴 Live trading is inactive")

def settings_page():
    """Enhanced settings page with modern UI."""
    st.markdown("## ⚙️ Settings")
    
    # Settings in tabs
    tab1, tab2 = st.tabs(["🔧 General Settings", "📊 Advanced Settings"])
    
    with tab1:
        st.markdown("### General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_ticker = st.text_input(
                "Default Ticker",
                value="BTC-USD",
                help="Default ticker symbol"
            )
            
            default_interval = st.selectbox(
                "Default Interval",
                ["1d", "1h", "15m", "5m", "1m"],
                help="Default data interval"
            )
            
            cache_enabled = st.checkbox(
                "Enable Caching",
                value=True,
                help="Enable data caching"
            )
        
        with col2:
            default_start_cash = st.number_input(
                "Default Starting Cash",
                min_value=1000,
                max_value=100000,
                value=10000,
                help="Default starting portfolio value"
            )
            
            default_window_size = st.slider(
                "Default Window Size",
                min_value=10,
                max_value=128,
                value=64,
                help="Default window size for training"
            )
            
            auto_save = st.checkbox(
                "Auto-save Results",
                value=True,
                help="Automatically save training results"
            )
    
    with tab2:
        st.markdown("### Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            log_level = st.selectbox(
                "Log Level",
                ["INFO", "DEBUG", "WARNING", "ERROR"],
                help="Logging level"
            )
            
            max_cache_size = st.number_input(
                "Max Cache Size (MB)",
                min_value=100,
                max_value=10000,
                value=1000,
                help="Maximum cache size in MB"
            )
        
        with col2:
            update_frequency = st.selectbox(
                "Update Frequency",
                ["1s", "5s", "10s", "30s", "1m"],
                help="GUI update frequency"
            )
            
            chart_theme = st.selectbox(
                "Chart Theme",
                ["plotly", "plotly_white", "plotly_dark"],
                help="Chart theme"
            )
    
    # Save settings
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        settings = {
            "default_ticker": default_ticker,
            "default_interval": default_interval,
            "cache_enabled": cache_enabled,
            "default_start_cash": default_start_cash,
            "default_window_size": default_window_size,
            "auto_save": auto_save,
            "log_level": log_level,
            "max_cache_size": max_cache_size,
            "update_frequency": update_frequency,
            "chart_theme": chart_theme
        }
        
        # Save to file
        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=2)
        
        st.success("✅ Settings saved successfully!")

# Helper functions for enhanced charts
def create_enhanced_price_chart(data):
    """Create enhanced price chart with modern styling."""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data['Datetime'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='#2ca02c',
        decreasing_line_color='#d62728'
    ))
    
    fig.update_layout(
        title="Market Data",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        template="plotly_white",
        hovermode='x unified',
        showlegend=False
    )
    
    return fig

def create_enhanced_portfolio_chart(portfolio_history):
    """Create enhanced portfolio performance chart."""
    if not portfolio_history:
        return go.Figure()
    
    fig = go.Figure()
    
    dates = [entry['date'] for entry in portfolio_history]
    values = [entry['value'] for entry in portfolio_history]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def create_enhanced_equity_chart(equity_curves):
    """Create enhanced equity curve chart."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (strategy, data) in enumerate(equity_curves.items()):
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['values'],
            mode='lines',
            name=strategy.upper(),
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title="Equity Curves Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def create_enhanced_live_trading_chart(data, portfolio_history):
    """Create enhanced live trading chart."""
    fig = go.Figure()
    
    # Price data
    fig.add_trace(go.Scatter(
        x=data['Datetime'],
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4', width=2),
        yaxis='y'
    ))
    
    # Portfolio value
    if portfolio_history:
        dates = [entry['date'] for entry in portfolio_history]
        values = [entry['value'] for entry in portfolio_history]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#ff7f0e', width=3),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="Live Trading",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2=dict(
            title="Portfolio Value ($)",
            overlaying='y',
            side='right'
        ),
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

# Quick action functions
def fetch_sample_data():
    """Quick action to fetch sample data."""
    try:
        data_loader = DataLoader()
        data = data_loader.load_data(
            source="yfinance",
            ticker="BTC-USD",
            start="2022-01-01",
            end="2022-12-31",
            interval="1d",
            cache=True
        )
        st.session_state.data = data
        st.success(f"✅ Fetched {len(data):,} data points")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def quick_train():
    """Quick action to train a model."""
    if st.session_state.data is None:
        st.error("Please fetch data first")
        return
    
    try:
        config = OmegaConf.load("configs/ppo.yaml")
        config.train.total_timesteps = 5000
        config_path = f"configs/quick_ppo_{int(time.time())}.yaml"
        OmegaConf.save(config, config_path)
        
        model_path = train_agent(
            config_path=config_path,
            ticker="BTC-USD",
            start_date=st.session_state.data['Datetime'].min().strftime("%Y-%m-%d"),
            end_date=st.session_state.data['Datetime'].max().strftime("%Y-%m-%d"),
            interval="1d"
        )
        
        st.session_state.trained_models.append({
            'algo': 'ppo',
            'model_path': model_path,
            'config_path': config_path,
            'timestamp': datetime.now(),
            'total_timesteps': 5000
        })
        
        st.success("✅ Quick training completed!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def quick_eval():
    """Quick action to evaluate a model."""
    if not st.session_state.trained_models:
        st.error("No trained models available")
        return
    
    try:
        model = st.session_state.trained_models[-1]  # Latest model
        results = evaluate_agent(
            run_path=model['model_path'],
            ticker="BTC-USD",
            start_date="2023-01-01",
            end_date="2023-12-31",
            interval="1d",
            baseline_strategies=["buy_and_hold"]
        )
        st.success("✅ Quick evaluation completed!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Modern Browser-Focused Dash GUI for Trading Bot Simulator."""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
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

# Initialize Dash app with modern theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Advanced Trading Bot Simulator with RL"},
        {"name": "theme-color", "content": "#1f77b4"}
    ]
)

# Custom CSS for modern web design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Trading Bot Simulator - Advanced RL Trading Environment</title>
        {%favicon%}
        {%css%}
        <style>
            /* Modern CSS Variables */
            :root {
                --primary-color: #1f77b4;
                --secondary-color: #ff7f0e;
                --success-color: #2ca02c;
                --warning-color: #d62728;
                --info-color: #9467bd;
                --light-bg: #f8f9fa;
                --dark-bg: #2c3e50;
                --border-radius: 12px;
                --box-shadow: 0 4px 16px rgba(0,0,0,0.1);
                --transition: all 0.3s ease;
            }
            
            /* Global Styles */
            * {
                font-family: 'Inter', sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
                padding: 0;
            }
            
            /* Header Styling */
            .main-header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: var(--border-radius);
                margin: 1rem;
                padding: 2rem;
                box-shadow: var(--box-shadow);
                text-align: center;
            }
            
            .main-header h1 {
                font-size: 3rem;
                font-weight: 700;
                color: var(--primary-color);
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            .main-header p {
                font-size: 1.2rem;
                color: #666;
                margin: 0.5rem 0 0 0;
            }
            
            /* Card Styling */
            .metric-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 1.5rem;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                border-left: 4px solid var(--primary-color);
                transition: var(--transition);
                margin-bottom: 1rem;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(0,0,0,0.15);
            }
            
            .metric-card h3 {
                font-size: 0.9rem;
                font-weight: 600;
                color: #666;
                margin: 0 0 0.5rem 0;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .metric-card h2 {
                font-size: 2rem;
                font-weight: 700;
                color: var(--primary-color);
                margin: 0;
            }
            
            /* Navigation */
            .nav-tabs {
                border: none;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: var(--border-radius);
                padding: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .nav-tabs .nav-link {
                border: none;
                border-radius: 8px;
                margin: 0 0.25rem;
                padding: 0.75rem 1.5rem;
                font-weight: 500;
                color: #666;
                transition: var(--transition);
            }
            
            .nav-tabs .nav-link.active {
                background: var(--primary-color);
                color: white;
                box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
            }
            
            .nav-tabs .nav-link:hover {
                background: rgba(31, 119, 180, 0.1);
                color: var(--primary-color);
            }
            
            /* Content Area */
            .content-area {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: var(--border-radius);
                padding: 2rem;
                box-shadow: var(--box-shadow);
                margin: 1rem;
            }
            
            /* Form Controls */
            .form-control, .form-select {
                border-radius: 8px;
                border: 2px solid #e9ecef;
                transition: var(--transition);
            }
            
            .form-control:focus, .form-select:focus {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 0.2rem rgba(31, 119, 180, 0.25);
            }
            
            /* Buttons */
            .btn {
                border-radius: 8px;
                font-weight: 600;
                padding: 0.75rem 1.5rem;
                transition: var(--transition);
                border: none;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, var(--primary-color) 0%, #0056b3 100%);
                box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(31, 119, 180, 0.4);
            }
            
            .btn-success {
                background: linear-gradient(135deg, var(--success-color) 0%, #1e7e34 100%);
                box-shadow: 0 4px 12px rgba(44, 160, 44, 0.3);
            }
            
            .btn-danger {
                background: linear-gradient(135deg, var(--warning-color) 0%, #c82333 100%);
                box-shadow: 0 4px 12px rgba(214, 39, 40, 0.3);
            }
            
            /* Status Indicators */
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }
            
            .status-active {
                background: var(--success-color);
                box-shadow: 0 0 8px var(--success-color);
            }
            
            .status-inactive {
                background: var(--warning-color);
                box-shadow: 0 0 8px var(--warning-color);
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            /* Charts */
            .chart-container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: var(--border-radius);
                padding: 1.5rem;
                box-shadow: var(--box-shadow);
                margin-bottom: 1rem;
            }
            
            /* Tables */
            .table {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: var(--border-radius);
                overflow: hidden;
                box-shadow: var(--box-shadow);
            }
            
            .table thead th {
                background: var(--primary-color);
                color: white;
                border: none;
                font-weight: 600;
            }
            
            /* Alerts */
            .alert {
                border-radius: var(--border-radius);
                border: none;
                backdrop-filter: blur(10px);
            }
            
            .alert-success {
                background: rgba(44, 160, 44, 0.1);
                color: var(--success-color);
                border-left: 4px solid var(--success-color);
            }
            
            .alert-danger {
                background: rgba(214, 39, 40, 0.1);
                color: var(--warning-color);
                border-left: 4px solid var(--warning-color);
            }
            
            .alert-warning {
                background: rgba(255, 193, 7, 0.1);
                color: #856404;
                border-left: 4px solid #ffc107;
            }
            
            /* Loading Spinner */
            .loading-spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(31, 119, 180, 0.3);
                border-radius: 50%;
                border-top-color: var(--primary-color);
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .main-header h1 {
                    font-size: 2rem;
                }
                
                .content-area {
                    padding: 1rem;
                    margin: 0.5rem;
                }
                
                .metric-card {
                    padding: 1rem;
                }
                
                .metric-card h2 {
                    font-size: 1.5rem;
                }
            }
            
            /* Dark Mode Support */
            @media (prefers-color-scheme: dark) {
                :root {
                    --light-bg: #2c3e50;
                    --dark-bg: #f8f9fa;
                }
                
                body {
                    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                }
                
                .main-header, .content-area, .metric-card, .chart-container {
                    background: rgba(44, 62, 80, 0.95);
                    color: white;
                }
                
                .form-control, .form-select {
                    background: rgba(44, 62, 80, 0.95);
                    color: white;
                    border-color: #495057;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Global state management
global_state = {
    'data': None,
    'trained_models': [],
    'live_trading': False,
    'portfolio_history': [],
    'current_portfolio': None,
    'current_page': 'dashboard'
}

def create_header():
    """Create modern header component."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("🤖 Trading Bot Simulator", className="mb-2"),
                    html.P("Advanced Reinforcement Learning Trading Environment", className="text-muted mb-0")
                ], className="main-header")
            ])
        ])
    ], fluid=True)

def create_navigation():
    """Create modern navigation tabs."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label=html.Span([html.I(className="fas fa-chart-line me-2"), "Dashboard"]), 
                           tab_id="dashboard", className="nav-link"),
                    dbc.Tab(label=html.Span([html.I(className="fas fa-database me-2"), "Data Management"]), 
                           tab_id="data", className="nav-link"),
                    dbc.Tab(label=html.Span([html.I(className="fas fa-brain me-2"), "Training"]), 
                           tab_id="training", className="nav-link"),
                    dbc.Tab(label=html.Span([html.I(className="fas fa-chart-bar me-2"), "Evaluation"]), 
                           tab_id="evaluation", className="nav-link"),
                    dbc.Tab(label=html.Span([html.I(className="fas fa-play-circle me-2"), "Live Trading"]), 
                           tab_id="live", className="nav-link"),
                    dbc.Tab(label=html.Span([html.I(className="fas fa-cog me-2"), "Settings"]), 
                           tab_id="settings", className="nav-link")
                ], id="tabs", active_tab="dashboard", className="nav-tabs")
            ])
        ])
    ], fluid=True)

def create_metric_card(title, value, icon, color="primary"):
    """Create a modern metric card."""
    return html.Div([
        html.H3([html.I(className=f"fas fa-{icon} me-2"), title]),
        html.H2(value)
    ], className="metric-card")

def create_dashboard_layout():
    """Create modern dashboard layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📊 Dashboard Overview", className="mb-4")
            ])
        ]),
        
        # Metrics Row
        dbc.Row([
            dbc.Col(create_metric_card("Data Points", 
                                     f"{len(global_state['data']) if global_state['data'] is not None else 0:,}", 
                                     "chart-line"), width=3),
            dbc.Col(create_metric_card("Trained Models", 
                                     f"{len(global_state['trained_models'])}", 
                                     "robot"), width=3),
            dbc.Col(create_metric_card("Live Trading", 
                                     "🟢 Active" if global_state['live_trading'] else "🔴 Inactive", 
                                     "play-circle"), width=3),
            dbc.Col(create_metric_card("Portfolio Value", 
                                     f"${global_state['current_portfolio'].get_portfolio_value(0):,.2f}" if global_state['current_portfolio'] else "$0.00", 
                                     "wallet"), width=3)
        ], className="mb-4"),
        
        # Main Content
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("📈 Market Data", className="mb-3"),
                    dcc.Graph(id="price-chart", config={'displayModeBar': False})
                ], className="chart-container")
            ], width=8),
            dbc.Col([
                html.Div([
                    html.H3("📋 Recent Activity", className="mb-3"),
                    html.Div(id="recent-activity")
                ], className="chart-container")
            ], width=4)
        ]),
        
        # Portfolio Performance
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("💰 Portfolio Performance", className="mb-3"),
                    dcc.Graph(id="portfolio-chart", config={'displayModeBar': False})
                ], className="chart-container")
            ])
        ])
    ], fluid=True, className="content-area")

def create_data_management_layout():
    """Create modern data management layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📈 Data Management", className="mb-4")
            ])
        ]),
        
        dbc.Tabs([
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H4("🚀 Fetch Market Data", className="mb-3"),
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Data Source", className="fw-bold"),
                                    dbc.Select(
                                        id="data-source",
                                        options=[
                                            {"label": "Yahoo Finance", "value": "yfinance"},
                                            {"label": "CSV Upload", "value": "csv"}
                                        ],
                                        value="yfinance",
                                        className="mb-3"
                                    ),
                                    html.Div(id="yfinance-options", children=[
                                        dbc.Label("Ticker Symbol", className="fw-bold"),
                                        dbc.Input(
                                            id="ticker-input",
                                            type="text",
                                            value="BTC-USD",
                                            placeholder="e.g., BTC-USD, AAPL, TSLA",
                                            className="mb-3"
                                        ),
                                        dbc.Label("Time Interval", className="fw-bold"),
                                        dbc.Select(
                                            id="interval-select",
                                            options=[
                                                {"label": "Daily", "value": "1d"},
                                                {"label": "Hourly", "value": "1h"},
                                                {"label": "15 Minutes", "value": "15m"},
                                                {"label": "5 Minutes", "value": "5m"},
                                                {"label": "1 Minute", "value": "1m"}
                                            ],
                                            value="1d",
                                            className="mb-3"
                                        )
                                    ]),
                                    html.Div(id="csv-options", style={"display": "none"}, children=[
                                        dcc.Upload(
                                            id="csv-upload",
                                            children=html.Div([
                                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                                                html.Br(),
                                                "Drag and Drop or Click to Upload CSV"
                                            ]),
                                            style={
                                                'width': '100%',
                                                'height': '100px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '8px',
                                                'textAlign': 'center',
                                                'margin': '10px'
                                            }
                                        )
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.H4("📅 Date Range", className="mb-3"),
                                    dbc.Label("Start Date", className="fw-bold"),
                                    dcc.DatePickerSingle(
                                        id="start-date",
                                        date=datetime(2023, 1, 1).date(),
                                        className="mb-3"
                                    ),
                                    dbc.Label("End Date", className="fw-bold"),
                                    dcc.DatePickerSingle(
                                        id="end-date",
                                        date=datetime.now().date(),
                                        className="mb-3"
                                    ),
                                    dbc.Checkbox(
                                        id="cache-data",
                                        label="Cache Data",
                                        value=True,
                                        className="mb-3"
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-download me-2"), "Fetch Data"],
                                        id="fetch-data-btn",
                                        color="primary",
                                        size="lg",
                                        className="w-100"
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ], label="🚀 Fetch Data", tab_id="fetch-tab"),
            
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H4("📊 Data Preview", className="mb-3"),
                        html.Div(id="data-preview")
                    ])
                ])
            ], label="📊 Data Preview", tab_id="preview-tab"),
            
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H4("🗑️ Cache Management", className="mb-3"),
                        dbc.Button(
                            [html.I(className="fas fa-trash me-2"), "Clear Cache"],
                            id="clear-cache-btn",
                            color="danger",
                            className="me-3"
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-info-circle me-2"), "Cache Info"],
                            id="cache-info-btn",
                            color="info"
                        ),
                        html.Div(id="cache-info-display", className="mt-3")
                    ])
                ])
            ], label="🗑️ Cache Management", tab_id="cache-tab")
        ], id="data-tabs")
    ], fluid=True, className="content-area")

def create_training_layout():
    """Create modern training layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("🎯 Model Training", className="mb-4")
            ])
        ]),
        
        dbc.Tabs([
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H4("⚙️ Training Configuration", className="mb-3"),
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("🤖 Algorithm Selection", className="mb-3"),
                                    dbc.Label("Algorithm", className="fw-bold"),
                                    dbc.Select(
                                        id="algo-select",
                                        options=[
                                            {"label": "PPO (Proximal Policy Optimization)", "value": "ppo"},
                                            {"label": "DQN (Deep Q-Network)", "value": "dqn"},
                                            {"label": "A2C (Advantage Actor-Critic)", "value": "a2c"}
                                        ],
                                        value="ppo",
                                        className="mb-3"
                                    ),
                                    dbc.Label("Total Timesteps", className="fw-bold"),
                                    dbc.Input(
                                        id="total-timesteps",
                                        type="number",
                                        value=10000,
                                        min=1000,
                                        step=1000,
                                        className="mb-3"
                                    ),
                                    dbc.Label("Random Seed", className="fw-bold"),
                                    dbc.Input(
                                        id="seed-input",
                                        type="number",
                                        value=42,
                                        className="mb-3"
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.H5("🎯 Environment Settings", className="mb-3"),
                                    dbc.Label("Window Size", className="fw-bold"),
                                    dcc.Slider(
                                        id="window-size-slider",
                                        min=10,
                                        max=128,
                                        value=64,
                                        marks={i: str(i) for i in [10, 32, 64, 96, 128]},
                                        className="mb-3"
                                    ),
                                    dbc.Label("Fee (basis points)", className="fw-bold"),
                                    dcc.Slider(
                                        id="fee-slider",
                                        min=0,
                                        max=100,
                                        value=10,
                                        marks={i: str(i) for i in [0, 25, 50, 75, 100]},
                                        className="mb-3"
                                    ),
                                    dbc.Label("Reward Scheme", className="fw-bold"),
                                    dbc.Select(
                                        id="reward-scheme",
                                        options=[
                                            {"label": "Profit Increment", "value": "profit_increment"},
                                            {"label": "Risk Adjusted", "value": "risk_adjusted"},
                                            {"label": "Trade Penalty", "value": "trade_penalty"}
                                        ],
                                        value="profit_increment",
                                        className="mb-3"
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ], label="⚙️ Configuration", tab_id="config-tab"),
            
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H4("🚀 Start Training", className="mb-3"),
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Start Training"],
                            id="start-training-btn",
                            color="success",
                            size="lg",
                            className="w-100 mb-3"
                        ),
                        html.Div(id="training-progress"),
                        html.Div(id="training-status")
                    ])
                ])
            ], label="🚀 Training", tab_id="train-tab"),
            
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H4("📋 Trained Models", className="mb-3"),
                        html.Div(id="trained-models-list")
                    ])
                ])
            ], label="📋 Models", tab_id="models-tab")
        ], id="training-tabs")
    ], fluid=True, className="content-area")

def create_evaluation_layout():
    """Create modern evaluation layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📋 Model Evaluation", className="mb-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Model Selection", className="mb-3"),
                dbc.Select(
                    id="eval-model-select",
                    options=[],
                    placeholder="Select a model to evaluate",
                    className="mb-3"
                )
            ], width=6),
            dbc.Col([
                html.H4("Evaluation Settings", className="mb-3"),
                dbc.Select(
                    id="baseline-select",
                    options=[
                        {"label": "Buy and Hold", "value": "buy_and_hold"},
                        {"label": "SMA Crossover", "value": "sma_crossover"},
                        {"label": "All Baselines", "value": "all"}
                    ],
                    value="buy_and_hold",
                    className="mb-3"
                )
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-chart-bar me-2"), "Run Evaluation"],
                    id="run-eval-btn",
                    color="primary",
                    size="lg",
                    className="w-100 mb-3"
                )
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="evaluation-results")
            ])
        ])
    ], fluid=True, className="content-area")

def create_live_trading_layout():
    """Create modern live trading layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("💰 Live Trading Simulation", className="mb-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Model Selection", className="mb-3"),
                dbc.Select(
                    id="live-model-select",
                    options=[],
                    placeholder="Select a model for live trading",
                    className="mb-3"
                )
            ], width=6),
            dbc.Col([
                html.H4("Controls", className="mb-3"),
                dbc.Button(
                    [html.I(className="fas fa-play me-2"), "Start Live Trading"],
                    id="start-live-btn",
                    color="success",
                    className="me-2"
                ),
                dbc.Button(
                    [html.I(className="fas fa-stop me-2"), "Stop Live Trading"],
                    id="stop-live-btn",
                    color="danger",
                    className="me-2"
                ),
                dbc.Button(
                    [html.I(className="fas fa-redo me-2"), "Reset Portfolio"],
                    id="reset-portfolio-btn",
                    color="warning"
                )
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="live-trading-status")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("📈 Live Trading Chart", className="mb-3"),
                    dcc.Graph(id="live-trading-chart", config={'displayModeBar': False})
                ], className="chart-container")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("📋 Trading Log", className="mb-3"),
                    html.Div(id="trading-log")
                ], className="chart-container")
            ])
        ])
    ], fluid=True, className="content-area")

def create_settings_layout():
    """Create modern settings layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("⚙️ Settings", className="mb-4")
            ])
        ]),
        
        dbc.Tabs([
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H4("🔧 General Settings", className="mb-3"),
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Default Ticker", className="fw-bold"),
                                    dbc.Input(
                                        id="default-ticker",
                                        type="text",
                                        value="BTC-USD",
                                        className="mb-3"
                                    ),
                                    dbc.Label("Default Interval", className="fw-bold"),
                                    dbc.Select(
                                        id="default-interval",
                                        options=[
                                            {"label": "Daily", "value": "1d"},
                                            {"label": "Hourly", "value": "1h"},
                                            {"label": "15 Minutes", "value": "15m"},
                                            {"label": "5 Minutes", "value": "5m"},
                                            {"label": "1 Minute", "value": "1m"}
                                        ],
                                        value="1d",
                                        className="mb-3"
                                    ),
                                    dbc.Checkbox(
                                        id="cache-enabled",
                                        label="Enable Caching",
                                        value=True,
                                        className="mb-3"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Default Starting Cash", className="fw-bold"),
                                    dbc.Input(
                                        id="default-start-cash",
                                        type="number",
                                        value=10000,
                                        min=1000,
                                        max=100000,
                                        className="mb-3"
                                    ),
                                    dbc.Label("Default Window Size", className="fw-bold"),
                                    dcc.Slider(
                                        id="default-window-size",
                                        min=10,
                                        max=128,
                                        value=64,
                                        marks={i: str(i) for i in [10, 32, 64, 96, 128]},
                                        className="mb-3"
                                    ),
                                    dbc.Checkbox(
                                        id="auto-save",
                                        label="Auto-save Results",
                                        value=True,
                                        className="mb-3"
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ], label="🔧 General", tab_id="general-tab"),
            
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H4("📊 Advanced Settings", className="mb-3"),
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Log Level", className="fw-bold"),
                                    dbc.Select(
                                        id="log-level",
                                        options=[
                                            {"label": "INFO", "value": "INFO"},
                                            {"label": "DEBUG", "value": "DEBUG"},
                                            {"label": "WARNING", "value": "WARNING"},
                                            {"label": "ERROR", "value": "ERROR"}
                                        ],
                                        value="INFO",
                                        className="mb-3"
                                    ),
                                    dbc.Label("Max Cache Size (MB)", className="fw-bold"),
                                    dbc.Input(
                                        id="max-cache-size",
                                        type="number",
                                        value=1000,
                                        min=100,
                                        max=10000,
                                        className="mb-3"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Update Frequency", className="fw-bold"),
                                    dbc.Select(
                                        id="update-frequency",
                                        options=[
                                            {"label": "1 Second", "value": "1s"},
                                            {"label": "5 Seconds", "value": "5s"},
                                            {"label": "10 Seconds", "value": "10s"},
                                            {"label": "30 Seconds", "value": "30s"},
                                            {"label": "1 Minute", "value": "1m"}
                                        ],
                                        value="5s",
                                        className="mb-3"
                                    ),
                                    dbc.Label("Chart Theme", className="fw-bold"),
                                    dbc.Select(
                                        id="chart-theme",
                                        options=[
                                            {"label": "Default", "value": "plotly"},
                                            {"label": "White", "value": "plotly_white"},
                                            {"label": "Dark", "value": "plotly_dark"}
                                        ],
                                        value="plotly",
                                        className="mb-3"
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ], label="📊 Advanced", tab_id="advanced-tab")
        ], id="settings-tabs"),
        
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-save me-2"), "Save Settings"],
                    id="save-settings-btn",
                    color="primary",
                    size="lg",
                    className="w-100"
                )
            ])
        ])
    ], fluid=True, className="content-area")

# Main app layout
app.layout = dbc.Container([
    # Header
    create_header(),
    
    # Navigation
    create_navigation(),
    
    # Content Area
    html.Div(id="page-content"),
    
    # Stores for data persistence
    dcc.Store(id="data-store"),
    dcc.Store(id="models-store"),
    dcc.Store(id="settings-store"),
    
    # Interval for live updates
    dcc.Interval(
        id="interval-component",
        interval=5000,  # 5 seconds
        n_intervals=0
    )
], fluid=True)

# Callbacks for page content
@app.callback(
    Output("page-content", "children"),
    [Input("tabs", "active_tab")]
)
def update_page_content(active_tab):
    """Update page content based on active tab."""
    if active_tab == "dashboard":
        return create_dashboard_layout()
    elif active_tab == "data":
        return create_data_management_layout()
    elif active_tab == "training":
        return create_training_layout()
    elif active_tab == "evaluation":
        return create_evaluation_layout()
    elif active_tab == "live":
        return create_live_trading_layout()
    elif active_tab == "settings":
        return create_settings_layout()
    else:
        return create_dashboard_layout()

# Data source callback
@app.callback(
    [Output("yfinance-options", "style"),
     Output("csv-options", "style")],
    [Input("data-source", "value")]
)
def toggle_data_source_options(source):
    """Toggle between yfinance and CSV options."""
    if source == "yfinance":
        return {"display": "block"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block"}

# Fetch data callback
@app.callback(
    [Output("data-preview", "children"),
     Output("data-store", "data")],
    [Input("fetch-data-btn", "n_clicks")],
    [State("data-source", "value"),
     State("ticker-input", "value"),
     State("interval-select", "value"),
     State("start-date", "date"),
     State("end-date", "date"),
     State("cache-data", "value"),
     State("csv-upload", "contents")]
)
def fetch_data(n_clicks, source, ticker, interval, start_date, end_date, cache_data, csv_contents):
    """Fetch market data."""
    if n_clicks is None:
        return "", None
    
    try:
        data_loader = DataLoader()
        
        if source == "yfinance":
            data = data_loader.load_data(
                source="yfinance",
                ticker=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                cache=cache_data
            )
        else:
            # Handle CSV upload
            if csv_contents is not None:
                import base64
                import io
                content_type, content_string = csv_contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                data = df
            else:
                return dbc.Alert("Please upload a CSV file", color="warning"), None
        
        global_state['data'] = data
        
        # Create data preview
        preview = dbc.Row([
            dbc.Col([
                html.H5("Data Statistics", className="mb-3"),
                dbc.Table.from_dataframe(
                    data.describe().round(2),
                    striped=True,
                    bordered=True,
                    hover=True
                )
            ], width=6),
            dbc.Col([
                html.H5("Data Preview", className="mb-3"),
                dbc.Table.from_dataframe(
                    data.head(10),
                    striped=True,
                    bordered=True,
                    hover=True
                )
            ], width=6)
        ])
        
        return preview, data.to_dict('records')
        
    except Exception as e:
        return dbc.Alert(f"Error fetching data: {str(e)}", color="danger"), None

# Training callback
@app.callback(
    [Output("training-status", "children"),
     Output("models-store", "data")],
    [Input("start-training-btn", "n_clicks")],
    [State("algo-select", "value"),
     State("total-timesteps", "value"),
     State("seed-input", "value"),
     State("window-size-slider", "value"),
     State("fee-slider", "value"),
     State("reward-scheme", "value")]
)
def start_training(n_clicks, algo, total_timesteps, seed, window_size, fee_bps, reward_scheme):
    """Start model training."""
    if n_clicks is None:
        return "", []
    
    if global_state['data'] is None:
        return dbc.Alert("Please fetch data first", color="warning"), []
    
    try:
        # Create config with proper merging
        base_config = OmegaConf.load("configs/default.yaml")
        algo_config = OmegaConf.load(f"configs/{algo}.yaml")
        config = OmegaConf.merge(base_config, algo_config)
        
        # Update config values
        config.train.total_timesteps = total_timesteps
        config.train.seed = seed
        config.env.window_size = window_size
        config.env.fee_bps = fee_bps
        config.env.reward = reward_scheme
        
        # Save config
        config_path = f"configs/custom_{algo}_{int(time.time())}.yaml"
        OmegaConf.save(config, config_path)
        
        # Train agent
        model_path = train_agent(
            config_path=config_path,
            ticker="BTC-USD",
            start_date=global_state['data']['Datetime'].min().strftime("%Y-%m-%d"),
            end_date=global_state['data']['Datetime'].max().strftime("%Y-%m-%d"),
            interval="1d"
        )
        
        # Add to trained models
        model_info = {
            'algo': algo,
            'model_path': model_path,
            'config_path': config_path,
            'timestamp': datetime.now().isoformat(),
            'total_timesteps': total_timesteps
        }
        global_state['trained_models'].append(model_info)
        
        return dbc.Alert("Training completed successfully!", color="success"), global_state['trained_models']
        
    except Exception as e:
        return dbc.Alert(f"Training failed: {str(e)}", color="danger"), []

# Chart callbacks
@app.callback(
    Output("price-chart", "figure"),
    [Input("data-store", "data")]
)
def update_price_chart(data):
    """Update price chart."""
    if data is None:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
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

@app.callback(
    Output("portfolio-chart", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_portfolio_chart(n):
    """Update portfolio chart."""
    if not global_state['portfolio_history']:
        return go.Figure()
    
    fig = go.Figure()
    dates = [entry['date'] for entry in global_state['portfolio_history']]
    values = [entry['value'] for entry in global_state['portfolio_history']]
    
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

# Settings callback
@app.callback(
    Output("settings-store", "data"),
    [Input("save-settings-btn", "n_clicks")],
    [State("default-ticker", "value"),
     State("default-interval", "value"),
     State("cache-enabled", "value"),
     State("default-start-cash", "value"),
     State("default-window-size", "value"),
     State("auto-save", "value"),
     State("log-level", "value"),
     State("max-cache-size", "value"),
     State("update-frequency", "value"),
     State("chart-theme", "value")]
)
def save_settings(n_clicks, default_ticker, default_interval, cache_enabled, 
                 default_start_cash, default_window_size, auto_save, log_level,
                 max_cache_size, update_frequency, chart_theme):
    """Save settings."""
    if n_clicks is None:
        return {}
    
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
    
    return settings

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)

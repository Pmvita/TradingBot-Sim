"""Plotting functionality for trading results."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger(__name__)


def create_evaluation_plots(
    results: Dict[str, Any], data: pd.DataFrame, output_dir: Path
) -> None:
    """Create comprehensive evaluation plots.
    
    Args:
        results: Evaluation results
        data: OHLCV data
        output_dir: Directory to save plots
    """
    # Create equity curve plot
    create_equity_curve_plot(results, data, output_dir)
    
    # Create drawdown plot
    create_drawdown_plot(results, output_dir)
    
    # Create actions over price plot
    create_actions_plot(results, data, output_dir)
    
    # Create action frequency plot
    create_action_frequency_plot(results, output_dir)
    
    # Create rolling Sharpe ratio plot
    create_rolling_sharpe_plot(results, output_dir)
    
    logger.info("Created evaluation plots", output_dir=str(output_dir))


def create_equity_curve_plot(
    results: Dict[str, Any], data: pd.DataFrame, output_dir: Path
) -> None:
    """Create equity curve comparison plot.
    
    Args:
        results: Evaluation results
        data: OHLCV data
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot price data
    ax2 = ax.twinx()
    ax2.plot(data["Datetime"], data["Close"], alpha=0.3, color="gray", label="Price")
    ax2.set_ylabel("Price", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    
    # Plot portfolio values
    colors = ["blue", "red", "green", "orange", "purple"]
    color_idx = 0
    
    for strategy, strategy_data in results.items():
        if strategy == "metadata":
            continue
        
        portfolio_values = strategy_data["portfolio_values"]
        if len(portfolio_values) > 0:
            # Create time index
            if len(portfolio_values) == len(data):
                time_index = data["Datetime"]
            else:
                # Pad with initial value if needed
                initial_value = portfolio_values[0]
                padded_values = [initial_value] + portfolio_values
                time_index = data["Datetime"].iloc[:len(padded_values)]
                portfolio_values = padded_values
            
            ax.plot(
                time_index,
                portfolio_values,
                label=strategy,
                color=colors[color_idx % len(colors)],
                linewidth=2,
            )
            color_idx += 1
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Equity Curve Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_drawdown_plot(results: Dict[str, Any], output_dir: Path) -> None:
    """Create drawdown comparison plot.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ["blue", "red", "green", "orange", "purple"]
    color_idx = 0
    
    for strategy, strategy_data in results.items():
        if strategy == "metadata":
            continue
        
        portfolio_values = strategy_data["portfolio_values"]
        if len(portfolio_values) > 0:
            # Calculate drawdown
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (running_max - portfolio_values) / running_max
            
            ax.plot(
                drawdowns,
                label=strategy,
                color=colors[color_idx % len(colors)],
                linewidth=2,
            )
            color_idx += 1
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_actions_plot(
    results: Dict[str, Any], data: pd.DataFrame, output_dir: Path
) -> None:
    """Create actions over price plot.
    
    Args:
        results: Evaluation results
        data: OHLCV data
        output_dir: Output directory
    """
    # Find agent results
    agent_data = None
    for strategy, strategy_data in results.items():
        if strategy == "agent":
            agent_data = strategy_data
            break
    
    if not agent_data or not agent_data["actions"]:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price data
    ax1.plot(data["Datetime"], data["Close"], label="Price", color="black", linewidth=1)
    ax1.set_ylabel("Price")
    ax1.set_title("Price and Actions")
    ax1.grid(True, alpha=0.3)
    
    # Plot actions
    actions = agent_data["actions"]
    action_colors = {0: "gray", 1: "green", 2: "red"}
    action_labels = {0: "Hold", 1: "Buy", 2: "Sell"}
    
    for i, action in enumerate(actions):
        if i < len(data):
            ax1.scatter(
                data["Datetime"].iloc[i],
                data["Close"].iloc[i],
                color=action_colors[action],
                s=50,
                alpha=0.7,
                marker="o" if action == 0 else "^" if action == 1 else "v",
            )
    
    # Create legend for actions
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Hold"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="green", markersize=10, label="Buy"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="red", markersize=10, label="Sell"),
    ]
    ax1.legend(handles=legend_elements)
    
    # Plot action frequency
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in actions:
        action_counts[action] += 1
    
    actions_list = list(action_counts.keys())
    counts = list(action_counts.values())
    colors = [action_colors[a] for a in actions_list]
    labels = [action_labels[a] for a in actions_list]
    
    ax2.bar(actions_list, counts, color=colors, alpha=0.7)
    ax2.set_xlabel("Action")
    ax2.set_ylabel("Count")
    ax2.set_title("Action Frequency")
    ax2.set_xticks(actions_list)
    ax2.set_xticklabels(labels)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "actions_over_price.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_action_frequency_plot(results: Dict[str, Any], output_dir: Path) -> None:
    """Create action frequency comparison plot.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = []
    hold_counts = []
    buy_counts = []
    sell_counts = []
    
    for strategy, strategy_data in results.items():
        if strategy == "metadata":
            continue
        
        actions = strategy_data.get("actions", [])
        if actions:
            strategies.append(strategy)
            hold_counts.append(actions.count(0))
            buy_counts.append(actions.count(1))
            sell_counts.append(actions.count(2))
    
    if not strategies:
        return
    
    x = np.arange(len(strategies))
    width = 0.25
    
    ax.bar(x - width, hold_counts, width, label="Hold", color="gray", alpha=0.7)
    ax.bar(x, buy_counts, width, label="Buy", color="green", alpha=0.7)
    ax.bar(x + width, sell_counts, width, label="Sell", color="red", alpha=0.7)
    
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Action Count")
    ax.set_title("Action Frequency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "action_frequency.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_rolling_sharpe_plot(results: Dict[str, Any], output_dir: Path) -> None:
    """Create rolling Sharpe ratio plot.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ["blue", "red", "green", "orange", "purple"]
    color_idx = 0
    
    for strategy, strategy_data in results.items():
        if strategy == "metadata":
            continue
        
        portfolio_values = strategy_data["portfolio_values"]
        if len(portfolio_values) > 20:  # Need enough data for rolling calculation
            # Calculate returns
            returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(ret)
            
            if len(returns) > 20:
                # Calculate rolling Sharpe ratio (20-day window)
                window = 20
                rolling_sharpe = []
                
                for i in range(window, len(returns)):
                    window_returns = returns[i-window:i]
                    if np.std(window_returns) > 0:
                        sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                        rolling_sharpe.append(sharpe)
                    else:
                        rolling_sharpe.append(0)
                
                if rolling_sharpe:
                    ax.plot(
                        rolling_sharpe,
                        label=strategy,
                        color=colors[color_idx % len(colors)],
                        linewidth=2,
                    )
                    color_idx += 1
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling Sharpe Ratio (20-day)")
    ax.set_title("Rolling Sharpe Ratio Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_sharpe.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_metrics_summary_plot(results: Dict[str, Any], output_dir: Path) -> None:
    """Create metrics summary comparison plot.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    # Extract metrics
    strategies = []
    total_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for strategy, strategy_data in results.items():
        if strategy == "metadata":
            continue
        
        metrics = strategy_data.get("metrics", {})
        if metrics:
            strategies.append(strategy)
            total_returns.append(metrics.get("total_return", 0))
            sharpe_ratios.append(metrics.get("sharpe_ratio", 0))
            max_drawdowns.append(metrics.get("max_drawdown", 0))
    
    if not strategies:
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Return
    ax1.bar(strategies, total_returns, color="skyblue", alpha=0.7)
    ax1.set_title("Total Return")
    ax1.set_ylabel("Return (%)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Sharpe Ratio
    ax2.bar(strategies, sharpe_ratios, color="lightgreen", alpha=0.7)
    ax2.set_title("Sharpe Ratio")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Max Drawdown
    ax3.bar(strategies, max_drawdowns, color="lightcoral", alpha=0.7)
    ax3.set_title("Maximum Drawdown")
    ax3.set_ylabel("Drawdown (%)")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Win Rate (if available)
    win_rates = []
    for strategy, strategy_data in results.items():
        if strategy != "metadata":
            metrics = strategy_data.get("metrics", {})
            win_rates.append(metrics.get("win_rate", 0))
    
    if win_rates:
        ax4.bar(strategies, win_rates, color="gold", alpha=0.7)
        ax4.set_title("Win Rate")
        ax4.set_ylabel("Win Rate (%)")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

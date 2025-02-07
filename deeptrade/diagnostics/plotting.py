from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig


@dataclass
class PlotStyle:
    """Centralized style configuration"""
    colors: Dict[str, str]
    fonts: Dict[str, Dict[str, any]]
    figure_sizes: Dict[str, Tuple[int, int]]

    @classmethod
    def from_config(cls, config: DictConfig):
        """Load style configuration from hydra config"""
        return cls(
            colors=dict(config.colors),
            fonts=dict(config.fonts),
            figure_sizes={k: tuple(v) for k, v in config.figure_sizes.items()}
        )

    def apply_style(self, ax: plt.Axes, plot_type: str):
        """Apply consistent styling to axis"""
        # Apply common styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Apply font styling
        ax.set_title(ax.get_title(), **self.fonts['title'])
        ax.set_xlabel(ax.get_xlabel(), **self.fonts['label'])
        ax.set_ylabel(ax.get_ylabel(), **self.fonts['label'])

        # Apply specific styling based on plot type
        if plot_type in self.fonts:
            ax.tick_params(**self.fonts[plot_type])


class Plotter:
    """OmegaConfiguration-driven plotting system"""

    def __init__(self, config: DictConfig, save_dir: Path):
        """
        Initialize plotter with Hydra config

        Args:
            config: Hydra config containing plot_style section
            save_dir: Directory to save plots
        """
        self.style = PlotStyle.from_config(config.plotting)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _create_figure(self, plot_type: str) -> Tuple[plt.Figure, plt.Axes]:
        """Create figure with standardized size"""
        fig_size = self.style.figure_sizes.get(plot_type, (10, 6))
        return plt.subplots(figsize=fig_size)

    def plot_training_metrics(self,
                            train_history: Dict[str, List[float]],
                            val_history: Dict[str, List[float]],
                            save_name: str = "training_metrics.png"):
        """Plot training metrics with standardized styling"""
        fig, (ax1, ax2) = plt.subplots(2, 1,
                                      figsize=self.style.figure_sizes['training'])

        # Loss plot
        ax1.plot(train_history['loss'],
                color=self.style.colors['train'],
                label='Training Loss',
                linewidth=2,
                alpha=0.8)
        ax1.plot(val_history['loss'],
                color=self.style.colors['validation'],
                label='Validation Loss',
                linewidth=2,
                alpha=0.8)
        ax1.set_title('Model Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        self.style.apply_style(ax1, 'training')
        ax1.legend()

        # MSE plot
        ax2.plot(train_history['MSE'],
                color=self.style.colors['train'],
                label='Training MSE',
                linewidth=2,
                alpha=0.8)
        ax2.plot(val_history['MSE'],
                color=self.style.colors['validation'],
                label='Validation MSE',
                linewidth=2,
                alpha=0.8)
        ax2.set_title('Model MSE Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        self.style.apply_style(ax2, 'training')
        ax2.legend()

        plt.tight_layout()
        fig.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_predictions(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        save_name: str = "predictions.png"):
        """Plot predictions with standardized styling"""
        fig, (ax1, ax2) = plt.subplots(1, 2,
                                      figsize=self.style.figure_sizes['prediction'])

        # Scatter plot
        ax1.scatter(y_true, y_pred,
                   color=self.style.colors['scatter'],
                   alpha=0.5)

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val],
                 color=self.style.colors['validation'],
                 linestyle='--',
                 label='Perfect Prediction')

        ax1.set_title('Predicted vs Actual Values')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        self.style.apply_style(ax1, 'prediction')
        ax1.legend()

        # Residuals plot
        residuals = y_true - y_pred
        ax2.hist(residuals.flatten(),
                bins=30,
                color=self.style.colors['train'],
                alpha=0.7,
                density=True)

        ax2.set_title('Distribution of Residuals')
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Density')
        self.style.apply_style(ax2, 'prediction')

        plt.tight_layout()
        fig.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_forecast(self,
                        actual_values: np.ndarray,
                        forecast_values: np.ndarray,
                        forecast_frequency: int = 100,
                        forecast_variance: Optional[np.ndarray] = None,
                        save_name: str = "forecast_comparison.png"):
        """
        Plot forecast values against actual values with confidence intervals

        Args:
            actual_values: Array of actual values (n_values)
            forecast_values: Array of forecasted values [n_values, forecast]
            forecast_frequency: How often to plot forecast values
            forecast_variance: Optional array of forecast variances for confidence intervals
            save_name: Name of the output file
        """
        fig, ax = plt.subplots(figsize=self.style.figure_sizes.get('forecast', (16, 8)))

        # Create time index
        time_idx = np.arange(len(actual_values))

        # Plot actual values
        ax.plot(time_idx, actual_values,
                color=self.style.colors.get('actual', '#2ecc71'),
                label='Actual',
                linewidth=2,
                alpha=0.8)

        # Plot forecast values
        forecast_plotted = False
        for i in range(0, len(forecast_values), forecast_frequency):

            forecast = forecast_values[i]
            valid_indices = np.nonzero(forecast)[0]
            if len(valid_indices) > 0:
                label = 'Forecast' if not forecast_plotted else None
                ax.plot(time_idx[valid_indices], forecast[valid_indices],
                        color=self.style.colors.get('forecast', '#e74c3c'),
                        label=label,
                        linewidth=2,
                        alpha=0.8)
                forecast_plotted = True

        # Add confidence intervals if variance is provided
        if forecast_variance is not None:
            confidence_interval = 1.96 * np.sqrt(forecast_variance)  # 95% confidence interval
            ax.fill_between(time_idx,
                            forecast_values - confidence_interval,
                            forecast_values + confidence_interval,
                            color=self.style.colors.get('confidence', '#e74c3c'),
                            alpha=0.2,
                            label='95% Confidence Interval')

        # Add chart title and labels
        ax.set_title('Forecast vs Actual Values')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')

        # Apply standard styling
        self.style.apply_style(ax, 'forecast')

        # Add legend
        ax.legend(loc='best')

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save plot
        plt.tight_layout()
        fig.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_forecast_analysis(self,
                                 actual_values: np.ndarray,
                                 forecast_values: np.ndarray,
                                 forecast_variance: Optional[np.ndarray] = None,
                                 split_idx: Optional[int] = None,
                                 save_name: str = "forecast_analysis.png"):
        """
        Create a comprehensive forecast analysis plot with multiple subplots

        Args:
            actual_values: Array of actual values
            forecast_values: Array of forecasted values
            forecast_variance: Optional array of forecast variances
            split_idx: Optional index where forecast begins
            save_name: Name of the output file
        """
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 2, figure=fig)

        # Main forecast plot
        ax1 = fig.add_subplot(gs[0, :])
        time_idx = np.arange(len(actual_values))

        ax1.plot(time_idx, actual_values,
                color=self.style.colors.get('actual', '#2ecc71'),
                label='Actual',
                linewidth=2)
        ax1.plot(time_idx, forecast_values,
                color=self.style.colors.get('forecast', '#e74c3c'),
                label='Forecast',
                linewidth=2)

        if forecast_variance is not None:
            confidence_interval = 1.96 * np.sqrt(forecast_variance)
            ax1.fill_between(time_idx,
                            forecast_values - confidence_interval,
                            forecast_values + confidence_interval,
                            color=self.style.colors.get('confidence', '#e74c3c'),
                            alpha=0.2,
                            label='95% Confidence Interval')

        if split_idx is not None:
            ax1.axvline(x=split_idx,
                        color=self.style.colors.get('split', '#7f8c8d'),
                        linestyle='--',
                        label='Forecast Start')

        self.style.apply_style(ax1, 'forecast')
        ax1.set_title('Forecast vs Actual Values')
        ax1.legend()

        # Error plot
        ax2 = fig.add_subplot(gs[1, 0])
        forecast_error = actual_values - forecast_values
        ax2.plot(time_idx, forecast_error,
                color=self.style.colors.get('error', '#3498db'))
        if split_idx is not None:
            ax2.axvline(x=split_idx,
                        color=self.style.colors.get('split', '#7f8c8d'),
                        linestyle='--')
        self.style.apply_style(ax2, 'forecast')
        ax2.set_title('Forecast Error')

        # Error distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(forecast_error,
                bins=30,
                # color=self.style.colors.get('histogram', '#3498db'),
                alpha=0.7,
                density=True)
        self.style.apply_style(ax3, 'forecast')
        ax3.set_title('Error Distribution')

        plt.tight_layout()
        fig.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_overview(self,
                            price_data: np.ndarray,
                            positions: np.ndarray,
                            returns: np.ndarray,
                            benchmark_returns: Optional[np.ndarray] = None,
                            save_name: str = "performance_overview.png"):
        """
        Creates two focused plots: Price/Position Overview and Returns Distribution
        """
        fig = plt.figure(figsize=self.style.figure_sizes.get('overview', (15, 10)))
        gs = plt.GridSpec(2, 2, height_ratios=[2, 1], figure=fig)

        # 1. Price and Position Overview (Top Plot)
        ax_main = fig.add_subplot(gs[0, :])
        # Plot price
        ax_main.plot(price_data, color='gray', alpha=0.7, label='Price')
        ax_main.set_ylabel('Price [USD]')

        # Plot position size as blue shaded area
        ax_twin = ax_main.twinx()
        ax_twin.fill_between(range(len(positions)),
                            positions,
                            color=self.style.colors.get('position', '#3498db'),
                            alpha=0.2,
                            label='Position Size')
        ax_twin.set_ylabel('Position Size [-]')

        # Add trade markers (green up arrows for buys, red down arrows for sells)
        position_changes = np.diff(positions, prepend=0)
        trade_points = np.where(position_changes != 0)[0]
        for idx, point in enumerate(trade_points):
            direction = position_changes[point]
            if point in range(len(returns)):
                returns_at_trade = returns[point]
            color = '#2ecc71' if direction > 0 else '#e74c3c'  # Green for buys, red for sells
            marker = '^' if direction > 0 else 'v'  # Up arrow for buys, down for sells
            size = abs(returns_at_trade) * 5000  # Size based on profitability
            ax_main.scatter(point, price_data[point],
                        color=color,
                        marker=marker,
                        s=size,
                        alpha=0.7)

        ax_main.set_xlabel('Trading Day')
        ax_main.set_title('Trading Activity Overview')
        self.style.apply_style(ax_main, 'overview')

        # 2. Returns Distribution (Left Bottom)
        ax_ret = fig.add_subplot(gs[1, 0])
        ax_ret.set_xlabel('Return')
        ax_ret.set_ylabel('Frequency')
        # Plot returns histogram
        ax_ret.hist(returns, bins=50, density=True, alpha=0.7,
                    color=self.style.colors.get('returns', '#3498db'))

        # Add normal distribution fit
        mu, std = np.mean(returns), np.std(returns)
        x = np.linspace(min(returns), max(returns), 100)
        norm_dist = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * std**2))
        ax_ret.plot(x, norm_dist, 'r--', label='Normal Dist.')

        # Add key statistics
        stats = (f'Sharpe: {self._calculate_sharpe(returns):.2f}\n'
                f'Sortino: {self._calculate_sortino(returns):.2f}\n'
                f'Skew: {self._calculate_skew(returns):.2f}')
        ax_ret.text(0.02, 0.98, stats,
                    transform=ax_ret.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_ret.set_title('Returns Distribution')
        self.style.apply_style(ax_ret, 'overview')

        # 3. Drawdown Analysis (Right Bottom)
        ax_dd = fig.add_subplot(gs[1, 1])
        cumulative_returns = np.exp(np.cumsum(returns)) - 1
        drawdowns = self._calculate_drawdown_series(cumulative_returns)
        ax_dd.fill_between(range(len(drawdowns)),
                        drawdowns,
                        0,
                        color=self.style.colors.get('drawdown', '#e74c3c'),
                        alpha=0.3)
        ax_dd.set_title('Drawdown Analysis')
        self.style.apply_style(ax_dd, 'overview')
        ax_dd.set_xlabel('Trading Day')
        ax_dd.set_ylabel('Drawdown [%]')

        plt.tight_layout()
        fig.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_trade_analysis(self,
                        positions: np.ndarray,
                        returns: np.ndarray,
                        price_data: np.ndarray,
                        save_name: str = "trade_analysis.png"):
        """
        Creates focused analysis of position holding and returns by size
        """
        fig = plt.figure(figsize=self.style.figure_sizes.get('trades', (15, 10)))
        gs = plt.GridSpec(2, 1, figure=fig)

        # 1. Position Holding Analysis
        ax_hold = fig.add_subplot(gs[0])
        durations = self._calculate_trade_durations(positions)

        # Create histogram of holding periods
        ax_hold.hist(durations, bins=30,
                    color=self.style.colors.get('duration', '#3498db'),
                    alpha=0.7)

        # Add mean and median lines
        ax_hold.axvline(np.mean(durations), color='r', linestyle='--',
                        label=f'Mean: {np.mean(durations):.1f} days')
        ax_hold.axvline(np.median(durations), color='g', linestyle='--',
                        label=f'Median: {np.median(durations):.1f} days')

        ax_hold.set_xlabel('Holding Period [Days]')
        ax_hold.set_ylabel('Number of Trades')
        ax_hold.set_title('Trade Duration Analysis')
        ax_hold.legend()
        self.style.apply_style(ax_hold, 'trades')

        # 2. Return by Position Size
        ax_size = fig.add_subplot(gs[1])
        ax_size.set_xlabel('Position Size')
        ax_size.set_ylabel('Trade Return [%]')
        position_changes = np.diff(positions, prepend=0)
        trade_points = np.where(position_changes != 0)[0]

        # Create scatter plot of position size vs returns
        sizes = np.abs(positions[trade_points])[:len(returns)]
        trade_returns = returns[trade_points[:len(returns)]]

        ax_size.scatter(sizes, trade_returns,
                    color=self.style.colors.get('scatter', '#3498db'),
                    alpha=0.5)

        # Add trend line
        z = np.polyfit(sizes, trade_returns, 1)
        p = np.poly1d(z)
        ax_size.plot(sizes, p(sizes), "r--", alpha=0.8,
                    label=f'Trend (slope: {z[0]:.4f})')

        ax_size.set_title('Returns vs Position Size')
        ax_size.set_xlabel('Position Size')
        ax_size.set_ylabel('Return')
        ax_size.legend()
        self.style.apply_style(ax_size, 'trades')

        plt.tight_layout()
        fig.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_risk_metrics(self,
                        returns: np.ndarray,
                        benchmark_returns: Optional[np.ndarray] = None,
                        window: int = 252,
                        save_name: str = "risk_metrics.png"):
        """
        Creates rolling performance metrics visualization
        """
        fig, ax = plt.subplots(figsize=self.style.figure_sizes.get('risk', (15, 10)))

        # Calculate rolling metrics
        rolling_returns = pd.Series(returns).rolling(window).mean() * 252  # Annualized
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol

        # Plot metrics
        ax.plot(rolling_returns, label='Returns [ann.]',
                color=self.style.colors.get('returns', '#2ecc71'))
        ax.plot(rolling_vol, label='Volatility [ann.]',
                color=self.style.colors.get('volatility', '#e74c3c'))
        ax2 = ax.twinx()
        ax2.plot(rolling_sharpe, label='Sharpe Ratio',
                    color=self.style.colors.get('sharpe', '#3498db'))
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Value')
        ax2.set_ylabel('Sharpe Ratio')

        # Add benchmark comparison if provided
        if benchmark_returns is not None:
            rolling_benchmark = pd.Series(benchmark_returns).rolling(window).mean() * 252
            ax.plot(rolling_benchmark, label='Benchmark Returns',
                    color=self.style.colors.get('benchmark', '#95a5a6'),
                    linestyle='--')

        ax.set_title(f'Rolling Performance Metrics ({window}-day window)')
        ax.legend(title='Risk Values', loc = 'upper left')
        ax2.legend(loc='upper right')
        self.style.apply_style(ax, 'risk')

        plt.tight_layout()
        fig.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_portfolio_optimization(self,
                                    weights: np.ndarray,
                                    returns: np.ndarray,
                                    asset_names: Optional[List[str]] = None,
                                    save_name: str = "portfolio_optimization.png"):
        """
        Create comprehensive portfolio optimization visualization

        Args:
            weights: Array of portfolio weights for each asset
            returns: Array of asset returns [n_assets, n_timesteps]
            asset_names: Optional list of asset names
            save_name: Name of output file
        """
        if asset_names is None:
            asset_names = [f"Asset {ida}" for ida in range(len(weights))]

        fig = plt.figure(figsize=self.style.figure_sizes.get('portfolio', (15, 10)))
        gs = plt.GridSpec(2, 2, figure=fig)

        # 1. Portfolio Allocation (Pie Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        wedges, texts, autotexts = ax1.pie(weights,
                                        labels=asset_names,
                                        autopct='%1.1f%%',
                                        colors=plt.cm.Set3(np.linspace(0, 1, len(weights))))
        ax1.set_title('Portfolio Weight Allocation')

        # 2. Risk-Return Scatter
        ax2 = fig.add_subplot(gs[0, 1])
        returns_mean = np.mean(returns, axis=1) * 252  # Annualized
        returns_vol = np.std(returns, axis=1) * np.sqrt(252)

        # Plot individual assets
        ax2.scatter(returns_vol, returns_mean,
                    c=plt.cm.Set3(np.linspace(0, 1, len(weights))),
                    s=100,
                    alpha=0.6)

        # Plot portfolio point
        port_return = np.sum(weights * returns_mean)
        port_vol = np.sqrt(np.sum(weights**2 * returns_vol**2))  # Simplified, ignores correlations
        ax2.scatter(port_vol, port_return,
                    c='red',
                    s=200,
                    marker='*',
                    label='Portfolio')

        # Add asset labels
        for i, (vol, ret, name) in enumerate(zip(returns_vol, returns_mean, asset_names)):
            ax2.annotate(name, (vol, ret),
                            xytext=(5, 5),
                            textcoords='offset points')

        ax2.set_xlabel('Annualized Volatility')
        ax2.set_ylabel('Annualized Return')
        ax2.set_title('Risk-Return Profile')
        ax2.legend()

        # 3. Correlation Heatmap
        ax3 = fig.add_subplot(gs[1, :])
        corr_matrix = np.corrcoef(returns)
        im = ax3.imshow(corr_matrix,
                        cmap='RdYlBu',
                        vmin=-1,
                        vmax=1)

        # Add correlation values
        for i in range(len(weights)):
            for j in range(len(weights)):
                text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                ha="center",
                                va="center",
                                color="black")

        ax3.set_xticks(range(len(weights)))
        ax3.set_yticks(range(len(weights)))
        ax3.set_xticklabels(asset_names)
        ax3.set_yticklabels(asset_names)
        plt.colorbar(im, ax=ax3)
        ax3.set_title('Asset Correlation Matrix')

        # Apply styling
        for ax in [ax1, ax2, ax3]:
            self.style.apply_style(ax, 'portfolio')

        plt.tight_layout()
        fig.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio"""
        return np.sqrt(252) * np.mean(returns) / np.std(returns)

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio"""
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns)
        return np.sqrt(252) * np.mean(returns) / downside_std if len(negative_returns) > 0 else np.inf

    def _calculate_skew(self, returns: np.ndarray) -> float:
        """Calculate returns skewness"""
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns)
        return (n * np.sum((returns - mean) ** 3)) / ((n - 1) * (n - 2) * std ** 3)

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

    def _calculate_drawdown_series(self, cumulative_returns: np.ndarray) -> np.ndarray:
        """Calculate full drawdown series"""
        peak = np.maximum.accumulate(cumulative_returns)
        return (cumulative_returns - peak) / peak

    def _calculate_trade_returns(self, positions: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Calculate returns for individual trades"""
        position_changes = np.diff(positions)
        trade_ends = np.where(position_changes != 0)[0]
        return returns[trade_ends]

    def _calculate_trade_durations(self, positions: np.ndarray) -> np.ndarray:
        """Calculate duration of each trade"""
        position_changes = np.diff(positions)
        trade_starts = np.where(position_changes != 0)[0]
        trade_ends = np.roll(trade_starts, -1)
        return trade_ends[:-1] - trade_starts[:-1]

    def _calculate_rolling_sharpe(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling Sharpe ratio"""
        rolled = self._rolling_window(returns, window)
        return np.sqrt(252) * (np.mean(rolled, axis=1) / np.std(rolled, axis=1))

    def _calculate_rolling_volatility(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling volatility"""
        rolled = self._rolling_window(returns, window)
        return np.sqrt(252) * np.std(rolled, axis=1)

    def _calculate_rolling_beta(self, returns: np.ndarray, benchmark_returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling beta to benchmark"""
        rolled_returns = self._rolling_window(returns, window)
        rolled_benchmark = self._rolling_window(benchmark_returns, window)
        return np.array([np.cov(r, b)[0,1]/np.var(b) for r, b in zip(rolled_returns, rolled_benchmark)])

    def _calculate_rolling_correlation(self, returns: np.ndarray, benchmark_returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling correlation to benchmark"""
        rolled_returns = self._rolling_window(returns, window)
        rolled_benchmark = self._rolling_window(benchmark_returns, window)
        return np.array([np.corrcoef(r, b)[0,1] for r, b in zip(rolled_returns, rolled_benchmark)])

    def _plot_position_bubbles(self, ax: plt.Axes, positions: np.ndarray, returns: np.ndarray, price_data: np.ndarray):
        """Create bubble plot of positions sized by return"""
        position_changes = np.diff(positions)
        trade_points = np.where(position_changes != 0)[0]
        trade_returns = returns[trade_points]

        # Scale returns for bubble size
        sizes = np.abs(trade_returns) * 1000
        colors = np.where(trade_returns > 0,
                            self.style.colors['train'],
                            self.style.colors['scatter'])

        ax.scatter(trade_points, price_data[trade_points],
                    s=sizes,
                    c=colors,
                    alpha=0.6)
        ax.plot(price_data, color='gray', alpha=0.5)

    @staticmethod
    def _rolling_window(arr: np.ndarray, window: int) -> np.ndarray:
        """Create rolling window views of array"""
        shape = (arr.shape[0] - window + 1, window)
        strides = (arr.strides[0], arr.strides[0])
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

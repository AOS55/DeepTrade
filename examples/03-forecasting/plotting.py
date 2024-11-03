import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

class Plotters:
    """Class for creating stylized plots for training and prediction visualization"""

    def __init__(self, save_dir: str = "./"):
        """
        Initialize the plotter

        Args:
            save_dir: Directory to save the plots
        """
        self.save_dir = save_dir
        self.style = {
            'train_color': '#2ecc71',
            'val_color': '#e74c3c',
            'mse_train_color': '#3498db',
            'mse_val_color': '#e67e22',
            'pred_color': '#3498db',
            'perfect_line_color': '#e74c3c',
            'hist_color': '#3498db',
            'actual_color': '#2ecc71',
            'predicted_color': '#e74c3c'
        }

    def _setup_axis_style(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str):
        """Helper method to apply consistent styling to axes"""
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_training_metrics(self,
                            train_history: Dict[str, List[float]],
                            val_history: Dict[str, List[float]],
                            save_name: str = "training_metrics.png"):
        """
        Plot training and validation metrics

        Args:
            train_history: Dictionary containing training metrics
            val_history: Dictionary containing validation metrics
            save_name: Name of the output file
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

        # Plot losses
        ax1.plot(train_history['loss'], color=self.style['train_color'],
                label='Training Loss', linewidth=2, alpha=0.8)
        ax1.plot(val_history['loss'], color=self.style['val_color'],
                label='Validation Loss', linewidth=2, alpha=0.8)
        self._setup_axis_style(ax1, 'Model Loss Over Time', 'Epoch', 'Loss')
        ax1.legend(fontsize=10, framealpha=0.9)

        # Plot MSE
        ax2.plot(train_history['MSE'], color=self.style['mse_train_color'],
                label='Training MSE', linewidth=2, alpha=0.8)
        ax2.plot(val_history['MSE'], color=self.style['mse_val_color'],
                label='Validation MSE', linewidth=2, alpha=0.8)
        self._setup_axis_style(ax2, 'Model MSE Over Time', 'Epoch', 'MSE')
        ax2.legend(fontsize=10, framealpha=0.9)

        plt.tight_layout(pad=3.0)
        fig.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_predictions(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        save_name: str = "prediction_results.png"):
        """
        Plot prediction results and residuals

        Args:
            y_true: True values (numpy array)
            y_pred: Predicted values (numpy array)
            save_name: Name of the output file
        """
        residuals = y_true - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.5, color=self.style['pred_color'], s=50)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val],
                linestyle="--", lw=2, label='Perfect Prediction',
                color=self.style['perfect_line_color'])
        self._setup_axis_style(ax1, 'Predicted vs Actual Values',
                             'Actual Values', 'Predicted Values')
        ax1.legend(fontsize=10)

        # Residuals distribution
        ax2.hist(residuals.flatten(), bins=30, density=True, alpha=0.3,
                color=self.style['hist_color'], edgecolor='black')
        self._setup_axis_style(ax2, 'Distribution of Residuals',
                             'Residual Value', 'Density')

        plt.tight_layout(pad=3.0)
        fig.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prediction_timeline(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_var: Optional[np.ndarray] = None,
                               save_name: str = "prediction_timeline.png"):
        """
        Plot prediction timeline with optional uncertainty bands

        Args:
            y_true: True values (numpy array)
            y_pred: Predicted values (numpy array)
            y_pred_var: Prediction variance (Optional[numpy array])
            save_name: Name of the output file
        """
        # Ensure we're working with the first dimension if multi-dimensional
        if y_true.ndim > 1:
            y_true = y_true[:, 0]
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 0]

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot actual timeline
        ax.plot(y_true, label='Actual', color=self.style['actual_color'],
                linewidth=2, alpha=0.8)

        # Plot predictions
        ax.plot(y_pred, label='Predicted', color=self.style['predicted_color'],
                linewidth=2, alpha=0.8)

        # Add variance bands if available
        if y_pred_var is not None:
            std_values = np.sqrt(y_pred_var)
            if std_values.ndim > 1:
                std_values = std_values[:, 0]

            ax.fill_between(range(len(y_pred)),
                           y_pred - 2*std_values,
                           y_pred + 2*std_values,
                           color=self.style['predicted_color'],
                           alpha=0.2,
                           label='95% Confidence Interval')

        self._setup_axis_style(ax, 'Prediction Timeline with Uncertainty',
                             'Time Step', 'Value')
        ax.legend(fontsize=10, loc='upper right')

        plt.tight_layout()
        fig.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.close()

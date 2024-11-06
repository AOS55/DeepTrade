import hydra
from omegaconf.omegaconf import OmegaConf
from omegaconf import DictConfig
import torch
import wandb
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from deeptrade.models import BasicEnsemble, GBM, OU, JDM
from deeptrade.util.finance import calculate_log_returns
from deeptrade.diagnostics.plotting import Plotter

from typing import Dict, List, Optional

class ForecastWorkspace:

    def __init__(self, cfg):

        # Setup directory configs
        self.work_dir = Path.cwd()
        self.cfg = cfg

        # Device setup
        if self.cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Setup wandb if enabled
        if cfg.use_wandb:
            run_name = f"forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project="deep_trade",
                       name=run_name,
                       config=OmegaConf.to_container(cfg))

        # Setup model
        self.model = BasicEnsemble(
            ensemble_size=cfg.ensemble_size,
            device=self.device,
            member_cfg=cfg.member_cfg
        )

        # TODO: Move this to a more appropriate place
        def as_numpy_array(value):
            return np.array(value)
        OmegaConf.register_new_resolver("as_numpy_array", as_numpy_array)

        # Generate price data
        generator = hydra.utils.instantiate(cfg.price_model)
        self.price_data = generator.generate(dt=cfg.dt, n_steps=cfg.n_steps)[0, :]
        self.log_price_data = calculate_log_returns(self.price_data)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        # For easier access
        self.input_size = cfg.member_cfg.in_size
        self.output_size = cfg.member_cfg.out_size

        # To plot figures
        self.plotter = Plotter(cfg, self.work_dir)

    def _create_sequences(self, data: np.ndarray, normalize: bool = False):
        """Create sequences from data for training"""
        X, y = [], []

        if normalize:
            data = self._normalize_windows(data)

        for idt in range(len(data) - self.input_size - self.output_size + 1):
            X.append(data[idt:idt + self.input_size])
            y.append(data[(idt + self.input_size):(idt + self.input_size + self.output_size)])
        return torch.FloatTensor(np.array(X)).to(self.device), torch.FloatTensor(np.array(y)).to(self.device)

    def _normalize_windows(self, log_returns: np.ndarray):
        """
        Normalize log returns data windows. For log returns, we can use standard
        normalization since log returns are already additive.

        Args:
            log_returns: Log returns data of shape [n_steps]
        Returns:
            Normalized log returns of shape [n_steps]
        """
        # Calculate rolling statistics for normalization
        window_stats = {}
        normalized_data = np.zeros_like(log_returns)

        for idx in range(0, len(log_returns) - self.input_size - self.output_size + 1):
            window = log_returns[idx:idx + self.input_size]
            window_mean = np.mean(window)
            window_std = np.std(window)
            window_stats[idx] = {'mean': window_mean, 'std': window_std}

            # Normalize both input and output windows using input window statistics
            full_window = log_returns[idx:idx + self.input_size + self.output_size]
            normalized_window = (full_window - window_mean) / (window_std + 1e-8)  # Add epsilon to prevent division by zero
            normalized_data[idx:idx + self.input_size + self.output_size] = normalized_window

        # Store normalization parameters for later use
        self.window_stats = window_stats

        return normalized_data

    def _denormalize_windows(self, normalized_data: np.ndarray, window_idx: Optional[int] = None):
        """
        Denormalize the data back to log returns space

        Args:
            normalized_data: Normalized log returns data
            window_idx: Optional index of the window to denormalize
        Returns:
            Denormalized log returns data
        """
        if not hasattr(self, 'window_stats'):
            raise ValueError("No normalization parameters found. Run normalize_windows first.")

        denormalized_data = np.zeros_like(normalized_data)

        if window_idx is not None:
            # Denormalize specific window
            if window_idx not in self.window_stats:
                raise ValueError(f"No normalization parameters for window {window_idx}")
            stats = self.window_stats[window_idx]
            denormalized_data = normalized_data * stats['std'] + stats['mean']
        else:
            # Denormalize all windows
            for idx, stats in self.window_stats.items():
                end_idx = idx + self.input_size + self.output_size
                denormalized_data[idx:end_idx] = (
                    normalized_data[idx:end_idx] * stats['std'] + stats['mean']
                )

        return denormalized_data


    def _create_data_loader(self, X: torch.Tensor, y: torch.Tensor, shuffle: bool = False):
        """Create data loader for mini-batch processing"""

        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            drop_last=False
        )

    def _convert_to_prices(self, log_returns: np.ndarray, initial_price: float):
        """
        Convert log returns back to raw prices

        Args:
            log_returns: Array of log returns
            initial_price: Initial price to start the conversion
        Returns:
            Array of prices
        """
        return initial_price * np.exp(np.cumsum(log_returns))

    def train(self):
        """Train the ensemble"""

        # Prepare data
        X, y = self._create_sequences(self.log_price_data, normalize=True)

        # Debug prints for data preparation
        print("Training data statistics:")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X mean: {X.mean().item():.4f}, std: {X.std().item():.4f}")
        print(f"y mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")

        split_idx = int((1-self.cfg.val_split) * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create data loaders
        train_loader = self._create_data_loader(X_train, y_train, shuffle=self.cfg.shuffle_dataloader)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)

        train_history = {"loss": [], "MSE": []}
        val_history = {"loss": [], "MSE": []}

        for epoch in range(self.cfg.epochs):

            # Train
            self.model.train()
            epoch_losses = []
            epoch_mses = []

            # ensemble_indices = [torch.randperm(len(X_train)) for _ in range(self.cfg.ensemble_size)]  # Shuffling the indices

            # for idx in range(0, len(X_train), self.cfg.batch_size):
                # batch_end = min(idx + self.cfg.batch_size, len(X_train))

            for batch_X, batch_y in train_loader:
                # Create bootstrapped samples for each ensemble member
                bootstrap_X = []
                bootstrap_y = []

                for _ in range(self.cfg.ensemble_size):
                    member_indices = torch.randint(0, len(batch_X), (len(batch_X),))
                    bootstrap_X.append(X_train[member_indices])
                    bootstrap_y.append(y_train[member_indices])

                self.optimizer.zero_grad()
                loss, meta = self.model.loss(bootstrap_X, bootstrap_y)
                loss.backward()
                self.optimizer.step()

                batch_mse = np.mean([meta[f'model_{i}']['train_mse'] for i in range(self.cfg.ensemble_size)])
                epoch_losses.append(loss.item())
                epoch_mses.append(batch_mse)

            # Record training metrics
            train_history['loss'].append(np.mean(epoch_losses))
            train_history['MSE'].append(np.mean(epoch_mses))

            # Validate

            self.model.eval()
            val_losses = []
            val_mses = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:

                    # For validation, we don't need bootstrapping
                    loss, meta = self.model.loss([batch_X] * self.cfg.ensemble_size, [batch_y] * self.cfg.ensemble_size)
                    batch_mse = np.mean([meta[f'model_{i}']['train_mse'] for i in range(self.cfg.ensemble_size)])
                    val_losses.append(loss.item())
                    val_mses.append(batch_mse)

            # Record validation metrics
            val_history['loss'].append(np.mean(val_losses))
            val_history['MSE'].append(np.mean(val_mses))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}")
                print(f"Training - Loss: {train_history['loss'][-1]:.4f}, MSE: {train_history['MSE'][-1]:.4f}")
                print(f"Validation - Loss: {val_history['loss'][-1]:.4f}, MSE: {val_history['MSE'][-1]:.4f}")

        return train_history, val_history

    def evaluate(self):

        # Evaluate the ensemble
        with torch.no_grad():

            # Create normalized sequences
            input_sequence, _ = self._create_sequences(self.log_price_data, normalize=True)

            # Get model predictions
            predictions, prediction_variances = self.model(input_sequence)
            predictions = predictions.mean(axis=0).detach().cpu().numpy()

            # Denormalize predictions
            denormalized_predictions = np.array([
                self._denormalize_windows(pred, idx)
                for idx, pred in enumerate(predictions)
            ])

            # Convert predictions to prices with correct time alignment
            price_predictions = np.zeros((len(denormalized_predictions), self.output_size))

            for idx, pred in enumerate(denormalized_predictions):
                start_idx = idx + self.input_size
                initial_price = self.price_data[start_idx - 1]
                pred_prices = self._convert_to_prices(pred, initial_price)
                price_predictions[idx] = pred_prices

            # Create forecast array that matches the time axis
            full_forecast = np.zeros((len(predictions), len(self.price_data)))

            for i in range(len(predictions)):
                start_idx = i + self.input_size
                end_idx = start_idx + self.output_size
                # Only fill the relevant portion of the forecast
                if end_idx <= len(self.price_data):
                    full_forecast[i, start_idx:end_idx] = price_predictions[i, :len(self.price_data)-start_idx]

            forecast_frequency = 100
            self.plotter.plot_forecast(
                actual_values=self.price_data,
                forecast_values=full_forecast,
                forecast_frequency=forecast_frequency,
                forecast_variance=None
            )

            # Create sequences for the last portion of data for visualization
            X_test, y_test = self._create_sequences(
                self.log_price_data[-100:],  # Use last 100 points
                normalize=True
            )
            test_predictions, test_variances = self.model(X_test)

            actual_values = y_test.detach().cpu().numpy()[:, 0]
            forecast_values = test_predictions.mean(axis=0).detach().cpu().numpy()[:, 0]
            forecast_variance = test_variances.mean(axis=0).detach().cpu().numpy()[:, 0] if test_variances is not None else None

            # Plot detailed forecast analysis
            self.plotter.plot_forecast_analysis(
                actual_values=actual_values,
                forecast_values=forecast_values,
                forecast_variance=forecast_variance,
            )

    def run(self):
        """Main execution flow"""
        train_history, val_history = self.train()
        self.plotter.plot_training_metrics(train_history, val_history)
        self.evaluate()


@hydra.main(config_path="../configs", config_name="03-single_forecast.yaml")
def main(cfg):
    workspace = ForecastWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

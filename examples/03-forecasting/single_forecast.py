import hydra
from omegaconf.omegaconf import OmegaConf
import torch
import wandb
import torch.nn as nn
from datetime import datetime
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from deeptrade.models import BasicEnsemble, GBM, OU, JDM
from deeptrade.util.finance import calculate_log_returns

from typing import Dict, List, Optional
from plotting import Plotters

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
        self.raw_price_data = generator.generate(dt=cfg.dt, n_steps=cfg.n_steps)[0, :]
        self.price_data = calculate_log_returns(self.raw_price_data)
        # self.price_data = self.raw_price_data

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        # For easier access
        self.input_size = cfg.member_cfg.in_size
        self.output_size = cfg.member_cfg.out_size

        # To plot figures
        self.plotters = Plotters(self.work_dir)

    def train(self):
        """Train the ensemble"""

        # Prepare data
        X, y = self._create_sequences(self.price_data)

        # Debug prints for data preparation
        print("Training data statistics:")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X mean: {X.mean().item():.4f}, std: {X.std().item():.4f}")
        print(f"y mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")

        split_idx = int((1-self.cfg.val_split) * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_history = {"loss": [], "MSE": []}
        val_history = {"loss": [], "MSE": []}

        for epoch in range(self.cfg.epochs):

            # Train

            self.model.train()
            epoch_losses = []
            epoch_mses = []

            ensemble_indices = [torch.randperm(len(X_train)) for _ in range(self.cfg.ensemble_size)]  # Shuffling the indices

            for idx in range(0, len(X_train), self.cfg.batch_size):
                batch_end = min(idx + self.cfg.batch_size, len(X_train))

                # Create bootstrapped samples for each ensemble member
                bootstrap_X = []
                bootstrap_y = []

                for indices in ensemble_indices:
                    member_indices = indices[idx:batch_end]
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
                for i in range(0, len(X_val), self.cfg.batch_size):
                    batch_X = X_val[i:i + self.cfg.batch_size]
                    batch_y = y_val[i:i + self.cfg.batch_size]

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

    def _create_sequences(self, data: np.ndarray):
        """Create sequences from data for training"""
        X, y = [], []
        for idt in range(len(data) - self.input_size - self.output_size + 1):
            X.append(data[idt:idt + self.input_size])
            y.append(data[(idt + self.input_size):(idt + self.input_size + self.output_size)])
        return torch.FloatTensor(np.array(X)).to(self.device), torch.FloatTensor(np.array(y)).to(self.device)

    def evaluate(self):

        # Evaluate the ensemble
        with torch.no_grad():
            input_sequence = torch.FloatTensor(self.price_data[-self.input_size:]).unsqueeze(0).to(self.device)
            predictions, _ = self.model(input_sequence)

            # Create sequences for the last portion of data for visualization
            X_test, y_test = self._create_sequences(
                self.price_data[-100:],  # Use last 100 points
            )
            test_predictions, test_variances = self.model(X_test)

            # Plot predictions
            self.plotters.plot_predictions(y_test.detach().cpu().numpy(), test_predictions.mean(axis=0).detach().cpu().numpy())
            self.plotters.plot_prediction_timeline(y_test.detach().cpu().numpy(), test_predictions.mean(axis=0).detach().cpu().numpy(), None)

    def run(self):
        """Main execution flow"""
        train_history, val_history = self.train()
        self.plotters.plot_training_metrics(train_history, val_history)
        self.evaluate()

@hydra.main(config_path="../configs/03-forecasting", config_name="single_forecast.yaml")
def main(cfg):
    workspace = ForecastWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

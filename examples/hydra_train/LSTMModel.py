import torch
import torch.nn as nn
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt

from deeptrade.models import BasicEnsemble
from typing import Dict, List, Optional


def plot_training_metrics(train_history: Dict[str, List[float]], val_history: Dict[str, List[float]]):
    """Enhanced version of training metrics visualization"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Plot losses with enhanced styling
    ax1.plot(train_history['loss'], color='#2ecc71', label='Training Loss',
             linewidth=2, alpha=0.8)
    ax1.plot(val_history['loss'], color='#e74c3c', label='Validation Loss',
             linewidth=2, alpha=0.8)
    ax1.set_title('Model Loss Over Time', fontsize=14, pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot MSE with enhanced styling
    ax2.plot(train_history['mse'], color='#3498db', label='Training MSE',
             linewidth=2, alpha=0.8)
    ax2.plot(val_history['mse'], color='#e67e22', label='Validation MSE',
             linewidth=2, alpha=0.8)
    ax2.set_title('Model MSE Over Time', fontsize=14, pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    fig.savefig("training_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Enhanced version of prediction visualization"""

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    residuals = y_true_np - y_pred_np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Scatter plot with enhanced styling
    ax1.scatter(y_true_np, y_pred_np, alpha=0.5, color='#3498db', s=50)
    min_val = min(y_true_np.min(), y_pred_np.min())
    max_val = max(y_true_np.max(), y_pred_np.max())
    ax1.plot([min_val, max_val], [min_val, max_val],
             linestyle="--", lw=2, label='Perfect Prediction', color='#e74c3c')
    ax1.set_title('Predicted vs Actual Values', fontsize=14, pad=15)
    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Residuals distribution with enhanced styling
    # sns.kdeplot(data=residuals.flatten(), ax=ax2, color='#2ecc71', fill=True)
    ax2.hist(residuals.flatten(), bins=30, density=True, alpha=0.3,
             color='#3498db', edgecolor='black')
    ax2.set_title('Distribution of Residuals', fontsize=14, pad=15)
    ax2.set_xlabel('Residual Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    fig.savefig("prediction_results.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_timeline(y_true: torch.Tensor, y_pred: torch.Tensor, y_pred_var: Optional[torch.Tensor] = None):
    """Plot continuous timeline with predictions and variance bands"""
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    y_true = y_true_np[:, 0]
    y_pred = y_pred_np[:, 0]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot actual timeline
    ax.plot(y_true, label='Actual', color='#2ecc71',
            linewidth=2, alpha=0.8)
    
    # Plot predictions
    ax.plot(y_pred, label='Predicted', color='#e74c3c',
            linewidth=2, alpha=0.8)
    
    # Add variance bands if available
    if y_pred_var is not None:
        y_pred_std = np.sqrt(y_pred_var.cpu().numpy())
        std_values = []
        for i in range(len(y_pred_std)):
            std_values.append(y_pred_std[i, 0])
        std_values = np.array(std_values)
        
        ax.fill_between(range(len(predicted_values)),
                       predicted_values - 2*std_values,  # 2 standard deviations
                       predicted_values + 2*std_values,
                       color='#e74c3c', alpha=0.2,
                       label='95% Confidence Interval')
    
    ax.set_title('Prediction Timeline with Uncertainty', fontsize=14, pad=15)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig("prediction_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_sequences(data, input_size, output_size):
    """Create sequences from data for training"""
    X, y = [], []
    for i in range(len(data) - input_size - output_size + 1):
        X.append(data[i:(i + input_size)])
        y.append(data[(i + input_size):(i + input_size + output_size)])
    return torch.FloatTensor(X), torch.FloatTensor(y)


def train_ensemble(ensemble, price_data, input_size, output_size, ensemble_size, num_epochs=100, batch_size=32):
    """
    Train the ensemble model and collect metrics
    """
    optimizer = torch.optim.AdamW(ensemble.parameters())  # Using AdamW instead of Adam

    # Prepare data
    X, y = create_sequences(price_data, input_size, output_size)

    # Split data into train and validation sets (80-20 split)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_history = {'loss': [], 'mse': []}
    val_history = {'loss': [], 'mse': []}

    for epoch in range(num_epochs):
        # Training phase
        ensemble.train()
        epoch_losses = []
        epoch_mses = []

        ensemble_indices = [torch.randperm(len(X_train)) for _ in range(ensemble_size)]

        
        for idx in range(0, len(X_train), batch_size):
            batch_end = min(idx + batch_size, len(X_train))

            # Create bootstrapped samples for each ensemble member
            bootstrap_X = []
            bootstrap_y = []
            batch_size_actual = len(batch_X)
            
            for indices in ensemble_indices:
                member_indices = indices[idx:batch_end]
                bootstrap_X.append(batch_X[member_indices])
                bootstrap_y.append(batch_y[member_indices])

            optimizer.zero_grad()
            loss, meta = ensemble.loss(bootstrap_X, bootstrap_y)
            loss.backward()
            optimizer.step()

            batch_mse = np.mean([meta[f'model_{i}']['train_mse'] for i in range(ensemble_size)])
            epoch_losses.append(loss.item())
            epoch_mses.append(batch_mse)

        # Record training metrics
        train_history['loss'].append(np.mean(epoch_losses))
        train_history['mse'].append(np.mean(epoch_mses))

        # Validation phase
        ensemble.eval()
        val_losses = []
        val_mses = []

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i + batch_size]
                batch_y = y_val[i:i + batch_size]

                # For validation, we don't need bootstrapping
                loss, meta = ensemble.loss([batch_X] * ensemble_size, [batch_y] * ensemble_size)
                batch_mse = np.mean([meta[f'model_{i}']['train_mse'] for i in range(ensemble_size)])
                val_losses.append(loss.item())
                val_mses.append(batch_mse)

        # Record validation metrics
        val_history['loss'].append(np.mean(val_losses))
        val_history['mse'].append(np.mean(val_mses))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"Training - Loss: {train_history['loss'][-1]:.4f}, MSE: {train_history['mse'][-1]:.4f}")
            print(f"Validation - Loss: {val_history['loss'][-1]:.4f}, MSE: {val_history['mse'][-1]:.4f}")

    return train_history, val_history


if __name__ == "__main__":

    input_size = 10
    output_size = 5
    ensemble_size = 5

    # Create configuration for ensemble members
    member_cfg = DictConfig({
        "_target_": "deeptrade.models.PricePredictionModel",
        "input_size": input_size,  # Length of input sequence
        "hidden_size": 64,
        "output_size": output_size,  # Number of future steps to predict
    })

    # Create the ensemble
    ensemble = BasicEnsemble(
        ensemble_size=ensemble_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        member_cfg=member_cfg,
        propagation_method="expectation"
    )

    # Generate sample price data
    np.random.seed(42)
    price_data = np.cumsum(np.random.randn(1000)) + 100
    price_data = (price_data - price_data.mean()) / price_data.std()

    # Train the ensemble and get histories
    train_history, val_history = train_ensemble(ensemble, price_data, input_size, output_size, ensemble_size, num_epochs=1000, batch_size=32)

    # Generate plots
    plot_training_metrics(train_history, val_history)

    # Make predictions for the last sequence
    with torch.no_grad():
        input_sequence = torch.FloatTensor(price_data[-member_cfg.input_size:]).unsqueeze(0)
        predictions, _ = ensemble(input_sequence)

        # Create sequences for the last portion of data for visualization
        X_test, y_test = create_sequences(
            price_data[-100:],  # Use last 100 points
            input_size,
            output_size
        )
        test_predictions, test_variances = ensemble(X_test)

        # Plot predictions
        plot_predictions(y_test, test_predictions)
        plot_prediction_timeline(y_test, test_predictions, test_variances)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional

import matplotlib.pyplot as plt


def prepare_data(
    prices: np.ndarray, 
    sequence_length: int = 50, 
    batch_size: int = 32,
    train_split: float = 0.8
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Prepare financial time series data for the VAE.
    
    Args:
        prices: Price data from GBM generator with shape [n_instruments, n_steps+1]
        sequence_length: Length of sequences to create
        batch_size: Batch size for training
        train_split: Fraction of data to use for training (if < 1.0)
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (if train_split < 1.0)
    
    Raises:
        ValueError: If prices array doesn't have correct shape or if parameters are invalid
    """
    # Input validation
    if not isinstance(prices, np.ndarray):
        raise ValueError(f"prices must be a numpy array, got {type(prices)}")
    
    if len(prices.shape) != 2:
        raise ValueError(f"prices must be 2D array [n_instruments, n_steps+1], got shape {prices.shape}")
    
    if sequence_length >= prices.shape[1]:
        raise ValueError(f"sequence_length ({sequence_length}) must be less than number of time steps ({prices.shape[1]})")
    
    # Normalize the data (per instrument)
    means = np.mean(prices, axis=1, keepdims=True)
    stds = np.std(prices, axis=1, keepdims=True)
    normalized_prices = (prices - means) / stds
    
    # Create sequences
    sequences = []
    n_instruments, n_steps = prices.shape
    
    for i in range(n_steps - sequence_length):
        # Extract sequence and transpose to [sequence_length, n_instruments]
        sequence = normalized_prices[:, i:(i + sequence_length)].T
        sequences.append(sequence)
    
    # Convert to tensor [n_sequences, sequence_length, n_instruments]
    sequences = torch.FloatTensor(np.array(sequences))
    
    # Split into train and validation if requested
    if train_split < 1.0:
        train_size = int(len(sequences) * train_split)
        train_sequences = sequences[:train_size]
        val_sequences = sequences[train_size:]
        
        train_dataset = TensorDataset(train_sequences)
        val_dataset = TensorDataset(val_sequences)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )
        
        return train_loader, val_loader
    
    # If no split requested, return single loader
    dataset = TensorDataset(sequences)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    return train_loader, None


def train_vae(model, train_loader, optimizer, epochs=100, device='cpu'):
    """
    Train the VAE model.
    
    Args:
        model: VAE model
        train_loader: DataLoader containing training data
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to train on
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):  # Note the (data,) to unpack the tuple
            data = data.to(device)
            optimizer.zero_grad()
            
            # recon_batch, mu, log_var = model(data)
            loss, recon_loss, kld_loss = model.loss(data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Average loss = {avg_loss:.4f}')
            

def plot_reconstruction(
    model,
    data_loader: DataLoader,
    device: torch.device,
    n_samples: int = 3,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot original vs reconstructed time series using pure matplotlib styling.
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader containing test data
        device: Device to run inference on
        n_samples: Number of random samples to plot
        figsize: Figure size for the plot
    """
    # Define colors - modern palette
    colors = [
        ('#1f77b4', '#7cc7ff'),  # Blue pair
        ('#2ca02c', '#98df8a'),  # Green pair
        ('#ff7f0e', '#ffbb78'),  # Orange pair
        ('#d62728', '#ff9896'),  # Red pair
        ('#9467bd', '#c5b0d5'),  # Purple pair
    ]
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
    if n_samples == 1:
        axes = [axes]
    
    # Set figure background
    fig.patch.set_facecolor('#f0f0f0')
    
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        batch, = next(iter(data_loader))
        batch = batch.to(device)
        
        # Get reconstructions
        recon_batch, _, _ = model(batch)
        
        # Move tensors to CPU and convert to numpy
        original = batch.cpu().numpy()
        reconstructed = recon_batch.cpu().numpy()
        
        # Plot samples
        time_steps = np.arange(original.shape[1])
        
        for i in range(min(n_samples, len(original))):
            ax = axes[i]
            
            # Set plot background
            ax.set_facecolor('#ffffff')
            
            # Add grid with custom style
            ax.grid(True, linestyle='--', alpha=0.3, color='#666666', linewidth=0.5)
            
            # Plot each feature
            for j in range(original.shape[2]):
                orig_color, recon_color = colors[j % len(colors)]
                
                # Plot original data
                ax.plot(time_steps, original[i, :, j],
                       label=f'Original (Feature {j+1})',
                       color=orig_color,
                       linestyle='-',
                       linewidth=2.5,
                       alpha=0.9,
                       zorder=2)
                
                # Plot reconstructed data
                ax.plot(time_steps, reconstructed[i, :, j],
                       label=f'Reconstructed (Feature {j+1})',
                       color=recon_color,
                       linestyle='--',
                       linewidth=2.5,
                       alpha=0.9,
                       zorder=2)
            
            # Customize spines
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('#222222')
            
            # Set title with custom style
            ax.set_title(f'Sample {i+1}',
                        fontsize=12,
                        pad=10,
                        fontweight='bold',
                        color='#222222')
            
            # Customize axis labels
            ax.set_xlabel('Time Steps',
                         fontsize=10,
                         color='#222222',
                         labelpad=8)
            ax.set_ylabel('Normalized Price',
                         fontsize=10,
                         color='#222222',
                         labelpad=8)
            
            # Customize ticks
            ax.tick_params(axis='both',
                          which='major',
                          labelsize=9,
                          colors='#222222',
                          length=4,
                          width=0.5,
                          direction='out')
            
            # Add legend with custom style
            legend = ax.legend(bbox_to_anchor=(1.02, 1),
                             loc='upper left',
                             frameon=True,
                             framealpha=1,
                             edgecolor='#cccccc')
            
            # Customize legend appearance
            frame = legend.get_frame()
            frame.set_facecolor('#ffffff')
            frame.set_linewidth(0.5)
            
            # Set text color in legend
            for text in legend.get_texts():
                text.set_color('#222222')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
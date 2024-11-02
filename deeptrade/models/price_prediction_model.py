import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any

class PricePredictionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super().__init__()

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=1,  # Single feature (price)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=1,  # Single feature (price)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

        # Required model properties for ensemble
        self.in_size = input_size
        self.out_size = output_size
        self.deterministic = True # Indicate this is a deterministic model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass returning mean predictions and optional logvars"""

        # Encoder
        _, (hidden, cell) = self.encoder(x.unsqueeze(-1))
        decoder_input = x[:, -1:, None]
        predictions = []

        for _ in range(self.out_size):

            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.linear(output)
            predictions.append(pred)
            decoder_input = pred

        predictions = torch.cat(predictions, dim=1)
        predictions = predictions.squeeze(-1)

        return predictions, None

    def _compute_loss(self, model_in: torch.Tensor, target: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Helper method to compute loss and metrics"""
        with torch.set_grad_enabled(training):
            predictions, _ = self.forward(model_in)
            loss = self.criterion(predictions, target)

        meta = {
            f"{'train' if training else 'eval'}_mse": loss.item()
        }

        return loss, meta

    def loss(self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute training loss"""
        assert target is not None

        if model_in.size(0) != target.size(0):
            raise ValueError(f"Batch size mismatch: {model_in.size(0)} vs {target.size(0)}")

        if target.size(1) != self.out_size:
            raise ValueError(f"Output size mismatch: {target.size(1)} vs {self.out_size}")

        return self._compute_loss(model_in, target, training=True)

    def eval_score(self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute evaluation score"""
        assert target is not None
        return self._compute_loss(model_in, target, training=False)

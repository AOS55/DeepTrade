import torch
from torch import nn
from typing import Union, Optional, Tuple, Dict, Any

class LSTMForecastModel(nn.Module):
    def __init__(self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 2,
        hid_size: int = 200,
    ):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=1,  # Single feature (price)
            hidden_size=hid_size,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True
        ).to(device)

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=1,  # Single feature (price)
            hidden_size=hid_size,
            num_layers=num_layers,
            batch_first=True
        ).to(device)

        self.linear = nn.Linear(hid_size, 1).to(device)
        self.criterion = nn.MSELoss().to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
        # Required model properties for ensemble
        self.deterministic = True # Indicate this is a deterministic model
        self.device = device

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass returning mean predictions and optional logvars"""
        
        # Encoder
        _, (hidden, cell) = self.encoder(x.unsqueeze(-1))
        decoder_input = x[:, -1:, None]
        predictions = []

        for step in range(self.out_size):

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

            # Add regularization if required
            if training:
                l2_reg = torch.norm(self.linear.weight)
                loss += 1e-5 * l2_reg

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

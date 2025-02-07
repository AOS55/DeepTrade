import torch
from torch import nn
from typing import Union, Optional, Tuple, Dict, Any


class MLPForecastModel(nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 2,
        hid_size: int = 200,
    ):

        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        layers = [nn.Linear(in_size, hid_size), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hid_size, hid_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Linear(hid_size, out_size))

        self.network = nn.Sequential(*layers).to(device)
        self.criterion = nn.MSELoss().to(device)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Required model properties for ensemble training
        self.deterministic = True
        self.device = device

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass returning mean predictions and optional logvars"""
        # Flatten input if needed
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        predictions = self.network(x_flat)
        return predictions, None

    def _compute_loss(self, model_in: torch.Tensor, target: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
            """Helper method to compute loss and metrics"""
            with torch.set_grad_enabled(training):
                predictions, _ = self.forward(model_in)
                loss = self.criterion(predictions, target)

                # Add L2 regularization if training
                if training:
                    l2_reg = 0
                    for param in self.parameters():
                        l2_reg += torch.norm(param)
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

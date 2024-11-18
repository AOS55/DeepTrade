# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Sequence
from typing import List, Tuple

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

import deeptrade.types
import deeptrade.util.nn_math


def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""

    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        deeptrade.util.math.truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)
    if isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            deeptrade.util.math.truncated_normal_(m.weight.data[i], std=stddev)
        m.bias.data.fill_(0.0)


class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...])
            if self.use_bias:
                return xw + self.bias[self.elite_models, ...]
            else:
                return xw
        else:
            xw = x.matmul(self.weight)
            if self.use_bias:
                return xw + self.bias
            else:
                return xw

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


def to_tensor(x: deeptrade.types.TensorType):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise ValueError("Input must be torch.Tensor or np.ndarray.")


# TODO [maybe] this could be computed in closed form but this is much simpler
def get_cnn_output_size(
    conv_layers: nn.ModuleList,
    num_input_channels: int,
    image_shape: Tuple[int, int],
) -> int:
    dummy = torch.zeros(1, num_input_channels, image_shape[0], image_shape[1])
    with torch.no_grad():
        for cnn_layer in conv_layers:
            dummy = cnn_layer(dummy)
    return dummy.shape[1:]


class Conv2dEncoder(nn.Module):
    def __init__(
        self,
        layers_config: Tuple[Tuple[int, int, int, int], ...],
        image_shape: Tuple[int, int],
        encoding_size: int,
        activation_func: str = "ReLU",
    ):
        """Implements an image encoder with a desired configuration.

        The architecture will be a number of `torch.nn.Conv2D` layers followed by a
        single linear layer. The given activation function will be applied to all
        convolutional layers, but not to the linear layer. If the flattened output
        of the last layer is equal to ``encoding_size``, then a `torch.nn.Identity`
        will be used instead of a linear layer.

        Args:
            layers_config (tuple(tuple(int))): each tuple represents the configuration
                of a convolutional layer, in the order
                (in_channels, out_channels, kernel_size, stride). For example,
                ( (3, 32, 4, 2), (32, 64, 4, 3) ) adds a layer
                `nn.Conv2d(3, 32, 4, stride=2)`, followed by
                `nn.Conv2d(32, 64, 4, stride=3)`.
            image_shape (tuple(int, int)): the shape of the image being encoded, which
                is used to compute the size of the output of the last convolutional
                layer.
            encoding_size (int): the desired size of the encoder's output.
            activation_func (str): the `torch.nn` activation function to use after
                each convolutional layer. Defaults to ``"ReLU"``.
        """
        super().__init__()
        activation_cls = getattr(torch.nn, activation_func)
        conv_modules = []
        for layer_cfg in layers_config:
            conv_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        layer_cfg[0], layer_cfg[1], layer_cfg[2], stride=layer_cfg[3]
                    ),
                    activation_cls(),
                )
            )
        self.convs = nn.ModuleList(conv_modules)
        cnn_out_size = np.prod(
            get_cnn_output_size(self.convs, layers_config[0][0], image_shape)
        )
        if cnn_out_size == encoding_size:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(cnn_out_size, encoding_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        conv = self.convs[0](obs)
        for i in range(1, len(self.convs)):
            conv = self.convs[i](conv)
        h = conv.view(conv.size(0), -1)
        return self.fc(h)


# decoder config's first element is the shape of the input map, second element is as
# the encoder config but for Conv2dTranspose layers.
class Conv2dDecoder(nn.Module):
    """Implements an image decoder with a desired configuration.

    The architecture will be a linear layer, followed by a number of
    `torch.nn.ConvTranspose2D` layers. The given activation function will be
    applied only to all deconvolution layers except the last one.

    Args:
        encoding_size (int): the size that was used for the encoding.
        deconv_input_shape (tuple of 3 ints): the that the output of the linear layer
            will be converted to when passing to the deconvolution layers.
        layers_config (tuple(tuple(int))): each tuple represents the configuration
            of a deconvolution layer, in the order
            (in_channels, out_channels, kernel_size, stride). For example,
            ( (3, 32, 4, 2), (32, 64, 4, 3) ) adds a layer
            `nn.ConvTranspose2d(3, 32, 4, stride=2)`, followed by
            `nn.ConvTranspose2d(32, 64, 4, stride=3)`.
        encoding_size (int): the desired size of the encoder's output.
        activation_func (str): the `torch.nn` activation function to use after
            each deconvolution layer. Defaults to ``"ReLU"``.
    """

    def __init__(
        self,
        encoding_size: int,
        deconv_input_shape: Tuple[int, int, int],
        layers_config: Tuple[Tuple[int, int, int, int], ...],
        activation_func: str = "ReLU",
    ):
        super().__init__()
        self.encoding_size = encoding_size
        self.deconv_input_shape = deconv_input_shape
        activation_cls = getattr(torch.nn, activation_func)
        self.fc = nn.Linear(encoding_size, np.prod(self.deconv_input_shape))
        deconv_modules = []
        for i, layer_cfg in enumerate(layers_config):
            layer = nn.ConvTranspose2d(
                layer_cfg[0], layer_cfg[1], layer_cfg[2], stride=layer_cfg[3]
            )
            if i == len(layers_config) - 1:
                # no activation after the last layer
                deconv_modules.append(layer)
            else:
                deconv_modules.append(nn.Sequential(layer, activation_cls()))
        self.deconvs = nn.ModuleList(deconv_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deconv = self.fc(x).view(-1, *self.deconv_input_shape)
        for i in range(len(self.deconvs)):
            deconv = self.deconvs[i](deconv)
        return deconv


class VAE(nn.Module):
    
    def __init__(self,
                 obs_dim: int,
                 code_dim: int,
                 vae_beta: float,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.code_dim = code_dim
        
        self.make_networks(obs_dim, code_dim)
        self.beta = vae_beta
        
        # TODO: add custom weight initialization
        self.device = device
        
    def make_networks(self, obs_dim, code_dim):
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 150), nn.ReLU(),
            nn.Linear(150, 150), nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 150), nn.ReLU(),
            nn.Linear(150, 150), nn.ReLU(),
            nn.Linear(150, obs_dim)
        )
    
    def encode(self, obs):
        enc_features = self.encoder(obs)
        mu = self.encoder_mu(enc_features)
        logvar = self.enc_logvar(enc_features)
        std = (0.5 * logvar).exp()
        return mu, std, logvar

    def forward(self, obs, epsilon):
        mu, stds, logvar = self.encode(obs)
        code = epsilon * stds + mu
        obs_distr_params = self.decoder(code)
        return obs_distr_params, (mu, stds, logvar)

    def loss(self, obs):
        # TODO: Does this need seeding?
        epsilon = torch.randn([obs.size(0), self.code_dim]).to(self.device) 
        obs_distr_params, (mu, stds, logvar) = self.forward(obs, epsilon)
        kle = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        log_prob = F.mse_loss(obs, obs_distr_params, reduction='none')
        loss = self.beta * kle + log_prob.mean()
        return loss, log_prob.sum(list(range(1, len(log_prob.shape)))).view(log_prob.shape[0], 1)


class ConvVAE(nn.Module):
    def __init__(self, sequence_length, n_features, latent_dim=8, hidden_dim=64, beta=1.0):
        """
        Variational Autoencoder using CNNs for time series data.
        
        Args:
            sequence_length (int): Length of the input time series
            n_features (int): Number of features per time step
            latent_dim (int): Dimension of the latent space
            hidden_dim (int): Number of hidden units in the encoder/decoder
            beta (float): Weight for the KL divergence term in the loss function, 1.0 is standard VAE
        """
        super().__init__()
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Flatten size for FC layers
        self.flatten_size = hidden_dim * 2 * sequence_length
        
        # Mean and variance for latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, n_features, kernel_size=3, padding=1),
        )
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
    def encode(self, x):
        """Encode the input into latent space parameters."""
        x = x.permute(0, 2, 1)  # [batch, features, sequence]
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def decode(self, z):
        """Decode from latent space to time series."""
        x = self.decoder_input(z)
        x = x.reshape(-1, self.hidden_dim * 2, self.sequence_length)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)  # [batch, sequence, features]
        return x
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling from latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass through the VAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss(self, x):
        """
        Loss function for the VAE combining reconstruction loss and KL divergence.
        
        Args:
            x: Original time series
        """
        recon, mu, log_var = self.forward(x)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + self.beta * kld_loss
        
        return total_loss, recon_loss, kld_loss


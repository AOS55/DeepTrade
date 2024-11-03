import numpy as np
import torch
import omegaconf
import gymnasium as gym
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
from IPython import display

import deeptrade.models as models
import deeptrade.planning as planning
import deeptrade.util.common as common_util
from deeptrade.models.util import ConvVAE
from deeptrade.env import reward_fns, termination_fns
import deeptrade.models.util as model_util

import deeptrade.types
import deeptrade.util.math

from typing import Tuple, Optional, Callable
import torch


class VAETransitionRewardModel(models.OneDTransitionRewardModel):
    """Wrapper for dynamics models that work with VAE-encoded states and rewards.
    
    This model extends OneDTransitionRewardModel to properly handle the dimensionality
    of VAE-encoded states while maintaining the reward prediction separately.
    """
    
    # First modify the VAETransitionRewardModel class initialization:
    def __init__(
        self,
        model: models.Model,
        vae: torch.nn.Module,
        target_is_delta: bool = True,
        normalize: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[deeptrade.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
        num_elites: Optional[int] = None,
    ):
        # Store VAE before calling super()
        self.vae = vae
        # Get the input size from the base model if it exists
        in_size = getattr(model, 'in_size', None)
        if in_size is None and hasattr(model, 'model'):
            in_size = getattr(model.model, 'in_size', None)
        
        super().__init__(
            model,
            target_is_delta=target_is_delta,
            normalize=normalize,
            normalize_double_precision=normalize_double_precision,
            learned_rewards=learned_rewards,
            obs_process_fn=obs_process_fn,
            no_delta_list=no_delta_list,
            num_elites=num_elites
        )
        # Add input size as attribute if we found it
        if in_size is not None:
            self.in_size = in_size

    # Then modify how you create the dynamics model in train_pets_with_vae_preprocessing:
        # Create dynamics model
        cfg.dynamics_model.in_size = latent_dim + 1  # latent_dim + action_dim 
        base_model = common_util.create_one_dim_tr_model(cfg, (latent_dim,), act_shape)
            
        dynamics_model = VAETransitionRewardModel(
            base_model,
            vae=vae,
            target_is_delta=cfg.algorithm.target_is_delta,
            normalize=cfg.algorithm.normalize,
            normalize_double_precision=cfg.algorithm.get("normalize_double_precision", False),
            learned_rewards=cfg.algorithm.learned_rewards,
            obs_process_fn=cfg.overrides.get("obs_process_fn", None),
            no_delta_list=cfg.overrides.get("no_delta_list", None),
            num_elites=cfg.overrides.get("num_elites", None)
    )
    
    def _process_batch(
        self, batch: deeptrade.types.TransitionBatch, _as_float: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of transitions for training or evaluation.
        
        Encodes the observations using the VAE and handles rewards separately.
        """
        obs, action, next_obs, reward, _, _ = batch.astuple()
        
        # Convert to tensors and move to device
        obs = model_util.to_tensor(obs).to(self.device)
        next_obs = model_util.to_tensor(next_obs).to(self.device)
        
        # Encode observations using VAE
        with torch.no_grad():
            # Reshape observations for VAE if needed
            if len(obs.shape) == 2:
                batch_size = obs.shape[0]
                sequence_length = self.vae.sequence_length
                remaining_features = obs.shape[1] // sequence_length
                obs = obs.view(batch_size, sequence_length, remaining_features)
                next_obs = next_obs.view(batch_size, sequence_length, remaining_features)
            
            _, obs_encoded, _ = self.vae(obs)
            _, next_obs_encoded, _ = self.vae(next_obs)
        
        if self.target_is_delta:
            target_obs = next_obs_encoded - obs_encoded
            for dim in self.no_delta_list:
                target_obs[..., dim] = next_obs_encoded[..., dim]
        else:
            target_obs = next_obs_encoded
            
        # Get model input
        model_in = self._get_model_input(obs_encoded, action)
        
        # Handle rewards
        if self.learned_rewards:
            reward = model_util.to_tensor(reward).to(self.device).unsqueeze(reward.ndim)
            target = torch.cat([target_obs, reward], dim=obs_encoded.ndim - 1)
        else:
            target = target_obs
            
        return model_in.float(), target.float()
    
    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """Samples next observations and rewards from the model.
        
        Works with VAE-encoded states while maintaining proper dimensions for rewards.
        """
        obs_encoded = model_util.to_tensor(model_state["obs"]).to(self.device)
        model_in = self._get_model_input(obs_encoded, act)
        
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "VAETransitionRewardModel requires wrapped model to define method sample_1d"
            )
            
        preds, next_model_state = self.model.sample_1d(
            model_in, model_state, rng=rng, deterministic=deterministic
        )
        
        # Split predictions into next state and reward
        next_obs_encoded = preds[:, :-1] if self.learned_rewards else preds
        
        if self.target_is_delta:
            tmp_ = next_obs_encoded + obs_encoded
            for dim in self.no_delta_list:
                tmp_[:, dim] = next_obs_encoded[:, dim]
            next_obs_encoded = tmp_
            
        rewards = preds[:, -1:] if self.learned_rewards else None
        next_model_state["obs"] = next_obs_encoded
        
        return next_obs_encoded, rewards, None, next_model_state
    
    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Reset the model state with initial encoded observation."""
        if not hasattr(self.model, "reset_1d"):
            raise RuntimeError(
                "VAETransitionRewardModel requires wrapped model to define method reset_1d"
            )
            
        obs = model_util.to_tensor(obs).to(self.device)
        
        # Encode initial observation
        with torch.no_grad():
            if len(obs.shape) == 2:
                batch_size = obs.shape[0]
                sequence_length = self.vae.sequence_length
                remaining_features = obs.shape[1] // sequence_length
                obs = obs.view(batch_size, sequence_length, remaining_features)
            _, obs_encoded, _ = self.vae(obs)
            
        model_state = {"obs": obs_encoded}
        model_state.update(self.model.reset_1d(obs_encoded, rng=rng))
        return model_state


class TradingEnvFunctions:
    """
    Wrapper class for trading environment functions that handles both raw and encoded states.
    Maintains separate state trackers for raw and encoded observations.
    """
    def __init__(
        self,
        vae: Optional[torch.nn.Module] = None,
        initial_margin: float = 1000.0,
        action_bounds: Tuple[float, float] = (-10.0, 10.0),
        sequence_length: int = 10
    ):
        self.vae = vae
        self.initial_margin = initial_margin
        self.action_bounds = action_bounds
        self.sequence_length = sequence_length
        
        # State tracking for raw observations
        self.raw_margin = torch.tensor(initial_margin)
        self.raw_positions = torch.tensor(0.0)
        
        # State tracking for encoded observations
        self.encoded_margin = torch.tensor(initial_margin)
        self.encoded_positions = torch.tensor(0.0)
        
    def decode_state(self, encoded_state: torch.Tensor) -> torch.Tensor:
        """Decodes the latent state back to observation space if VAE is present."""
        if self.vae is None:
            return encoded_state
        
        with torch.no_grad():
            decoded_state = self.vae.decode(encoded_state)
            # Reshape decoded state to match expected format
            batch_size = decoded_state.shape[0]
            return decoded_state.view(batch_size, -1)
    
    def make_termination_fn(self) -> Callable:
        """Creates a termination function that works with both raw and encoded states."""
        def termination_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
            # Determine if we're working with encoded states
            is_encoded = self.vae is not None and next_obs.shape[-1] != self.sequence_length - 1
            
            # Use appropriate state tracking
            margin = self.encoded_margin if is_encoded else self.raw_margin
            
            # Move to correct device if needed
            if margin.device != act.device:
                margin = margin.to(act.device)
                if is_encoded:
                    self.encoded_margin = margin
                else:
                    self.raw_margin = margin
            
            # Expand margin if needed
            if margin.dim() == 0 and act.dim() == 2:
                margin = margin.expand(act.shape[0])
            
            # Get returns from either encoded or raw state
            if is_encoded:
                decoded_state = self.decode_state(next_obs)
                latest_returns = decoded_state[:, -1]
            else:
                latest_returns = next_obs[:, -1]
            
            # Calculate PnL and update margin
            positions = act[:, -1]
            pnl = positions * latest_returns
            margin = margin + pnl
            
            # Update tracked margin
            if is_encoded:
                self.encoded_margin = margin
            else:
                self.raw_margin = margin
            
            return margin < 0
        
        def reset():
            self.raw_margin = torch.tensor(self.initial_margin)
            self.encoded_margin = torch.tensor(self.initial_margin)
        
        termination_fn.reset = reset
        return termination_fn
    
    def make_reward_fn(self) -> Callable:
        """Creates a reward function that works with both raw and encoded states."""
        def reward_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
            # Determine if we're working with encoded states
            is_encoded = self.vae is not None and next_obs.shape[-1] != self.sequence_length - 1
            
            # Use appropriate state tracking
            positions = self.encoded_positions if is_encoded else self.raw_positions
            
            # Move to correct device if needed
            if positions.device != act.device:
                positions = positions.to(act.device)
                if is_encoded:
                    self.encoded_positions = positions
                else:
                    self.raw_positions = positions
            
            # Expand positions if needed
            if positions.dim() == 0 and act.dim() == 2:
                positions = positions.expand(act.shape[0])
            
            # Calculate new positions with clipping
            position_changes = act[:, 0]
            new_positions = positions + position_changes
            new_positions = torch.clamp(new_positions, self.action_bounds[0], self.action_bounds[1])
            
            # Get returns from either encoded or raw state
            if is_encoded:
                decoded_state = self.decode_state(next_obs)
                latest_returns = decoded_state[:, -1]
            else:
                latest_returns = next_obs[:, -1]
            
            # Calculate rewards using current position
            rewards = positions * latest_returns
            
            # Update positions for next step
            if is_encoded:
                self.encoded_positions = new_positions
            else:
                self.raw_positions = new_positions
            
            return rewards.unsqueeze(-1)
        
        def reset():
            self.raw_positions = torch.tensor(0.0)
            self.encoded_positions = torch.tensor(0.0)
        
        reward_fn.reset = reset
        return reward_fn

# Example usage
def create_trading_env_functions(
    vae: Optional[torch.nn.Module] = None,
    initial_margin: float = 1000.0,
    action_bounds: Tuple[float, float] = (-10.0, 10.0),
    sequence_length: int = 10
) -> Tuple[Callable, Callable]:
    """
    Creates trading environment functions that handle both raw and encoded states.
    
    Args:
        vae: Optional VAE model for state encoding/decoding
        initial_margin: Initial account margin
        action_bounds: Tuple of (min, max) allowed positions
        sequence_length: Length of observation sequence
        
    Returns:
        Tuple of (termination_fn, reward_fn)
    """
    env_fns = TradingEnvFunctions(vae, initial_margin, action_bounds, sequence_length)
    return env_fns.make_termination_fn(), env_fns.make_reward_fn()

def create_config_dicts(device: str, latent_dim: int, ensemble_size: int = 5,
                       trial_length: int = 200, num_trials: int = 10) -> Tuple[Dict, Dict]:
    """Creates the configuration dictionaries following the project structure."""
    
    cfg_dict = {
        "dynamics_model": {
            "_target_": "deeptrade.models.GaussianMLP",
            "device": device,
            "num_layers": 3,
            "ensemble_size": ensemble_size,
            "hid_size": 200,
            "in_size": latent_dim + 1,  # latent_dim + action_dim
            "out_size": latent_dim + 1,  # latent_dim + reward
            "deterministic": False,
            "propagation_method": "fixed_model",
            "activation_fn_cfg": {
                "_target_": "torch.nn.LeakyReLU",
                "negative_slope": 0.01
            }
        },
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True
        },
        "overrides": {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
    }
    
    agent_cfg_dict = {
        "_target_": "deeptrade.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 15,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        "optimizer_cfg": {
            "_target_": "deeptrade.planning.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.1,
            "population_size": 500,
            "alpha": 0.1,
            "device": device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "clipped_normal": False
        }
    }
    
    return omegaconf.OmegaConf.create(cfg_dict), omegaconf.OmegaConf.create(agent_cfg_dict)


def train_vae(vae: ConvVAE, replay_buffer: common_util.ReplayBuffer, 
              batch_size: int, num_epochs: int, device: str) -> List[float]:
    """Trains the VAE on the stored observations.
    
    Args:
        vae: The VAE model
        replay_buffer: Replay buffer containing observations
        batch_size: Training batch size
        num_epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        List of training losses per epoch
    """
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae_losses = []
    
    # Add debug prints at the start
    sample_batch = replay_buffer.sample(1)
    print(f"Original observation shape: {sample_batch.obs.shape}")
    print(f"VAE sequence length: {vae.sequence_length}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        for _ in range(len(replay_buffer) // batch_size):
            batch = replay_buffer.sample(batch_size)
            
            # Get observations and reshape for VAE
            batch_obs = torch.FloatTensor(batch.obs).to(device)
            
            # Observation shape is [batch_size, features]
            # Need to reshape to [batch_size, sequence_length, remaining_features]
            remaining_features = batch_obs.shape[1] // vae.sequence_length
            if remaining_features * vae.sequence_length != batch_obs.shape[1]:
                raise ValueError(f"Observation size {batch_obs.shape[1]} must be divisible by sequence length {vae.sequence_length}")
            
            batch_obs = batch_obs.view(batch_size, vae.sequence_length, remaining_features)
            
            vae_optimizer.zero_grad()
            total_loss, recon_loss, kld_loss = vae.loss(batch_obs)
            total_loss.backward()
            vae_optimizer.step()
            epoch_losses.append(total_loss.item())
        
        avg_epoch_loss = np.mean(epoch_losses)
        vae_losses.append(avg_epoch_loss)
        if epoch % 10 == 0:
            print(f"VAE Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
            
    return vae_losses


def encode_batch(vae: ConvVAE, obs: np.ndarray, sequence_length: int, device: str) -> np.ndarray:
    """Encodes a batch of observations using the VAE.
    
    Args:
        vae: The VAE model
        obs: Observations to encode [batch_size, features]
        sequence_length: Length of input sequence
        device: Device to use for encoding
        
    Returns:
        Encoded observations [batch_size, latent_dim]
    """
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).to(device)
        
        # Debug print
        print(f"Original obs shape: {obs_tensor.shape}")
        
        # Reshape for batch processing
        if len(obs_tensor.shape) == 2:
            batch_size = obs_tensor.shape[0]
            # For single feature per timestep, just reshape directly
            obs_tensor = obs_tensor.view(batch_size, sequence_length, 1)
            
            # Debug print
            print(f"Reshaped for VAE: {obs_tensor.shape}")
        
        # Get encoded states
        _, mu, _ = vae(obs_tensor)
        
        # Debug print
        print(f"Encoded shape: {mu.shape}")
        
        return mu.cpu().numpy()


def train_pets_with_vae_preprocessing(
    env_cfg: Dict,
    sequence_length: int,
    latent_dim: int,
    num_trials: int,
    trial_length: int,
    seed: Optional[int] = None
):
    """Main training loop for PETS with VAE state preprocessing."""
    
    # Set up environment and seeding
    env = gym.make("SingleInstrument-v0", **env_cfg)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if seed is not None:
        env.reset(seed=seed)
        rng = np.random.default_rng(seed=seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        rng = np.random.default_rng()
        generator = torch.Generator(device=device)
    
    # Create visualization objects
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
    ax_text = axs[0].text(300, 50, "")
    
    # Initialize metrics tracking
    train_losses = []
    val_scores = []
    
    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        val_scores.append(val_score.mean().item())
    
    # Create configurations
    cfg, agent_cfg = create_config_dicts(
        device, latent_dim, ensemble_size=5,
        trial_length=trial_length,
        num_trials=num_trials
    )
    
    # Create replay buffers - one for raw observations, one for encoded states
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    raw_replay_buffer = common_util.create_replay_buffer(
        cfg, obs_shape, act_shape, rng=rng
    )
    encoded_replay_buffer = common_util.create_replay_buffer(
        cfg, (latent_dim,), act_shape, rng=rng
    )
    
    # Initial exploration - collect raw observations
    print("Collecting initial experience...")
    common_util.rollout_agent_trajectories(
        env,
        trial_length * 2,  # Collect more initial data for VAE training
        planning.RandomAgent(env),
        {},
        replay_buffer=raw_replay_buffer,
        trial_length=trial_length
    )
        
    vae = ConvVAE(
        sequence_length=sequence_length,
        n_features=1,  # Features per timestep
        latent_dim=latent_dim,
        hidden_dim=64,
        beta=1.0
    ).to(device)
    
    vae_losses = train_vae(vae, raw_replay_buffer, batch_size=32, num_epochs=100, device=device)
    
    # Encode collected experience for PETS
    print("Encoding experience for PETS...")
    raw_data = raw_replay_buffer.get_all()
    encoded_obs = encode_batch(vae, raw_data.obs, sequence_length, device)
    encoded_next_obs = encode_batch(vae, raw_data.next_obs, sequence_length, device)
    encoded_replay_buffer.add_batch(
        encoded_obs,
        raw_data.act,
        encoded_next_obs,
        raw_data.rewards,
        raw_data.terminateds,
        raw_data.truncateds
    )
    
    cfg.dynamics_model.in_size = latent_dim + 1  # latent_dim + action_dim 
    raw_model = common_util.create_one_dim_tr_model(cfg, (latent_dim,), act_shape)
    
    # Create dynamics model
    dynamics_model = VAETransitionRewardModel(
        raw_model,
        vae=vae,
        target_is_delta=cfg.algorithm.target_is_delta,
        normalize=cfg.algorithm.normalize
    )
   
    # Create reward and termination fns
    # termination_fn = termination_fns.make_single_instrument_termination_fn()
    # reward_fn = reward_fns.make_single_instrument_reward_fn()

    termination_fn, reward_fn = create_trading_env_functions(vae)
    
    # Create model environment
    model_env = models.ModelEnv(
        env,
        dynamics_model,
        termination_fn=termination_fn,
        reward_fn=reward_fn,
        generator=generator
    )
    
    # Create agent
    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
    )
    
    # Create model trainer
    model_trainer = models.ModelTrainer(
        dynamics_model,
        optim_lr=1e-3,
        weight_decay=5e-5
    )
    
    def update_axes(_axs, _text, _trial, _steps_trial, _all_rewards, force_update=False):
        if not force_update and (_steps_trial % 10 != 0):
            return
        _axs[0].clear()
        _axs[0].plot(train_losses)
        _axs[0].set_ylabel("PETS Training Loss")
        _axs[1].clear()
        _axs[1].set_xlim([0, num_trials + .1])
        if _all_rewards:
            _axs[1].set_ylim([min(_all_rewards), max(_all_rewards)])
        _axs[1].set_xlabel("Trial")
        _axs[1].set_ylabel("Trial reward")
        _axs[1].plot(_all_rewards, 'bs-')
        _text.set_text(f"Trial {_trial + 1}: {_steps_trial} steps")
        display.display(plt.gcf())
        display.clear_output(wait=True)
    
    # PETS training loop
    print("Starting PETS training...")
    all_rewards = [0]
    
    for trial in range(num_trials):
        raw_obs, _ = env.reset()
        obs = encode_batch(vae, raw_obs[None], sequence_length, device)[0]  # Encode single observation
        agent.reset()
        
        terminated = False
        total_reward = 0.0
        steps_trial = 0
        
        while not terminated:
            # Train dynamics model at start of trial
            if steps_trial == 0:
                dynamics_model.update_normalizer(encoded_replay_buffer.get_all())
                
                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    encoded_replay_buffer,
                    cfg.overrides.model_batch_size,
                    cfg.overrides.validation_ratio,
                    ensemble_size=cfg.dynamics_model.ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False
                )
                
                model_trainer.train(
                    dataset_train,
                    dataset_val=dataset_val,
                    num_epochs=50,
                    patience=50,
                    callback=train_callback,
                    silent=True
                )
            
            # Get action from agent using encoded state
            action = agent.act(obs)
            
            # Step environment and encode next state
            next_raw_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = encode_batch(vae, next_raw_obs[None], sequence_length, device)[0]
            
            # Store both raw and encoded transitions
            raw_replay_buffer.add(raw_obs, action, next_raw_obs, reward, terminated, truncated)
            encoded_replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
            
            update_axes(axs, ax_text, trial, steps_trial, all_rewards)
            
            raw_obs = next_raw_obs
            obs = next_obs
            total_reward += reward
            steps_trial += 1
            
            if steps_trial == trial_length:
                terminated = True
        
        all_rewards.append(total_reward)
        
    update_axes(axs, ax_text, trial, steps_trial, all_rewards, force_update=True)
    
    # Final plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].plot(train_losses)
    ax[0].set_xlabel("Total training epochs")
    ax[0].set_ylabel("Training loss (avg. NLL)")
    ax[1].plot(val_scores) 
    ax[1].set_xlabel("Total training epochs")
    ax[1].set_ylabel("Validation score (avg. MSE)")
    plt.show()
    
    return vae, dynamics_model, agent, {
        'rewards': all_rewards,
        'train_losses': train_losses,
        'val_scores': val_scores,
        'vae_losses': vae_losses
    }

# Example usage
if __name__ == "__main__":
    window_size = 10
    
    env_config = {
        "price_gen_info": {
            "name": "GBM",
            "S0": np.array([100.0]),
            "mu": np.array([0.1]),
            "cov_matrix": np.array([0.2]),
            "n_steps": 1000
        },
        "starting_cash": 10000.0,
        "window": window_size
    }
    
    results = train_pets_with_vae_preprocessing(
        env_config,
        sequence_length=window_size-1,
        latent_dim=8,
        num_trials=50,
        trial_length=200,
        seed=42
    )
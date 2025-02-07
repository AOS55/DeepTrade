import numpy as np
import torch
import omegaconf
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display


import deeptrade.env
from deeptrade.env import reward_fns, termination_fns
from deeptrade.models import ModelEnv, GaussianMLP
from deeptrade.optimization import RandomAgent, TrajectoryOptimizerAgent
import deeptrade.optimization as planning
import deeptrade.models as models
import deeptrade.util.common as common_util
import deeptrade.models.util as model_util
from typing import Tuple, Optional


def create_trading_configs(device: str, ensemble_size: int = 5,
                         trial_length: int = 200, num_trials: int = 50) -> Tuple[omegaconf.DictConfig, omegaconf.DictConfig]:
    """Creates configuration dictionaries for the trading environment."""
    
    cfg_dict = {
        "dynamics_model": {
            "_target_": "deeptrade.models.GaussianMLP",
            "device": device,
            "num_layers": 3,
            "ensemble_size": ensemble_size,
            "hid_size": 200,
            "in_size": 10,  # window_size + action_dim
            "out_size": 10,  # window_size + reward
            "deterministic": False,
            "propagation_method": "fixed_model",
            "activation_fn_cfg": {
                "_target_": "torch.nn.LeakyReLU",
                "negative_slope": 0.01
            }
        },
        "algorithm": {
            "learned_rewards": True,  # Important: Set to True for trading
            "target_is_delta": True,
            "normalize": True,
            "normalize_double_precision": False
        },
        "overrides": {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
    }
    
    agent_cfg_dict = {
        "_target_": "deeptrade.optimization.TrajectoryOptimizerAgent",
        "planning_horizon": 15,
        "replan_freq": 1,
        "verbose": False,
        "optimizer_cfg": {
            "_target_": "deeptrade.optimization.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.1,
            "population_size": 500,
            "alpha": 0.1,
            "device": device,
            "return_mean_elites": True,
            "clipped_normal": False
        }
    }
    
    return omegaconf.OmegaConf.create(cfg_dict), omegaconf.OmegaConf.create(agent_cfg_dict)


def train_trading_agent(
    window_size: int = 10,
    trial_length: int = 200,
    num_trials: int = 50,
    ensemble_size: int = 5,
    starting_cash: float = 10000.0,
    position_limits: Tuple[float, float] = (-10.0, 10.0),
    seed: Optional[int] = None
):
    """Main training loop for PETS trading agent."""
    
    # Set up environment and devices
    env_config = {
        "price_gen_info": {
            "name": "GBM",
            "S0": np.array([100.0]),
            "mu": np.array([0.1]),
            "cov_matrix": np.array([0.2]),
            "n_steps": 1000
        },
        "starting_cash": starting_cash,
        "window": window_size
    }
    
    env = gym.make("SingleInstrument-v0", **env_config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if seed is not None:
        env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = torch.Generator(device=device)
    
    # Create reward and termination functions
    reward_fn = reward_fns.make_single_instrument_reward_fn(position_limits)
    term_fn = termination_fns.make_single_instrument_termination_fn(position_limits)
    
    # Create configurations with modified input size
    cfg, agent_cfg = create_trading_configs(
        device, ensemble_size=ensemble_size,
        trial_length=trial_length,
        num_trials=num_trials
    )
    # Update input size to match flattened observation dimension
    # cfg["dynamics_model"]["in_size"] = window_size + 1  # flattened window + action
    # cfg["dynamics_model"]["out_size"] = window_size + 1  # flattened window + reward
    
    # Set action bounds in agent config
    agent_cfg.action_lb = float(env.action_space.low[0])
    agent_cfg.action_ub = float(env.action_space.high[0])
    agent_cfg.optimizer_cfg.lower_bound = float(env.action_space.low[0])
    agent_cfg.optimizer_cfg.upper_bound = float(env.action_space.high[0])
    
    # Create replay buffer
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape)
    
    # Initial exploration
    print("Collecting initial experience...")
    common_util.rollout_agent_trajectories(
        env,
        trial_length * 2,  # Collect more initial data
        RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
        trial_length=trial_length
    )
    
    # Add debugging prints
    print("\nDEBUG INFO:")
    buffer_data = replay_buffer.get_all()
    print(f"Buffer obs shape: {buffer_data.obs.shape}")
    print(f"Buffer act shape: {buffer_data.act.shape}")
    print(f"Buffer next_obs shape: {buffer_data.next_obs.shape}")
    print(f"Buffer rewards shape: {buffer_data.rewards.shape}")
    print(f"First few observations:\n{buffer_data.obs[:3]}")
    print(f"First few actions:\n{buffer_data.act[:3]}")
    print(f"Model input size: {cfg.dynamics_model.in_size}")
    
    # Create dynamics model
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    print(f"\nModel device: {dynamics_model.device}")
    print(f"Normalizer shape: {dynamics_model.input_normalizer.mean.shape if dynamics_model.input_normalizer else 'No normalizer'}")
    
    # Prepare and reshape the entire dataset
    full_batch = replay_buffer.get_all()
    obs_data = full_batch.obs
    act_data = full_batch.act
    
    print(f"\nData for normalizer:")
    print(f"obs_data shape: {obs_data.shape}")
    print(f"act_data shape: {act_data.shape}")
    
    # Reshape observations to 2D if needed
    if obs_data.ndim == 1:
        print("Reshaping 1D observations...")
        obs_data = obs_data.reshape(1, -1)
        act_data = act_data.reshape(1, -1)
    elif obs_data.ndim == 3:
        print("Reshaping 3D observations...")
        obs_data = obs_data.reshape(obs_data.shape[0], -1)
    
    print(f"Final shapes before concatenation:")
    print(f"obs_data: {obs_data.shape}")
    print(f"act_data: {act_data.shape}")
    
    # Try to prepare the input for the normalizer
    try:
        model_input = np.concatenate([obs_data, act_data], axis=1)
        print(f"Concatenated input shape: {model_input.shape}")
        print(f"First row of model input: {model_input[0]}")
    except Exception as e:
        print(f"Error during concatenation: {str(e)}")
    
    # Create model environment
    model_env = models.ModelEnv(
        env,
        dynamics_model,
        term_fn,
        reward_fn,
        generator=generator
    )
    
    # Create planning agent
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
    
    # Training metrics
    train_losses = []
    val_scores = []
    all_rewards = [0]
    positions = []
    
    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        if val_score is not None:
            val_scores.append(val_score.mean().item())
    
    # Main PETS training loop
    print("Starting PETS training...")
    for trial in range(num_trials):
        obs, _ = env.reset()
        reward_fn.reset()
        agent.reset()
        
        terminated = False
        total_reward = 0.0
        steps_trial = 0
        
        while not terminated and steps_trial < trial_length:
            # Train dynamics model at start of trial
            if steps_trial == 0:
                # Update normalizer with properly shaped data
                model_input = np.concatenate([obs_data, act_data], axis=1)
                dynamics_model.input_normalizer.update_stats(model_input)
                
                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
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
            
            # Step environment and add to buffer
            next_obs, reward, terminated, truncated, _ = common_util.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )
            
            # Update state
            obs = next_obs
            total_reward += reward
            steps_trial += 1
            
            if steps_trial % 10 == 0:
                print(f"\rTrial {trial + 1}, Step {steps_trial}, Reward: {total_reward:.2f}", end="")
        
        all_rewards.append(total_reward)
        print(f"\nTrial {trial + 1}/{num_trials}, Total Reward: {total_reward:.2f}")
    
    # Training visualization
    fig, ax = plt.subplots(3, 1, figsize=(12, 15))
    ax[0].plot(train_losses)
    ax[0].set_xlabel("Training epochs")
    ax[0].set_ylabel("Training loss (NLL)")
    ax[0].set_title("Model Training Loss")
    
    ax[1].plot(val_scores)
    ax[1].set_xlabel("Training epochs")
    ax[1].set_ylabel("Validation MSE")
    ax[1].set_title("Model Validation Score")
    
    ax[2].plot(all_rewards, 'b.-')
    ax[2].set_xlabel("Trial")
    ax[2].set_ylabel("Total Reward")
    ax[2].set_title("Trading Performance")
    
    fig.tight_layout()
    fig.savefig("training_metrics.png")
    
    return dynamics_model, agent, {
        'rewards': all_rewards,
        'train_losses': train_losses,
        'val_scores': val_scores
    }

if __name__ == "__main__":
    model, agent, metrics = train_trading_agent(
        window_size=10,
        trial_length=200,
        num_trials=50,
        ensemble_size=5,
        starting_cash=10000.0,
        position_limits=(-10.0, 10.0),
        seed=42
    )
import gymnasium as gym
import numpy as np
import deeptrade.env
import torch
from deeptrade.models import GaussianMLP, OneDTransitionRewardModel, ModelTrainer, ModelEnv
from deeptrade.util.replay_buffer import ReplayBuffer
from deeptrade.planning import create_trajectory_optim_agent_for_model
from omegaconf import DictConfig


def collect_random_data(env, num_steps: int = 10000) -> ReplayBuffer:
    """Collect random transitions for training the dynamics model."""
    
    obs_shape = env.observation_space["returns"].shape
    act_shape = env.action_space.shape
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        capacity=num_steps,
        obs_shape=obs_shape,
        act_shape=act_shape
    )
    
    obs, _ = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()  # Random action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Store transition
        replay_buffer.add(
            obs["returns"],
            action, 
            next_obs["returns"],
            reward,
            terminated,
            truncated
        )
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    return replay_buffer


def create_dynamics_model(env, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Create a probabilistic ensemble model for PETS."""
    
    # Configuration for the ensemble model
    model_cfg = {
        "ensemble_size": 5,
        "hid_size": 200,
        "num_layers": 3,
        "deterministic": False,
        "propagation_method": "random_model",
        "learn_logvar_bounds": True
    }
    
    in_size = env.observation_space["returns"].shape[0] + env.action_space.shape[0]
    out_size = env.observation_space["returns"].shape[0]
    
    # Create probabilistic ensemble
    ensemble = GaussianMLP(
        in_size=in_size,
        out_size=out_size,
        device=device,
        **model_cfg
    )
    
    # Wrap with transition reward model
    dynamics_model = OneDTransitionRewardModel(
        ensemble,
        target_is_delta=True,
        normalize=True,
        learned_rewards=True
    )
    
    return dynamics_model


def train_dynamics_model(model, replay_buffer, num_epochs: int = 100):
    """Train the dynamics model."""
    trainer = ModelTrainer(
        model=model,
        optim_lr=1e-3,
        weight_decay=1e-5
    )
    
    # Create training dataset
    dataset_train = deeptrade.util.common.get_basic_buffer_iterators(
        replay_buffer,
        batch_size=64,
        val_ratio=0.2,
        ensemble_size=len(model),
        shuffle_each_epoch=True
    )[0]
    
    # Train model
    model.update_normalizer(replay_buffer.get_all())
    trainer.train(dataset_train, num_epochs=num_epochs)
    
    return model


def create_dynamics_model(env, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Create a probabilistic ensemble model for PETS."""
    
    # Configuration for the ensemble model
    model_cfg = {
        "ensemble_size": 5,
        "hid_size": 200,
        "num_layers": 3,
        "deterministic": False,
        "propagation_method": "random_model",
        "learn_logvar_bounds": True
    }
    
    in_size = env.observation_space["returns"].shape[0] + env.action_space.shape[0]
    out_size = env.observation_space["returns"].shape[0]
    
    # Create probabilistic ensemble
    ensemble = GaussianMLP(
        in_size=in_size,
        out_size=out_size,
        device=device,
        **model_cfg
    )
    
    # Wrap with transition reward model
    dynamics_model = OneDTransitionRewardModel(
        ensemble,
        target_is_delta=True,
        normalize=True,
        learned_rewards=True
    )
    
    return dynamics_model


def train_dynamics_model(model, replay_buffer, num_epochs: int = 100):
    """Train the dynamics model."""
    trainer = ModelTrainer(
        model=model,
        optim_lr=1e-3,
        weight_decay=1e-5
    )
    
    # Create training dataset
    dataset_train = deeptrade.util.common.get_basic_buffer_iterators(
        replay_buffer,
        batch_size=64,
        val_ratio=0.2,
        ensemble_size=len(model),
        shuffle_each_epoch=True
    )[0]
    
    # Train model
    model.update_normalizer(replay_buffer.get_all())
    trainer.train(dataset_train, num_epochs=num_epochs)
    
    return model


def create_pets_agent(model, env):
    """Create PETS agent with CEM trajectory optimization."""
    
    # Configuration for CEM optimizer
    optimizer_cfg = DictConfig({
        "_target_": "deeptrade.planning.CEMOptimizer",
        "num_iterations": 5,
        "elite_ratio": 0.1,
        "population_size": 400,
        "alpha": 0.1,
        "device": model.device,
        "return_mean_elites": True
    })
    
    # Create model environment
    model_env = ModelEnv(
        env=env,
        model=model,
        termination_fn=deeptrade.env.termination_fns.margin_call,
        reward_fn=deeptrade.env.reward_fns.single_instrument
    )
    
    # Create PETS agent
    agent = create_trajectory_optim_agent_for_model(
        model_env=model_env,
        agent_cfg=DictConfig({
            "_target_": "deeptrade.planning.TrajectoryOptimizerAgent",
            "planning_horizon": 30,
            "optimizer_cfg": optimizer_cfg,
            "action_lb": env.action_space.low,
            "action_ub": env.action_space.high,
            "verbose": False
        }),
        num_particles=20
    )
    
    return agent


def backtest(env, agent, episodes: int = 100):
    """Run backtest with trained agent."""
    returns = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            action = agent.act(obs["returns"])
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
            
        returns.append(episode_return)
        print(f"Episode {episode}: Return = {episode_return:.2f}")
        
    return returns

# Main training and backtesting workflow
def main():
    # Create environment
    price_gen_info = {
    "name": "GBM",  # Geometric Brownian Motion
    "S0": np.array([100, 150, 200]),  # Initial prices
    "mu": np.array([0.05, 0.07, 0.04]),  # Drift terms
    "cov_matrix": np.array([[1.0, 0.5, 0.3], 
                           [0.5, 1.0, 0.4], 
                           [0.3, 0.4, 1.0]]),  # Correlation matrix
    "dt": 1/252,  # Daily timesteps
    "n_steps": 1000  # Number of timesteps
    }

    env = gym.make('MultiInstrument-v0',
                n_instruments=3,
                starting_cash=10000.0,
                window=10,
                price_gen_info=price_gen_info)
    
    # Collect random data
    replay_buffer = collect_random_data(env, num_steps=50000)
    
    # Create and train dynamics model
    dynamics_model = create_dynamics_model(env)
    trained_model = train_dynamics_model(dynamics_model, replay_buffer)
    
    # Create PETS agent
    agent = create_pets_agent(trained_model, env)
    
    # Run backtest
    returns = backtest(env, agent)
    
    # Print results
    print(f"\nAverage return: {np.mean(returns):.2f}")
    print(f"Std deviation: {np.std(returns):.2f}")

if __name__ == "__main__":
    main()
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any, Union, Optional
import gymnasium as gym
from datetime import datetime
import wandb

from deeptrade.env import reward_fns, termination_fns
from deeptrade.models import ModelEnv, GaussianMLP
from deeptrade.optimization import RandomAgent, TrajectoryOptimizerAgent
import deeptrade.optimization as planning
import deeptrade.models as models
import deeptrade.util.common as common_util
from deeptrade.util import Logger
from deeptrade.models.time_series_processes import Sine

class ModelLearningWorkspace:
    def __init__(self, cfg):
        # Setup directory configs
        self.work_dir = Path.cwd()
        self.cfg = cfg

        # Seeding
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        # Device setup
        self.device = 'cuda:0' if torch.cuda.is_available() and cfg.device == "cuda" else 'cpu'
        self.generator = torch.Generator(device=self.device)
        if cfg.seed is not None:
            self.generator.manual_seed(cfg.seed)

        # Setup wandb if enabled
        if cfg.use_wandb:
            run_name = f"model_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project="deep_trade_planning",
                      name=run_name,
                      config=OmegaConf.to_container(cfg))

        # Create environment
        self.env = self._create_environment()
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        # Setup dynamics model and training components
        self.dynamics_model = self._create_dynamics_model()
        self.model_env = self._create_model_env()
        self.model_trainer = self._create_model_trainer()
        self.replay_buffer = self._create_replay_buffer()

        # Initialize metrics storage
        self.val_scores = []
        self.prediction_errors = []

    def _create_environment(self):
        """Create and return the trading environment."""
        env_dict = {
            'starting_cash': self.cfg.env.starting_cash,
            'window': self.cfg.env.window,
            'seed': self.cfg.seed,
        }

        # Use sine wave for deterministic testing
        prices_data = Sine(
            amp=np.array([1.0]),
            freq=np.array([1e-3])
        ).generate(dt=1.0, n_steps=1000)[0]
        env_dict["prices_data"] = prices_data

        return gym.make("SingleInstrument-v0", **env_dict)

    def _create_dynamics_model(self):
        """Create the dynamics model."""
        in_size = self.obs_shape[0] + self.act_shape[0]
        out_size = self.obs_shape[0]
        reward_fn = reward_fns.single_instrument_reward

        model = GaussianMLP(
            in_size=in_size,
            out_size=out_size,
            device=self.device,
            num_layers=self.cfg.dynamics_model.num_layers,
            ensemble_size=self.cfg.dynamics_model.ensemble_size,
            hid_size=self.cfg.dynamics_model.hid_size,
            deterministic=self.cfg.dynamics_model.deterministic,
            propagation_method=self.cfg.dynamics_model.propagation_method
        )

        if self.cfg.use_wandb:
            wandb.watch(model, log="all", log_freq=100)

        return models.OneDTransitionRewardModel(
            model,
            target_is_delta=self.cfg.algorithm.target_is_delta,
            normalize=self.cfg.algorithm.normalize,
            reward_fn=reward_fn,
        )

    def _create_model_env(self):
        """Create the model environment."""
        reward_fn = reward_fns.single_instrument_reward
        term_fn = termination_fns.make_single_instrument_termination_fn(self.cfg.position_limits)

        return ModelEnv(
            self.env,
            self.dynamics_model,
            term_fn,
            reward_fn,
            generator=self.generator
        )

    def _create_model_trainer(self):
        """Create the model trainer."""
        return models.ModelTrainer(
            self.dynamics_model,
            optim_lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay
        )

    def _create_replay_buffer(self):
            """Create and return the replay buffer using the existing utility function."""
            return common_util.create_replay_buffer(
                self.cfg,
                self.obs_shape,
                self.act_shape
            )

    def collect_data(self):
            """Collect random exploration data using existing utility function."""
            print("\nCollecting random exploration data...")

            random_agent = RandomAgent(self.env)

            common_util.rollout_agent_trajectories(
                self.env,
                steps_or_trials_to_collect=self.cfg.training.exploration_steps,
                agent=random_agent,
                agent_kwargs={},
                replay_buffer=self.replay_buffer,
                trial_length=self.cfg.trial_length
            )

            print(f"\nData collection complete:")
            print(f"Buffer size: {len(self.replay_buffer)}")

    def train_model(self):
        """Train the dynamics model."""
        print("\nTraining dynamics model...")

        # Update normalizer with collected data
        self.dynamics_model.update_normalizer(self.replay_buffer.get_all())

        # Create training and validation datasets
        dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
            self.replay_buffer,
            batch_size=self.cfg.training.batch_size,
            val_ratio=self.cfg.training.val_ratio,
            ensemble_size=self.cfg.dynamics_model.ensemble_size,
            shuffle_each_epoch=True,
            bootstrap_permutes=True
        )

        train_losses, val_losses = self.model_trainer.train(
            dataset_train,
            dataset_val,
            num_epochs=self.cfg.training.epochs
        )

        return train_losses, val_losses

    def evaluate_predictions(self, num_samples=100):
        """Evaluate model predictions on random samples."""
        print("\nEvaluating model predictions...")

        buffer_data = self.replay_buffer.get_all()
        indices = np.random.choice(len(buffer_data.obs), num_samples)

        position_errors = []
        price_errors = []
        reward_errors = []

        for idx in indices:
            obs = buffer_data.obs[idx:idx+1]
            action = buffer_data.act[idx:idx+1]
            next_obs = buffer_data.next_obs[idx:idx+1]
            actual_reward = reward_fns.single_instrument_reward(
                torch.tensor(action), torch.tensor(next_obs)
            )

            model_state = self.model_env.reset(obs)

            # Get model prediction
            pred_next_obs, pred_reward, _, _ = self.model_env.step(
                action,
                model_state,
                sample=False
            )

            # Split observation into position and price components
            position_error = np.abs(next_obs[0,0] - pred_next_obs[0,0]).item()
            price_error = np.abs(next_obs[0,1] - pred_next_obs[0,1]).item()

            # Calculate reward error using same reward function
            pred_reward_manual = reward_fns.single_instrument_reward(
                torch.tensor(action), torch.tensor(obs)
            )
            reward_error = np.abs(actual_reward - pred_reward_manual).item()

            position_errors.append(position_error)
            price_errors.append(price_error)
            reward_errors.append(reward_error)

        # Print detailed statistics
        print("\nPrediction Error Statistics:")
        print(f"Position Prediction - Mean: {np.mean(position_errors):.4f}, Std: {np.std(position_errors):.4f}")
        print(f"Price Prediction - Mean: {np.mean(price_errors):.4f}, Std: {np.std(price_errors):.4f}")
        print(f"Reward Prediction - Mean: {np.mean(reward_errors):.4f}, Std: {np.std(reward_errors):.4f}")

        # Add visualizations for position vs price errors
        plt.figure(figsize=(15, 10))

        plt.subplot(2,2,1)
        plt.hist(position_errors, bins=20, alpha=0.7, label='Position Errors')
        plt.xlabel('Error Magnitude')
        plt.ylabel('Frequency')
        plt.title('Position Prediction Errors')

        plt.subplot(2,2,2)
        plt.hist(price_errors, bins=20, alpha=0.7, label='Price Errors')
        plt.xlabel('Error Magnitude')
        plt.ylabel('Frequency')
        plt.title('Price Prediction Errors')

        plt.savefig(self.work_dir / "prediction_evaluation.png")
        plt.close()

    def visualize_training(self, train_losses, val_losses):
        """Visualize training metrics."""
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Training Progress')

        if self.cfg.use_wandb:
            wandb.log({"training_plot": wandb.Image(plt)})

        plt.savefig(self.work_dir / "model_training.png")
        plt.close()

    def run(self):
        """Main execution flow."""
        # Collect exploration data
        self.collect_data()

        # Train the model
        train_losses, val_losses = self.train_model()

        # Evaluate and visualize results
        self.evaluate_predictions(num_samples=1000)
        self.visualize_training(train_losses, val_losses)

@hydra.main(config_path="configs", config_name="planning")
def main(cfg):
    workspace = ModelLearningWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

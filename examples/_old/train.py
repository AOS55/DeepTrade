import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any, Union
from collections import defaultdict
import gymnasium as gym
from datetime import datetime
import wandb

import deeptrade.env
from deeptrade.env import reward_fns, termination_fns
from deeptrade.models import ModelEnv, GaussianMLP, BasicEnsemble
from deeptrade.optimization import RandomAgent, TrajectoryOptimizerAgent
import deeptrade.optimization as planning
import deeptrade.models as models
import deeptrade.util.common as common_util
from deeptrade.util import Logger
from deeptrade.models.time_series_processes import Sine

class TradingLogger(Logger):
    """Trading Logger class to implement portfolio specific metrics"""

    def __init__(self, log_dir: Union[str, Path], use_wandb: bool = False):
            super().__init__(log_dir, use_wandb=use_wandb)

            if use_wandb:
                    # Configure wandb to group metrics
                    wandb.define_metric("episode/*")
                    wandb.define_metric("portfolio/*")
                    wandb.define_metric("market/*")
                    wandb.define_metric("performance/*")
                    wandb.define_metric("risk_metrics/*")
                    wandb.define_metric("trading_metrics/*")

                    # Create custom chart displays
                    wandb.config.update({
                        "custom_charts": {
                            "portfolio_value_chart": {
                                "value": {"chart": "line", "x": "time", "y": "value"}
                            },
                            "position_chart": {
                                "position": {"chart": "line", "x": "time", "y": "position"}
                            },
                            "pnl_chart": {
                                "pnl": {"chart": "line", "x": "time", "y": "pnl"}
                            }
                        }
                    })

            # Register trading metrics group
            self.register_group(
                "trading_metrics",
                [
                    ("position", "POS", "float"),
                    ("margin", "MAR", "float"),
                    ("time", "T", "int"),
                    ("current_price", "P", "float"),
                    ("delta_price", "ΔP", "float"),
                    ("reward", "R", "float"),
                    ("returns", "RET", "float")
                ],
                dump_frequency=100
            )

            # Register model metrics group
            self.register_group(
                "model_metrics",
                [
                    ("train_loss", "TLOSS", "float"),
                    ("val_loss", "VLOSS", "float"),
                    ("model_error", "ERR", "float")
                ],
                dump_frequency=100
            )

            # Store episode data for visualization
            self.episode_data = defaultdict(list)

    def log_step(self, obs: np.ndarray, action: np.ndarray, reward: float, info: Dict[str, Any]):
        """Log data from a single environment step."""
        metrics = {
            'portfolio/position': info['position'],
            'portfolio/margin': info['margin'],
            'market/time': info['time'],
            'market/current_price': info['current_price'],
            'market/delta_price': info.get('delta_price', 0.0),
            'performance/reward': reward,
            'performance/returns': obs[-1] if len(obs) > 0 else 0.0
        }

        self.log_data("trading_metrics", metrics)

        # Store for visualization
        for key, value in metrics.items():
            self.episode_data[key].append(value)

    def visualize_episode(self, episode: int):
        """Create and log visualization for episode performance."""
        if not self.use_wandb:
            return

        # Calculate episode metrics
        episode_metrics = {
            'episode/number': episode,
            'episode/total_reward': sum(self.episode_data['performance/reward']),
            'episode/final_margin': self.episode_data['portfolio/margin'][-1],
            'episode/mean_position': np.mean(self.episode_data['portfolio/position']),

            'risk_metrics/sharpe_ratio': self._calculate_sharpe_ratio(),
            'risk_metrics/max_drawdown': self._calculate_max_drawdown(),
            'risk_metrics/volatility': np.std(self.episode_data['performance/returns']),

            'trading_metrics/position_changes': np.sum(np.abs(np.diff(self.episode_data['portfolio/position']))),
            'trading_metrics/avg_position_size': np.mean(np.abs(self.episode_data['portfolio/position'])),
            'trading_metrics/win_rate': np.mean(np.array(self.episode_data['performance/reward']) > 0),
        }

        # Create matplotlib figures for custom visualizations
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Portfolio Value
        ax1.plot(self.episode_data['market/time'], self.episode_data['portfolio/margin'], 'b-', label='Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True)
        ax1.legend()

        # Position vs Price
        ax2.plot(self.episode_data['market/time'], self.episode_data['portfolio/position'], 'g-', label='Position')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.episode_data['market/time'], self.episode_data['market/current_price'], 'r-', label='Price', alpha=0.6)
        ax2.set_title('Position vs Price')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Size', color='g')
        ax2_twin.set_ylabel('Price', color='r')
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)
        ax2.grid(True)

        # Cumulative Returns
        cumulative_returns = np.cumsum(self.episode_data['performance/reward'])
        ax3.plot(self.episode_data['market/time'], cumulative_returns, 'b-', label='Cumulative PnL')
        ax3.set_title('Cumulative Returns')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Cumulative PnL')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()

        # Log metrics and figures to wandb
        wandb.log({
            **episode_metrics,
            'charts/trading_overview': wandb.Image(fig),

            # Custom line plots using wandb.log
            'portfolio_value': wandb.Table(
                data=[[x, y] for x, y in zip(self.episode_data['market/time'], self.episode_data['portfolio/margin'])],
                columns=['time', 'value']
            ),
            'position_size': wandb.Table(
                data=[[x, y] for x, y in zip(self.episode_data['market/time'], self.episode_data['portfolio/position'])],
                columns=['time', 'position']
            ),
            'cumulative_pnl': wandb.Table(
                data=[[x, y] for x, y in zip(self.episode_data['market/time'], cumulative_returns)],
                columns=['time', 'pnl']
            )
        })

        plt.close(fig)
        self.episode_data.clear()

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from rewards."""
        rewards = np.array(self.episode_data['performance/reward'])
        if len(rewards) < 2:
            return 0.0
        return np.sqrt(252) * (np.mean(rewards) - risk_free_rate) / (np.std(rewards) + 1e-6)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from account margin."""
        margins = np.array(self.episode_data['portfolio/margin'])
        peak = np.maximum.accumulate(margins)
        drawdown = (margins - peak) / peak
        return float(np.min(drawdown))


class TradingWorkspace:
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

        # Setup logging
        if cfg.use_wandb:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project="deep_trade",
                       name=run_name,
                       save_code=True)
            wandb.config.update(OmegaConf.to_container(cfg))
        self.logger = TradingLogger(self.work_dir, use_wandb=cfg.use_wandb)

        self.env_dict = {
            'price_gen_info': {
                'name': cfg.env.price_gen_info.name,
                'S0': np.array([cfg.env.price_gen_info.S0]),
                'mu': np.array([cfg.env.price_gen_info.mu]),
                'cov_matrix': np.array([cfg.env.price_gen_info.cov_matrix]),
                'n_steps': cfg.env.price_gen_info.n_steps
            },
            'starting_cash': cfg.env.starting_cash,
            'window': cfg.env.window,
            "seed": cfg.seed,
        }

        # Temporily added to test functionality
        sine_prices_data = Sine(amp = np.array([1.0]), freq = np.array([1e-3])).generate(dt=1.0, n_steps=1000)[0]
        self.env_dict["prices_data"] = sine_prices_data

        # Create environment
        self.env = gym.make("SingleInstrument-v0", **self.env_dict)
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        # Create reward and termination functions
        self.reward_fn = reward_fns.single_instrument_reward
        self.term_fn = termination_fns.make_single_instrument_termination_fn(cfg.position_limits)

        # Setup dynamics model
        self.dynamics_model = self._create_dynamics_model()

        # Create model environment
        self.model_env = models.ModelEnv(
            self.env,
            self.dynamics_model,
            self.term_fn,
            self.reward_fn,
            generator=self.generator
        )

        # Setup planning agent
        self.agent = self._create_planning_agent()

        # Create model trainer
        self.model_trainer = models.ModelTrainer(
            self.dynamics_model,
            optim_lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            logger=self.logger
        )

        # Create replay buffer
        self.replay_buffer = common_util.create_replay_buffer(cfg, self.obs_shape, self.act_shape)  # type: ignore
        self.dynamics_model.update_normalizer(self.replay_buffer.get_all())

        # Training metrics
        self.train_losses = []
        self.val_scores = []
        self.all_rewards = [0.0]

    def _create_dynamics_model(self):
        """Creates and returns the dynamics model based on config."""
        in_size = self.obs_shape[0] + self.act_shape[0]  # type: ignore
        out_size = self.obs_shape[0]  # type: ignore

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
            normalize=self.cfg.algorithm.normalize
        )

    def _create_planning_agent(self):
        """Creates and returns the planning agent based on config."""
        agent_cfg = planning.core.complete_agent_cfg(self.model_env, self.cfg.agent)

        action_lb = [float(self.env.action_space.low[0])]  # type: ignore
        action_ub = [float(self.env.action_space.high[0])]  # type: ignore

        agent_cfg.optimizer_cfg.lower_bound = action_lb
        agent_cfg.optimizer_cfg.upper_bound = action_ub

        agent = planning.TrajectoryOptimizerAgent(
            optimizer_cfg=agent_cfg.optimizer_cfg,
            action_lb=action_lb,
            action_ub=action_ub,
            planning_horizon=agent_cfg.planning_horizon,
            replan_freq=agent_cfg.replan_freq,
            verbose=agent_cfg.verbose
        )

        def trajectory_eval_fn(initial_state, action_sequence):
            return self.model_env.evaluate_action_sequences(
                action_sequence,
                initial_state,
                num_particles=self.cfg.algorithm.num_particles
            )

        agent.set_trajectory_eval_fn(trajectory_eval_fn)
        return agent

    def train_callback(self, _model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        """Callback for tracking training metrics."""
        self.train_losses.append(tr_loss)
        if val_score is not None:
            self.val_scores.append(val_score.mean().item())

        if _epoch % 10 == 0:  # Every 10 epochs:
            print(f"\nEpoch {_epoch}")
            print(f"Training Loss: {tr_loss:.4f}")
            if val_score is not None:
                print(f"Validation Score: {val_score.mean().item():.4f}")

    def collect_initial_experience(self):
        """Collect initial random experience for the replay buffer."""

        print("Collecting initial experience...")
        common_util.rollout_agent_trajectories(
            self.env,
            self.cfg.trial_length * 2000,  # Collect 200 trials of experience
            RandomAgent(self.env),
            {},
            replay_buffer=self.replay_buffer,
            trial_length=self.cfg.trial_length
        )

    def debug_training(self):
        """Comprehensive debugging of the training process"""
        def debug_environment(env, n_steps=100):
            """Debug environment dynamics and rewards"""
            obs, info = env.reset()
            total_reward = 0

            history = {
                'observations': [obs],
                'actions': [],
                'rewards': [],
                'positions': [info['position']],
                'prices': [info['current_price']]
            }

            print("\nInitial State:")
            print(f"Observation shape: {obs.shape}")
            print(f"Initial observation: {obs}")
            print(f"Initial info: {info}")
            print("\nAction Space:")
            print(f"Shape: {env.action_space.shape}")
            print(f"Bounds: [{env.action_space.low}, {env.action_space.high}]")

            # Random action rollout
            print("\nExecuting random actions...")
            for i in range(n_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                history['observations'].append(obs)
                history['actions'].append(action)
                history['rewards'].append(reward)
                history['positions'].append(info['position'])
                history['prices'].append(info['current_price'])
                total_reward += reward

                if i < 5:  # Print first few steps in detail
                    print(f"\nStep {i}:")
                    print(f"Action taken: {action}")
                    print(f"New observation: {obs}")
                    print(f"Reward: {reward}")
                    print(f"Position: {info['position']}")
                    print(f"Price: {info['current_price']}")

            # Plot key metrics
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))

            # Price movement
            axes[0].plot(history['prices'])
            axes[0].set_title('Price Movement')
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Price')
            axes[0].grid(True)

            # Position history
            axes[1].plot(history['positions'])
            axes[1].set_title('Position History')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Position Size')
            axes[1].grid(True)

            # Cumulative reward
            axes[2].plot(np.cumsum(history['rewards']))
            axes[2].set_title('Cumulative Reward')
            axes[2].set_xlabel('Step')
            axes[2].set_ylabel('Cumulative Reward')
            axes[2].grid(True)

            fig.tight_layout()
            fig.savefig(f"environment_debug.png")

            # Print summary statistics
            print("\nEnvironment Summary:")
            print(f"Total steps: {n_steps}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Average reward per step: {total_reward/n_steps:.2f}")
            print(f"Final position: {history['positions'][-1]:.2f}")
            print(f"Price range: [{min(history['prices']):.2f}, {max(history['prices']):.2f}]")

            return history

        def debug_model_predictions(model_env, replay_buffer, n_samples=10):
            """Debug model predictions vs actual transitions"""
            print("\n=== Model Prediction Analysis ===")

            if len(replay_buffer) == 0:
                print("Replay buffer is empty! Cannot analyze predictions.")
                return

            buffer_data = replay_buffer.get_all()

            # Randomly sample transitions
            buffer_size = len(buffer_data.obs)
            indices = np.random.choice(buffer_size, min(n_samples, buffer_size))
            prediction_errors = []

            print(f"\nAnalyzing {len(indices)} samples from buffer of size {buffer_size}")
            print(f"Observation shape: {buffer_data.obs.shape}")
            print(f"Action shape: {buffer_data.act.shape}")

            for idx in indices:
                # Get data and ensure correct batch dimension
                obs = buffer_data.obs[idx:idx+1]  # Add batch dimension
                action = buffer_data.act[idx:idx+1]
                next_obs = buffer_data.next_obs[idx:idx+1]
                actual_reward = buffer_data.rewards[idx]

                # Reset model env with current observation
                print(f"\nSample {idx}:")
                print(f"Input obs shape: {obs.shape}, action shape: {action.shape}")

                model_state = model_env.reset(obs)

                # Get prediction
                pred_next_obs, pred_reward, done, next_state = model_env.step(
                    action,
                    model_state,
                    sample=False
                )

                # Calculate error if prediction exists
                if pred_next_obs is not None:
                    error = np.abs(next_obs - pred_next_obs).mean()
                    prediction_errors.append(error)

                    print("\nState Transition:")
                    print(f"Current State:   {obs[0]}")
                    print(f"Action:          {action[0]}")
                    print(f"Actual Next:     {next_obs[0]}")
                    print(f"Predicted Next:  {pred_next_obs[0]}")
                    print(f"Prediction Error: {error:.4f}")

                    if pred_reward is not None:
                        reward_error = np.abs(actual_reward - pred_reward[0])
                        print("\nReward Prediction:")
                        print(f"Actual Reward:    {actual_reward:.4f}")
                        print(f"Predicted Reward: {pred_reward.item():.4f}")
                        print(f"Reward Error:     {reward_error.item():.4f}")

                    # Visualize the prediction
                    plt.figure(figsize=(12, 4))

                    # State comparison
                    plt.subplot(121)
                    x = np.arange(len(next_obs[0]))
                    width = 0.35
                    plt.bar(x - width/2, next_obs[0], width, label='Actual')
                    plt.bar(x + width/2, pred_next_obs[0], width, label='Predicted')
                    plt.title('State Comparison')
                    plt.legend()

                    # Error by dimension
                    plt.subplot(122)
                    errors = np.abs(next_obs[0] - pred_next_obs[0])
                    plt.bar(x, errors)
                    plt.title('Prediction Error by Dimension')
                    plt.xlabel('State Dimension')
                    plt.ylabel('Absolute Error')

                    plt.tight_layout()
                    plt.savefig(f"{idx}-prediction_comparison.png")

                else:
                    print("Model returned None prediction")

            if prediction_errors:
                print("\nPrediction Error Statistics:")
                print(f"Mean Error: {np.mean(prediction_errors):.4f}")
                print(f"Std Error:  {np.std(prediction_errors):.4f}")
                print(f"Min Error:  {np.min(prediction_errors):.4f}")
                print(f"Max Error:  {np.max(prediction_errors):.4f}")

                # Plot error distribution
                plt.figure(figsize=(8, 4))
                plt.hist(prediction_errors, bins=20)
                plt.title('Distribution of Prediction Errors')
                plt.xlabel('Prediction Error')
                plt.ylabel('Count')
                plt.savefig("debug_prediction.png")

        def debug_planning(agent, model_env, initial_state):
            """Debug the planning/optimization process"""
            print("\n=== Planning Process Analysis ===")

            # Get planned trajectory
            print("Computing optimal action...")
            action = agent.act(initial_state)
            print(f"Chosen action: {action}")

            # If agent has optimization results
            if hasattr(agent, '_last_optimization_results'):
                results = agent._last_optimization_results
                print("\nOptimization Results:")
                print(f"Number of candidates evaluated: {len(results['returns'])}")
                print(f"Best expected return: {max(results['returns']):.4f}")
                print(f"Mean expected return: {np.mean(results['returns']):.4f}")

                # Plot distribution of returns
                plt.figure(figsize=(10, 5))
                plt.hist(results['returns'], bins=30)
                plt.title('Distribution of Expected Returns')
                plt.xlabel('Expected Return')
                plt.ylabel('Count')
                plt.savefig("debug_planning.png")

        print("\n====== Starting Debug Analysis ======")

        # 1. Check environment behavior
        print("\n=== Environment Check ===")
        env_history = debug_environment(self.env)

        # 2. Collect initial experience
        print("\n=== Collecting Initial Experience ===")
        self.collect_initial_experience()
        print(f"Replay buffer size: {len(self.replay_buffer)}")

        # 3. Train model briefly
        print("\n=== Initial Model Training ===")
        self.train_model()

        # 4. Check model predictions
        debug_model_predictions(self.model_env, self.replay_buffer)

        # 5. Check planning
        print("\n=== Planning Check ===")
        obs, _ = self.env.reset()
        debug_planning(self.agent, self.model_env, obs)

        # 6. Test full episode
        print("\n=== Testing Full Episode ===")
        obs, info = self.env.reset()
        total_reward = 0
        steps = 0

        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'positions': [],
            'prices': []
        }

        while steps < self.cfg.trial_length:
            action = self.agent.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['positions'].append(info['position'])
            episode_data['prices'].append(info['current_price'])

            # total_reward += reward
            obs = next_obs
            steps += 1

            if terminated or truncated:
                break

        # Plot episode results
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        axes[0].plot(episode_data['prices'])
        axes[0].set_title('Price Movement')
        axes[0].grid(True)

        axes[1].plot(episode_data['positions'])
        axes[1].set_title('Agent Positions')
        axes[1].grid(True)

        axes[2].plot(np.cumsum(episode_data['rewards']))
        axes[2].set_title('Cumulative Reward')
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        print("\n=== Episode Summary ===")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per step: {total_reward/steps:.2f}")
        print(f"Final position: {episode_data['positions'][-1]:.2f}")

        print("\n====== Debug Analysis Complete ======")


    def train_model(self):
        """Train the dynamics model."""
        self.dynamics_model.update_normalizer(self.replay_buffer.get_all())

        dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
            self.replay_buffer,
            batch_size=self.cfg.model_batch_size,
            val_ratio=self.cfg.validation_ratio,
            ensemble_size=self.cfg.dynamics_model.ensemble_size,
            shuffle_each_epoch=True,
            bootstrap_permutes=False
        )

        self.model_trainer.train(
            dataset_train,
            dataset_val=dataset_val,
            num_epochs=self.cfg.num_epochs,
            patience=self.cfg.patience,
            callback=self.train_callback,
            silent=not self.cfg.verbose
        )

    def train(self):

        """Main training loop for PETS."""
        self.collect_initial_experience()

        print("Starting PETS training...")
        for trial in range(self.cfg.num_trials):

            obs, info = self.env.reset()
            # self.reward_fn.reset()  # type: ignore TODO: Fix typing
            self.agent.reset()

            terminated = False
            steps_trial = 0
            train_freq = 200

            while not terminated and steps_trial < self.cfg.trial_length:
                if steps_trial % train_freq == 0:
                    self.train_model()

                action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, info = common_util.step_env_and_add_to_buffer(
                    self.env, obs, self.agent, {}, self.replay_buffer
                )

                self.logger.log_step(obs, action, reward, info)

                obs = next_obs
                steps_trial += 1

            self.logger.visualize_episode(trial)

    def plot_training_metrics(self):
        """Plot and save training metrics."""
        fig, ax = plt.subplots(3, 1, figsize=(12, 15))

        ax[0].plot(self.train_losses)
        ax[0].set_xlabel("Training epochs")
        ax[0].set_ylabel("Training loss (NLL)")
        ax[0].set_title("Model Training Loss")

        ax[1].plot(self.val_scores)
        ax[1].set_xlabel("Training epochs")
        ax[1].set_ylabel("Validation MSE")
        ax[1].set_title("Model Validation Score")

        ax[2].plot(self.all_rewards, 'b.-')
        ax[2].set_xlabel("Trial")
        ax[2].set_ylabel("Total Reward")
        ax[2].set_title("Trading Performance")

        fig.tight_layout()
        fig.savefig(self.work_dir / "training_metrics.png")

        if self.cfg.use_wandb:
            wandb.log({"training_metrics": wandb.Image(fig)})

@hydra.main(config_path="configs", config_name="trading")
def main(cfg):
    workspace = TradingWorkspace(cfg)
    workspace.debug_training()
    workspace.train()
    workspace.debug_training()

if __name__ == "__main__":
    main()

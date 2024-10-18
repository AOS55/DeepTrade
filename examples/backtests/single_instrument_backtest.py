from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

import deeptrade.constants
import deeptrade.env.termination_fns as term_fns
import deeptrade.util
import deeptrade.util.common as common_utils
from deeptrade import models, planning
from deeptrade.diagnostics import Visualizer
from deeptrade.env import reward_fns
from deeptrade.env.single_instrument import SingleInstrumentEnv

EVAL_LOG_FORMAT = deeptrade.constants.EVAL_LOG_FORMAT

class Workspace:

    def __init__(self, cfg):

        # Setup dir configs
        self.work_dir = Path.cwd()
        self.log_dir = cfg.log_dir
        self.cfg = cfg

        # Seeding
        np.random.seed(self.cfg.seed)

        # Device
        if self.cfg.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Logging
        self.use_wandb = self.cfg.use_wandb
        if self.use_wandb:
            wandb.init(project="deep_trade",
                       name="test")
            wandb.config.update(OmegaConf.to_container(cfg))

        self.logger = deeptrade.util.Logger(self.work_dir, use_wandb=self.use_wandb)
        self.logger.register_group(
            deeptrade.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

        # Get data
        if self.cfg.data_type == "random":
            self.x_data, self.y_data = self.create_data()
        else:
            print("Data type not supported")  # Space to load real asset instrument data

        # Create env
        self.env = SingleInstrumentEnv(self.y_data)
        self.env.reset(self.cfg.seed)

        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape
        self.reward_fn = reward_fns.single_instrument
        self.term_fn = term_fns.margin_call

        # Setup models
        in_size = self.obs_shape[0] + self.act_shape[0]
        out_size = self.obs_shape[0] + int(self.cfg.algorithm.learned_rewards)

        self.model = models.GaussianMLP(
            in_size=in_size,
            out_size=out_size,
            device=self.device,
            num_layers=self.cfg.dynamics_model.num_layers,
            ensemble_size=self.cfg.ensemble_size,
            hid_size=self.cfg.dynamics_model.hid_size,
            deterministic=self.cfg.dynamics_model.deterministic,
            propagation_method=self.cfg.dynamics_model.propagation_method,
            learn_logvar_bounds=self.cfg.dynamics_model.learn_logvar_bounds,
            activation_fn_cfg=self.cfg.dynamics_model.activation_fn_cfg,
        )

        self.dynamics_model = models.OneDTransitionRewardModel(
            self.model,
            target_is_delta=self.cfg.algorithm.target_is_delta,
            normalize=self.cfg.algorithm.normalize,
            normalize_double_precision=self.cfg.algorithm.get(
                "normalize_double_precision", False
            ),
            learned_rewards=self.cfg.algorithm.learned_rewards,
            no_delta_list=self.cfg.overrides.get("no_delta_list", None),
            num_elites=self.cfg.overrides.get("num_elites", None)
        )

        self.model_env = models.ModelEnv(
            self.env,
            model=self.dynamics_model,
            termination_fn=self.term_fn,
            reward_fn=self.reward_fn
        )

        self.model_trainer = models.ModelTrainer(
            self.dynamics_model,
            optim_lr=self.cfg.optim_lr,
            weight_decay=self.cfg.weight_decay,
            logger=self.logger

        )

        # Setup agent
        agent_cfg = planning.core.complete_agent_cfg(self.model_env, self.cfg.algorithm.agent)
        self.agent = planning.TrajectoryOptimizerAgent(
            optimizer_cfg=agent_cfg.optimizer_cfg,
            action_lb=agent_cfg.action_lb,
            action_ub=agent_cfg.action_ub,
            planning_horizon=agent_cfg.planning_horizon,
            replan_freq=agent_cfg.replan_freq,
            verbose=agent_cfg.verbose,
        )

        def trajectory_eval_fn(initial_state, action_sequence):
            return self.model_env.evaluate_action_sequences(
                action_sequence, initial_state, num_particles=self.cfg.algorithm.num_particles
            )

        self.agent.set_trajectory_eval_fn(trajectory_eval_fn)
        self.replay_buffer = common_utils.create_replay_buffer(self.cfg, self.obs_shape, self.act_shape)

    def create_data(self):

        x_data = np.linspace(0, self.cfg.n_days, self.cfg.n_days)
        y_data = [0]
        for day in range(1, self.cfg.n_days):
            y_data.append(y_data[-1] + np.random.normal(0, self.cfg.var))
        y_data = np.array(y_data)
        return x_data, y_data


    def pretrain(self):

        # Populate initial environment buffer
        common_utils.rollout_agent_trajectories(
            env=self.env,
            steps_or_trials_to_collect=self.cfg.trial_length,
            agent=planning.RandomAgent(self.env),
            agent_kwargs={},
            replay_buffer=self.replay_buffer,
            seed=self.cfg.seed
        )

        # Custom logging

        training_losses = []
        val_scores = []

        def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
            training_losses.append(tr_loss)
            val_scores.append(val_score.mean().item())

        # Train dynamics model alone
        self.dynamics_model.update_normalizer(self.replay_buffer.get_all())

        dataset_train, dataset_val = common_utils.get_basic_buffer_iterators(
            replay_buffer=self.replay_buffer,
            batch_size=self.cfg.model_batch_size,
            val_ratio=self.cfg.validation_ratio,
            ensemble_size=self.cfg.ensemble_size,
            shuffle_each_epoch=False,
            bootstrap_permutes=False
        )

        self.model_trainer.train(
            dataset_train,
            dataset_val,
            num_epochs=self.cfg.num_epochs,
            callback=train_callback
        )

        # TODO: Add wandb logging


        # Train
        train_steps = 0
        current_trial = 0
        max_total_reward = -np.inf

        while train_steps < self.cfg.max_train_steps:

            obs, _ = self.env.reset()
            self.agent.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps_trial = 0

            while not terminated and not truncated:

                if train_steps % self.cfg.algorithm.freq_train_model == 0:
                    common_utils.train_model_and_save_model_and_data(
                        model=self.dynamics_model,
                        model_trainer=self.model_trainer,
                        cfg=self.cfg.overrides,
                        replay_buffer=self.replay_buffer,
                        work_dir=self.work_dir,
                    )

                next_obs, reward, terminated, truncated, _ = common_utils.step_env_and_add_to_buffer(
                    self.env, obs, self.agent, {}, replay_buffer=self.replay_buffer
                )

                obs = next_obs
                total_reward += reward
                steps_trial += 1
                train_steps += 1

                if self.cfg.debug_mode:
                    print(f"env_steps: {train_steps}, total_reward: {total_reward}")

            current_trial += 1
            max_total_reward = max(max_total_reward, total_reward)

        self.eval()

        return max_total_reward

    def eval(self):
        """
        Evaluate trained agent on the environment
        """

        self.visualizer = Visualizer(
            lookahead=25,
            model_env=self.model_env,
            agent=self.agent,
            env=self.env,
            cfg=self.cfg,
            log_dir=self.work_dir,
        )

        self.visualizer.run(use_mpc=False)

        # eval_steps = 0
        # max_total_reward = -np.inf

        # for idep in range(self.cfg.eval_episodes):

        #     obs, _ = self.env.reset()
        #     self.agent.reset()
        #     terminated = False
        #     truncated = False
        #     total_reward = 0.0
        #     steps_trial = 0

        #     ep_obs = []
        #     pred_obs = []
        #     actions = []
        #     rewards = []

        #     while not terminated and not truncated:

        #         action = self.agent.act(obs)
        #         next_obs, reward, terminated, truncated, _ = self.env.step(action)



        #         # x = np.concat([obs, action])
        #         # x_tensor = torch.from_numpy(x).unsqueeze(1).float().to(self.device)
        #         # x_tensor = self.dynamics_model.input_normalizer.normalize(x_tensor)

        #         # with torch.no_grad():
        #         #     p_obs, p_logvar = self.dynamics_model.model(x_tensor)
        #         #     p_obs = p_obs.cpu().squeeze().numpy()
        #         #     p_logvar = p_logvar.cpu().squeeze().numpy()

        #         pred_obs.append(p_obs)
        #         # p_obs, p_reward = self.dynamics_model.eval(obs, action)

        #         # pred_obs.append(p_obs)
        #         actions.append(action)
        #         ep_obs.append(obs)
        #         rewards.append(reward)

        #         obs = next_obs
        #         total_reward += reward
        #         steps_trial += 1
        #         eval_steps += 1

        #     # actions = np.array(actions)
        #     # observations = np.array(ep_obs)
        #     # x = np.concat([observations, actions], axis=1)
        #     # x_tensor = torch.from_numpy(x).float().to(self.device)
        #     # x_tensor = self.dynamics_model.input_normalizer.normalize(x_tensor)

        #     # with torch.no_grad():
        #     #     pred_obs, pred_logvar = self.dynamics_model.model(x_tensor)
        #     #     pred_obs = pred_obs[..., 0]
        #     #     pred_logvar = pred_logvar[..., 0]

        #     fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        #     ax[0].plot(rewards, color='b')
        #     ax[1].plot(ep_obs, color='b')
        #     ax[1].plot(pred_obs, color='k')
        #     fig.savefig(f"eval_{idep}.pdf")

        #     max_total_reward = max(max_total_reward, total_reward)



@hydra.main(config_path='configs/.', config_name='single_instrument')
def main(cfg):
    # from examples.backtests.single_instrument_backtest import Workspace as W
    workspace = Workspace(cfg)
    workspace.pretrain()


if __name__=="__main__":
    main()

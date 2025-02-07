import pathlib
from collections.abc import Generator
from typing import List, Optional, Tuple, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from omegaconf import DictConfig

import deeptrade
import deeptrade.models
import deeptrade.optimization
import deeptrade.util.common

VisData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

class DataVisualizer:

    def __init__(
        self,
        lookahead: int,
        results_dir: str,
        agent_dir: Optional[str],
        num_steps: Optional[int] = None,
        num_model_samples: int = 1,
        model_subdir: Optional[str] = None,
    ):
        self.lookahead = lookahead
        self.results_path = pathlib.Path(results_dir)
        self.model_path = self.results_path
        self.vis_path = self.results_path / "diagnostics"
        if model_subdir:
            self.model_path /= model_subdir
            # If model subdir is child of diagnostics, remove "diagnostics" before
            # appending to vis_path. This can happen, for example, if Finetuner
            # generated this model with a model_subdir
            if "diagnostics" in model_subdir:
                model_subdir = pathlib.Path(model_subdir).name
            self.vis_path /= model_subdir
        pathlib.Path.mkdir(self.vis_path, parents=True, exist_ok=True)

        self.num_model_samples = num_model_samples
        self.num_steps = num_steps

        self.cfg = deeptrade.util.common.load_hydra_cfg(self.results_path)
        self.handler = deeptrade.util.create_handler(self.cfg)

        self.env, term_fn, reward_fn = self.handler.make_env(self.cfg)
        self.reward_fn = reward_fn

        self.dynamics_model = deeptrade.util.common.create_one_dim_tr_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=self.model_path,
        )
        self.model_env = deeptrade.models.ModelEnv(
            self.env,
            self.dynamics_model,
            term_fn,
            reward_fn,
            generator=torch.Generator(self.dynamics_model.device),
        )

        self.agent: deeptrade.optimization.Agent
        if agent_dir is None:
            self.agent = deeptrade.optimization.RandomAgent(self.env)
        else:
            agent_cfg = deeptrade.util.common.load_hydra_cfg(agent_dir)
            if (
                agent_cfg.algorithm.agent._target_
                == "deeptrade.optimization.TrajectoryOptimizerAgent"
            ):
                agent_cfg.algorithm.agent.planning_horizon = lookahead
                print(f"agent_cfg.algorithm.agent: {agent_cfg.algorithm.agent}")
                self.agent = deeptrade.optimization.create_trajectory_optim_agent_for_model(
                    self.model_env,
                    agent_cfg.algorithm.agent,
                    num_particles=agent_cfg.algorithm.num_particles,
                )
            else:
                self.agent = deeptrade.optimization.load_agent(agent_dir, self.env)

        self.fig = None
        self.axs: List[plt.Axes] = []
        self.lines: List[plt.Line2D] = []
        self.writer = animation.FFMpegWriter(
            fps=15, metadata=dict(artist="Me"), bitrate=1800
        )

        # The total reward obtained while building the visualizationn
        self.total_reward = 0

    def get_obs_rewards_and_actions(
        self, obs: np.ndarray, use_mpc: bool = False
    ) -> VisData:
        if use_mpc:
            # When using MPC, rollout model trajectories to see the controller actions
            model_obses, model_rewards, actions = deeptrade.util.common.rollout_model_env(
                self.model_env,
                obs,
                plan=None,
                agent=self.agent,
                num_samples=self.num_model_samples,
            )
            # Then evaluate in the environment
            real_obses, real_rewards, _ = self.handler.rollout_env(
                cast(gym.wrappers.TimeLimit, self.env),
                obs,
                self.lookahead,
                agent=None,
                plan=actions,
            )
        else:
            # When not using MPC, rollout the agent on the environment and get its actions
            real_obses, real_rewards, actions = self.handler.rollout_env(
                cast(gym.wrappers.TimeLimit, self.env),
                obs,
                self.lookahead,
                agent=self.agent,
            )
            # Then see what the model would predict for this
            model_obses, model_rewards, _ = deeptrade.util.common.rollout_model_env(
                self.model_env,
                obs,
                agent=None,
                plan=actions,
                num_samples=self.num_model_samples,
            )
        return real_obses, real_rewards, model_obses, model_rewards, actions

    def vis_rollout(self, use_mpc: bool = False) -> Generator:
        obs, _ = self.env.reset()
        terminated = False
        truncated = False
        i = 0
        while not terminated and not truncated:
            vis_data = self.get_obs_rewards_and_actions(obs, use_mpc=use_mpc)
            action = self.agent.act(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.total_reward += reward
            obs = next_obs
            i += 1
            if self.num_steps and i == self.num_steps:
                break

            yield vis_data

    def set_data_lines_idx(
        self,
        plot_idx: int,
        data_idx: int,
        real_data: np.ndarray,
        model_data: np.ndarray,
    ):
        def adjust_ylim(ax, array):
            ymin, ymax = ax.get_ylim()
            real_ymin = array.min() - 0.5 * np.abs(array.min())
            real_ymax = array.max() + 0.5 * np.abs(array.max())
            if real_ymin < ymin or real_ymax > ymax:
                self.axs[plot_idx].set_ylim(min(ymin, real_ymin), max(ymax, real_ymax))
                self.axs[plot_idx].figure.canvas.draw()

        def fix_array_len(array):
            if len(array) < self.lookahead + 1:
                new_array = np.ones((self.lookahead + 1,) + tuple(array.shape[1:]))
                new_array *= array[-1]
                new_array[: len(array)] = array
                return new_array
            return array

        x_data = range(self.lookahead + 1)
        if real_data.ndim == 1:
            real_data = real_data[:, None]
        if model_data.ndim == 2:
            model_data = model_data[:, :, None]
        real_data = fix_array_len(real_data)
        model_data = fix_array_len(model_data)
        adjust_ylim(self.axs[plot_idx], real_data[:, data_idx])
        adjust_ylim(self.axs[plot_idx], model_data.mean(1)[:, data_idx])
        self.lines[4 * plot_idx].set_data(x_data, real_data[:, data_idx])
        model_obs_mean = model_data[:, :, data_idx].mean(axis=1)
        model_obs_min = model_data[:, :, data_idx].min(axis=1)
        model_obs_max = model_data[:, :, data_idx].max(axis=1)
        self.lines[4 * plot_idx + 1].set_data(x_data, model_obs_mean)
        self.lines[4 * plot_idx + 2].set_data(x_data, model_obs_min)
        self.lines[4 * plot_idx + 3].set_data(x_data, model_obs_max)

    def plot_func(self, data: VisData):
        real_obses, real_rewards, model_obses, model_rewards, actions = data

        num_plots = len(real_obses[0]) + 1
        assert len(self.lines) == 4 * num_plots
        for i in range(num_plots - 1):
            self.set_data_lines_idx(i, i, real_obses, model_obses)
        self.set_data_lines_idx(num_plots - 1, 0, real_rewards, model_rewards)

        return self.lines

    def create_axes(self):
        num_plots = self.env.observation_space.shape[0] + 1
        num_cols = int(np.ceil(np.sqrt(num_plots)))
        num_rows = int(np.ceil(num_plots / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols)
        fig.text(
            0.5, 0.04, f"Time step (lookahead of {self.lookahead} steps)", ha="center"
        )
        fig.text(
            0.04,
            0.17,
            "Predictions (blue/red) and ground truth (black).",
            ha="center",
            rotation="vertical",
        )

        axs = axs.reshape(-1)
        lines = []
        for i, ax in enumerate(axs):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_xlim(0, self.lookahead)
            if i < num_plots:
                (real_line,) = ax.plot([], [], "k")
                (model_mean_line,) = ax.plot([], [], "r" if i == num_plots - 1 else "b")
                (model_ub_line,) = ax.plot(
                    [], [], "r" if i == num_plots - 1 else "b", linewidth=0.5
                )
                (model_lb_line,) = ax.plot(
                    [], [], "r" if i == num_plots - 1 else "b", linewidth=0.5
                )
                lines.append(real_line)
                lines.append(model_mean_line)
                lines.append(model_lb_line)
                lines.append(model_ub_line)

        self.fig = fig

        self.axs = axs
        self.lines = lines

    def run(self, use_mpc: bool):
        self.create_axes()
        ani = animation.FuncAnimation(
            self.fig,
            self.plot_func,
            frames=lambda: self.vis_rollout(use_mpc=use_mpc),
            blit=True,
            interval=100,
            save_count=self.num_steps,
            repeat=False,
        )
        save_path = self.vis_path / f"rollout_{type(self.agent).__name__}_policy.mp4"
        ani.save(save_path, writer=self.writer)
        print(f"Video saved at {save_path}.")
        print(f"Total rewards obtained was: {self.total_reward}.")


class Visualizer:
    def __init__(
        self,
        lookahead: int,
        model_env: deeptrade.models.ModelEnv,
        agent: deeptrade.optimization.Agent,
        env: gym.Env,
        cfg: DictConfig,
        log_dir: pathlib.PosixPath,
        num_steps: Optional[int] = None,
        num_model_samples: int = 1,
    ):
        self.lookahead = lookahead
        self.model_env = model_env
        self.agent = agent
        self.env = env
        self.log_dir = log_dir
        # self.vis_path = self.log_dir.joinpath("diagnostics")
        self.vis_path = self.log_dir

        self.num_model_samples = num_model_samples
        self.num_steps = num_steps

        self.cfg = cfg
        self.handler = deeptrade.util.create_handler(self.cfg)

        self.fig = None
        self.axs: List[plt.Axes] = []
        self.lines: List[plt.Line2D] = []
        self.writer = animation.FFMpegWriter(
            fps=15, metadata=dict(artist="Me"), bitrate=1800
        )

        # The total reward obtained while building the visualization
        self.total_reward = 0

    def get_obs_rewards_and_actions(
        self, obs: np.ndarray, use_mpc: bool = False
    ) -> VisData:
        if use_mpc:
            # When using MPC, rollout model trajectories to see the controller actions
            model_obses, model_rewards, actions = deeptrade.util.common.rollout_model_env(
                self.model_env,
                obs,
                plan=None,
                agent=self.agent,
                num_samples=self.num_model_samples,
            )
            # Then evaluate in the environment
            real_obses, real_rewards, _ = self.handler.rollout_env(
                cast(gym.wrappers.TimeLimit, self.env),
                obs,
                self.lookahead,
                agent=None,
                plan=actions,
            )
        else:
            # When not using MPC, rollout the agent on the environment and get its actions
            real_obses, real_rewards, actions = self.handler.rollout_env(
                cast(gym.wrappers.TimeLimit, self.env),
                obs,
                self.lookahead,
                agent=self.agent,
            )
            # Then see what the model would predict for this
            model_obses, model_rewards, _ = deeptrade.util.common.rollout_model_env(
                self.model_env,
                obs,
                agent=None,
                plan=actions,
                num_samples=self.num_model_samples,
            )
        return real_obses, real_rewards, model_obses, model_rewards, actions

    def vis_rollout(self, use_mpc: bool = False) -> Generator:
        obs, _ = self.env.reset()
        terminated = False
        truncated = False
        i = 0
        while not terminated and not truncated:
            vis_data = self.get_obs_rewards_and_actions(obs, use_mpc=use_mpc)
            action = self.agent.act(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.total_reward += reward
            obs = next_obs
            i += 1
            if self.num_steps and i == self.num_steps:
                break

            yield vis_data

    def set_data_lines_idx(
        self,
        plot_idx: int,
        data_idx: int,
        real_data: np.ndarray,
        model_data: np.ndarray,
    ):
        def adjust_ylim(ax, array):
            ymin, ymax = ax.get_ylim()
            real_ymin = array.min() - 0.5 * np.abs(array.min())
            real_ymax = array.max() + 0.5 * np.abs(array.max())
            if real_ymin < ymin or real_ymax > ymax:
                self.axs[plot_idx].set_ylim(min(ymin, real_ymin), max(ymax, real_ymax))
                self.axs[plot_idx].figure.canvas.draw()

        def fix_array_len(array):
            if len(array) < self.lookahead + 1:
                new_array = np.ones((self.lookahead + 1,) + tuple(array.shape[1:]))
                new_array *= array[-1]
                new_array[: len(array)] = array
                return new_array
            return array

        x_data = range(self.lookahead + 1)
        if real_data.ndim == 1:
            real_data = real_data[:, None]
        if model_data.ndim == 2:
            model_data = model_data[:, :, None]
        real_data = fix_array_len(real_data)
        model_data = fix_array_len(model_data)
        adjust_ylim(self.axs[plot_idx], real_data[:, data_idx])
        adjust_ylim(self.axs[plot_idx], model_data.mean(1)[:, data_idx])
        self.lines[4 * plot_idx].set_data(x_data, real_data[:, data_idx])
        model_obs_mean = model_data[:, :, data_idx].mean(axis=1)
        model_obs_min = model_data[:, :, data_idx].min(axis=1)
        model_obs_max = model_data[:, :, data_idx].max(axis=1)
        self.lines[4 * plot_idx + 1].set_data(x_data, model_obs_mean)
        self.lines[4 * plot_idx + 2].set_data(x_data, model_obs_min)
        self.lines[4 * plot_idx + 3].set_data(x_data, model_obs_max)

    def plot_func(self, data: VisData):
        real_obses, real_rewards, model_obses, model_rewards, actions = data

        num_plots = len(real_obses[0]) + 1
        assert len(self.lines) == 4 * num_plots
        for i in range(num_plots - 1):
            self.set_data_lines_idx(i, i, real_obses, model_obses)
        self.set_data_lines_idx(num_plots - 1, 0, real_rewards, model_rewards)

        return self.lines

    def create_axes(self):
        num_plots = self.env.observation_space.shape[0] + 1
        num_cols = int(np.ceil(np.sqrt(num_plots)))
        num_rows = int(np.ceil(num_plots / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols)
        fig.text(
            0.5, 0.04, f"Time step (lookahead of {self.lookahead} steps)", ha="center"
        )
        fig.text(
            0.04,
            0.17,
            "Predictions (blue/red) and ground truth (black).",
            ha="center",
            rotation="vertical",
        )

        axs = axs.reshape(-1)
        lines = []
        for i, ax in enumerate(axs):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_xlim(0, self.lookahead)
            if i < num_plots:
                (real_line,) = ax.plot([], [], "k")
                (model_mean_line,) = ax.plot([], [], "r" if i == num_plots - 1 else "b")
                (model_ub_line,) = ax.plot(
                    [], [], "r" if i == num_plots - 1 else "b", linewidth=0.5
                )
                (model_lb_line,) = ax.plot(
                    [], [], "r" if i == num_plots - 1 else "b", linewidth=0.5
                )
                lines.append(real_line)
                lines.append(model_mean_line)
                lines.append(model_lb_line)
                lines.append(model_ub_line)

        self.fig = fig

        self.axs = axs
        self.lines = lines

    def run(self, use_mpc: bool):
        self.create_axes()
        ani = animation.FuncAnimation(
            self.fig,
            self.plot_func,
            frames=lambda: self.vis_rollout(use_mpc=use_mpc),
            blit=True,
            interval=100,
            save_count=self.num_steps,
            repeat=False,
        )
        save_path = self.vis_path / f"rollout_{type(self.agent).__name__}_policy.mp4"
        ani.save(save_path, writer=self.writer)
        print(f"Video saved at {save_path}.")
        print(f"Total rewards obtained was: {self.total_reward}.")


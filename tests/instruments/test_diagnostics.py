import os
import pathlib
import tempfile

import gymnasium as gym
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

import deeptrade.env
import deeptrade.diagnostics as diagnostics
import deeptrade.planning as planning
import deeptrade.util.common

_REPO_DIR = pathlib.Path(os.getcwd())
_DIR = tempfile.TemporaryDirectory()
_HYDRA_DIR = pathlib.Path(_DIR.name) / ".hydra"
pathlib.Path.mkdir(_HYDRA_DIR)

# Environment information
_ENV_NAME = "SingleInstrument-v0"
_ENV = gym.make(_ENV_NAME)
_OBS_SHAPE = _ENV.observation_space.shape
_ACT_SHAPE = _ENV.action_space.shape
_CONF_DIR = pathlib.Path(_REPO_DIR) / "examples" / "backtests" / "configs"

# Creating config files
with open(
    _REPO_DIR / _CONF_DIR / "dynamics_model" / "gaussian_mlp_ensemble.yaml",
    "r",
) as f:
    _MODEL_CFG = yaml.safe_load(f)

_CFG_DICT = {
    "algorithm": {
        "learned_rewards": True,
        "target_is_delta": True,
        "normalize": True,
        "dataset_size": 128,
    },
    "dynamics_model": _MODEL_CFG,
    "overrides": {
        "env": f"gym___{_ENV_NAME}",
        "term_fn": "no_termination",
        "model_batch_size": 32,
        "validation_ratio": 0.1,
        "num_elites": 5,
        "cem_num_iters": 5,
        "cem_elite_ratio": 0.1,
        "cem_population_size": 500,
        "cem_alpha": 0.1,
        "cem_clipped_normal": False,
    },
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}

with open(_REPO_DIR / _CONF_DIR / "algorithm" / "pets.yaml", "r") as f:
    _PETS_ALGO_CFG = yaml.safe_load(f)
with open(_REPO_DIR / _CONF_DIR / "action_optimizer" / "cem.yaml", "r") as f:
    _CEM_CFG = yaml.safe_load(f)

_CFG_DICT["algorithm"].update(_PETS_ALGO_CFG)
_CFG_DICT["algorithm"]["learned_rewards"] = True
_CFG_DICT["algorithm"]["agent"]["verbose"] = False
_CFG_DICT["action_optimizer"] = _CEM_CFG
_CFG_DICT["seed"] = 0

_CFG = OmegaConf.create(_CFG_DICT)

# Create a model to train and run then save to directory
one_dim_model = deeptrade.util.common.create_one_dim_tr_model(_CFG, _OBS_SHAPE, _ACT_SHAPE)
one_dim_model.set_elite(range(_CFG["overrides"]["num_elites"]))
one_dim_model.save(_DIR.name)

# Create replay buffers and save to directory with some data
_CFG.dynamics_model.in_size = "???"
_CFG.dynamics_model.out_size = "???"
replay_buffer = deeptrade.util.common.create_replay_buffer(_CFG, _OBS_SHAPE, _ACT_SHAPE)
deeptrade.util.common.rollout_agent_trajectories(
    _ENV, 128, planning.RandomAgent(_ENV), {}, replay_buffer=replay_buffer
)

replay_buffer.save(_DIR.name)

def test_eval_on_dataset():
    with open(_HYDRA_DIR / "config.yaml", "w") as f:
        OmegaConf.save(_CFG, f)

    evaluator = diagnostics.DatasetEvaluator(_DIR.name, _DIR.name, _DIR.name)
    evaluator.run()

    files = os.listdir(_DIR.name)
    for idx in range(_OBS_SHAPE[0] + 1):
        assert f"pred_dim{idx}.png" in files
        assert f"pred_dim{idx}.png" in files

def test_visualizer():
    with open(_HYDRA_DIR / "config.yaml", "w") as f:
        OmegaConf.save(_CFG, f)

    # print(f"cfg: {_CFG}, model_dir: {_DIR.name}")
    
    visualizer = diagnostics.DataVisualizer(
        5, _DIR.name, agent_dir=_DIR.name, num_steps=5, num_model_samples=5
    )
    visualizer.run(use_mpc=False)

    files = os.listdir(pathlib.Path(_DIR.name) / "diagnostics")
    assert "rollout_TrajectoryOptimizerAgent_policy.mp4" in files

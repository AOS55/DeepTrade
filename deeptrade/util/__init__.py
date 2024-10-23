# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Union

import omegaconf

from .logger import Logger
from .replay_buffer import (
    ReplayBuffer,
    SequenceTransitionIterator,
    SequenceTransitionSampler,
    TransitionIterator,
)


def create_handler(cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig]):
    """Creates a new environment handler from its string description.
        This method expects the configuration, ``cfg``,
        to have the following attributes (some are optional):

            - If ``cfg.overrides.env_cfg`` is present, this method
            instantiates the environment using `hydra.utils.instantiate(env_cfg)`.
            Otherwise, it expects attribute ``cfg.overrides.env``, which should be a
            string description of the environment where valid options are:
                - "instrument" or "Instrument" for the InstrumentEnvHandler.

    Returns:
        (EnvHandler): A handler for the associated gym environment
    """
    cfg = omegaconf.OmegaConf.create(cfg)
    env_cfg = cfg.overrides.get("env_cfg", None)
    if env_cfg is None:
        return create_handler_from_str(cfg.overrides.env)

    target = cfg.overrides.env_cfg.get("_target_")
    if "instrument" in target:
        from deeptrade.util.instrument import InstrumentEnvHandler
        return InstrumentEnvHandler()
    else:
        raise NotImplementedError


def create_handler_from_str(env_name: str):
    """Creates a new environment handler from its string description.

    Args:
        env_name (str): the string description of the environment. Where valid options are:
          - "instrument" or "Instrument" for the InstrumentEnvHandler.

    Returns:
        (EnvHandler): A handler for the associated gym environment
    """
    print(f"env_name: {env_name}")
    if "SingleInstrument" in env_name:
      from deeptrade.util.instrument import InstrumentEnvHandler
      return InstrumentEnvHandler()
    if "MultiInstrument" in env_name:
        from deeptrade.util.instrument import MultiInstrumentEnvHandler
        return MultiInstrumentEnvHandler()
    else:
        raise NotImplementedError

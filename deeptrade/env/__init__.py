import gymnasium as gym

from . import reward_fns, termination_fns
from .agents.breakout import BreakoutAgent
from .agents.ewmac import EWMACAgent
from .agents.hold import HoldAgent
from .single_instrument import SingleInstrumentEnv
from .multi_instrument import MultiInstrumentEnv

gym.register(
    id='SingleInstrument-v0',
    entry_point='deeptrade.env:SingleInstrumentEnv',
)

gym.register(
    id='MultiInstrument-v0',
    entry_point='deeptrade.env:MultiInstrumentEnv',
)

from .single_instrument import SingleInstrumentEnv

from .agents.hold import HoldAgent
from .agents.ewmac import EWMACAgent
from .agents.breakout import BreakoutAgent

from . import reward_fns, termination_fns 
import gymnasium as gym

gym.register(
    id='SingleInstrument-v0',
    entry_point='deeptrade.env:SingleInstrumentEnv',
)
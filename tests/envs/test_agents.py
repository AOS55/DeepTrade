import gymnasium as gym
import numpy as np

from deeptrade.env import EWMACAgent, HoldAgent, BreakoutAgent


def _make_test_env():
    return gym.make("SingleInstrument-v0")

def _test_agent(env, agent):

    prices, info = env.reset()
    agent = agent()
    position = np.array([info['position']])
    for _ in range(100):
        action = agent.act(prices, position)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated:
            break

def test_agents():
    env = _make_test_env()
    agents = [EWMACAgent, BreakoutAgent]
    for agent in agents:
        _test_agent(env, agent)

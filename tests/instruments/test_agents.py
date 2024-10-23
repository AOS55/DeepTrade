import gymnasium as gym

from deeptrade.env import EWMACAgent, HoldAgent, BreakoutAgent


def _make_test_env():
    return gym.make("SingleInstrument-v0")

def _test_agent(env, agent):

    state, _ = env.reset()
    agent = agent(env=env)
    for _ in range(100):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated:
            break

def test_agents():
    env = _make_test_env()
    agents = [HoldAgent, EWMACAgent, BreakoutAgent]
    for agent in agents:
        _test_agent(env, agent)

<style>
img
{
    display:block; 
    float:none; 
    margin-left:auto;
    margin-right:auto;
    width:60%;
}
</style> 

# DeepTrade

Deeptrade is a backtesting system and library designed to test and evaluate machine learning based strategies.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Pytorch](https://pytorch.org) 1.9.0 or higher

We recommend using a [conda environment](https://docs.anaconda.com/miniconda/miniconda-install/) to manage dependencies. You can create a new environment with the following command:

```bash
conda create --name deeptrade-env python=3.10
conda activate deeptrade-env
```

### Installation

#### Standard Installation

```bash
pip install deeptrade
```

#### Development Installation

If you want to modify the library, clone the repository and setup a development environment:

```bash
git clone https://github.com/AOS55/deeptrade.git
pip install -e .
```

### Running Tests

To test the library, either run `pytest` at root or specify test directories from root with:

```bash
python -m pytest tests/core
python -m pytest tests/instruments
```

## Usage

The core idea of DeepTrade is to backtest machine learning trading strategies based on either synthetic or real data. Backtesting is split into 2 datasets, training data, available at the start of the theoretical trading period and backtest data used to evaluate the strategy which is where you started the strategy from. The following provides an overview of the basic components of the library, examples of various backtests are provided in the [notebooks](notebooks) directory.

The train-backtest split is shown below:

<img align="center" src="https://github.com/AOS55/DeepTrade/blob/assets/assets/Backtest-Split.svg" width="500" alt="Train/Backtest split">

The classical [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP) is used to model the trading environment. The environment is defined by the following components:

- **Environment**: the trading environment represents the world the agent interacts with, $p(s'|s, a)$. This is responsible for providing the agent with observations, rewards, and other information about the state of the environment. The environment is defined by the `gymnasium` interface. These include:
  - `SingleInstrument-v0`: A single instrument trading environment designed for a simple single asset portfolio.
  - `MultiInstrument-v0`: A multi-instrument trading environment designed to hold a multiple asset portfolio.

  Each of the trading environments have the following key components:
    -  **Market data**: either generated synthetically or from a real dataset. Data is queried at time $t$ which is updated by a size `period` each time around the env-agent loop.
    -  **Account**: represents the portfolio consisting of:
       -  `Margin`: the amount of cash available.
       -  `Positions`: the quantity of the asset held.

  The observation of the environment is a numpy array consisting of:
    - `returns`, $r_{t-\tau:t}$ from the asset price, usually log returns over `window` $\tau$.
    - `position`, position of the portfolio in the asset.
    - `margin`, the amount of cash available.

- **Agent**: The agent, $\pi(a|s)$, is the decision maker that interacts with the environment. The agent is responsible for selecting actions based on observations from the environment. Model Based RL (MBRL) agents are provided along with classical systematic trading strategies. These include:
  - **MBRL agents**
    - `PETS`: Probabilistic Ensemble Trajectory Sampling from [Chua et al. (2018)](https://arxiv.org/abs/1805.12114).
    - `MBPO`: :construction: Model Based Policy Optimization from [Janner et al. (2019)](https://arxiv.org/abs/1906.08253). :construction:
    - `Dreamer`: :construction: Dream to Control from [Hafner et al. (2019)](https://arxiv.org/abs/1912.01603). :construction:
  - **Systematic agents**
    - `HoldAgent`: A simple buy and hold strategy.
    - `EWMACAgent`: Exponential Weighted Moving Average Crossover, momentum based trend following.
    - `BreakoutAgent`: Breakout strategy, based on the high and low of the previous `n` periods.

The environment has the following key components:

<img align="center" src="https://github.com/AOS55/DeepTrade/blob/assets/assets/DeepTrade-Env.png" width="500" alt="Agent/Env loop">

### Environment

```python
import gymnasium as gym
import deeptrade.env

env = gym.make("SingleInstrument-v0")

obs, info = env.reset()
truncated, terminated = False, False
while not truncated or not terminated:
    action = env.action_space.sample()
    obs, reward, truncated, info = env.step(action)
    print(f"Reward: {reward}")
```

### Agent

## Contributing

## Citing

If you use this project in your research, please consider citing it with:
```bibtex
@misc{deeptrade,
  author = {DeepTrade},
  title = {DeepTrade: A Model Based Reinforcement Learning System for Trading},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com./AOS55/deeptrade}},
}
```

## Disclaimer

DeepTrade is for educational and research purposes and should is used for live trading entirely at your own risk.
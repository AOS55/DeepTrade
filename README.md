# DeepTrade

Deeptrade is a backtesting system and library designed to test and evaluate machine learning based strategies. It is 

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

<img align="center" src="https://github.com/AOS55/DeepTrade/blob/assets/assets/Backtest-Split.svg" width="400" alt="Train/Backtest split">

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
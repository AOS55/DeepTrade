import numpy as np
import matplotlib.pyplot as plt

from deeptrade.env import SingleInstrumentEnv

def make_data(n_days: int = 1000, var: float = 0.1) -> np.ndarray:
    x_data = np.linspace(0, n_days, n_days)
    y_data = [0]
    for day in range(1, n_days):
        y_data.append(y_data[-1] + np.random.normal(0, var))
    y_data = np.array(y_data)
    return x_data, y_data


def main():
    x_data, y_data = make_data(n_days=100, var=0.01)
    env = SingleInstrumentEnv(price_data=y_data)
    obs, info = env.reset()
    
    margin = []
    position = []
    price = []
    
    done = False
    while not done:
        action = 20
        obs, reward, truncated, terminated, info = env.step(action)
        margin.append(obs[-1])
        position.append(obs[-2])
        price.append(obs[-3])
        done = terminated or truncated
     
    fig, ax = plt.subplots(3)
    ax[0].plot(y_data[env.start_time:], label="price")
    ax[1].plot(margin, label="margin")
    ax[2].plot(position, label="position")
    ax[0].plot(price, label="price")
    
    plt.savefig("test_env.pdf")
    
    print(f"obs: {obs}, info: {info}")
    return

if __name__=="__main__":
    main()
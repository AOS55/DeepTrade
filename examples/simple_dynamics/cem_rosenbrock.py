import matplotlib.pyplot as plt
import numpy as np
import torch
from mbrl import planning


def rosenbrock_fn(x: np.array, a: np.float64 = 1.0, b: np.float64 = 100.0) -> np.float64:
    _, d = x.shape
    assert d % 2 == 0
    val = 0.0
    for idx in range(d//2):
        x_slice = x[:, 2*idx]
        y_slice = x[:, 2*idx+1]
        val += (a - x_slice)**2 + b*(y_slice - x_slice**2)**2
        return -val

def main():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    lower_bound = [-2.0, -2.0]
    upper_bound = [2.0, 2.0]
    n = 100
    x0 = np.linspace(lower_bound[0], upper_bound[0], n)
    x1 = np.linspace(lower_bound[1], upper_bound[1], n)
    x0s, x1s = np.meshgrid(x0, x1)
    x = np.stack([x0s.flatten(), x1s.flatten()]).transpose()
    z = rosenbrock_fn(x)
    z = z.reshape(n, n)
    ax.contour(x0s, x1s, z, levels=30)

    n_iters = 5
    max_values = np.zeros(n_iters)
    mean_values = np.zeros(n_iters)
    legend = []

    def plot_population(population, values, idx):
        mu_population = population.mean(axis=0)
        ax.plot(population[::10, 0], population[::10, 1], 'ok', alpha = (idx+1)/(n_iters+1)/2, markersize=4)
        ax.plot(mu_population[0], mu_population[1], 'ok', alpha=(idx+1)/(n_iters+1), markersize=10)
        max_values[idx] = values.max().item()
        mean_values[idx] = values.mean().item()

    cem = planning.CEMOptimizer(n_iters, 0.01, 1000, lower_bound, upper_bound, 0.1, torch.device('cpu'))
    best = cem.optimize(rosenbrock_fn, torch.zeros(2), callback=plot_population)
    best_value = rosenbrock_fn(best.reshape(1, -1))
    print(f"Best x: {best}")
    ax.plot(best[0], best[1], 'bs', label='best_solution')
    ax.plot(1, 1, 'r+', markersize=10, label='optimal_solution')
    plt.legend()
    plt.savefig('cem_rosenbrock.pdf')



if __name__=="__main__":
    main()

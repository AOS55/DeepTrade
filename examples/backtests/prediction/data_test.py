import matplotlib.pyplot as plt
import numpy as np
import torch

import deeptrade.util.common as common_utils
from deeptrade import models
from deeptrade.util import replay_buffer


def make_data(n_days: int = 10000, increment: int=1000, var: list=[0.1]):
    x_data = np.linspace(0, n_days, n_days)
    y_data = [0]
    idvar = 0
    for day in range(1, n_days):
        if day % increment == 0:
            idvar += 1
            if idvar >= len(var):
                idvar = 0
        y_data.append(y_data[-1] + np.random.normal(0, var[idvar]))
    y_data = np.array(y_data)
    return x_data, y_data

def train(x_data, y_data, lookahead: int = 500):

    # Train Params
    num_members = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_val = x_data[:-lookahead], x_data[-lookahead:]
    y_train, y_val = y_data[:-lookahead], y_data[-lookahead:]

    train_buffer = replay_buffer.ReplayBuffer(x_train.shape[0], (1,), (0,))
    val_buffer = replay_buffer.ReplayBuffer(x_val.shape[0], (1,), (0,))
    for idx in range(x_train.shape[0]):
        train_buffer.add(x_train[idx], 0, y_train[idx], 0, False)
    for idx in range(x_val.shape[0]):
        val_buffer.add(x_val[idx], 0, y_val[idx], 0, False)
    train_dataset, _ = common_utils.get_basic_buffer_iterators(
        train_buffer, 2048, 0, ensemble_size=num_members, shuffle_each_epoch=True)
    val_dataset, _ = common_utils.get_basic_buffer_iterators(
        val_buffer, 2048, 0, ensemble_size=1)

    ensemble = models.GaussianMLP(
        1,
        1,
        device,
        num_layers=3,
        hid_size=64,
        activation_fn_cfg={"_target_": "torch.nn.SiLU"},
        ensemble_size=num_members
    )

    wrapper = models.OneDTransitionRewardModel(ensemble, target_is_delta=False, normalize=True, learned_rewards=False)

    wrapper.update_normalizer(train_buffer.get_all())
    trainer = models.ModelTrainer(wrapper, optim_lr=0.001, weight_decay=5e-5)
    train_losses , val_losses = trainer.train(train_dataset, val_dataset, num_epochs=5000, patience=500)

    x_tensor = torch.from_numpy(x_data).unsqueeze(1).float().to(device)
    x_tensor = wrapper.input_normalizer.normalize(x_tensor)

    with torch.no_grad():
        y_pred, y_pred_logvar = ensemble(x_tensor)
        y_pred = y_pred[..., 0]
        y_pred_logvar = y_pred_logvar[..., 0]
    y_var_epi = y_pred.var(dim=0).cpu().numpy()
    y_var = y_pred_logvar.exp()
    y_pred = y_pred.mean(dim=0).cpu().numpy()
    y_var_ale = y_var.mean(dim=0).cpu().numpy()
    y_std = np.sqrt(y_var_epi + y_var_ale)
    plt.figure(figsize=(16, 8))
    plt.plot(x_data, y_data, 'k')
    plt.plot(x_train, y_train, '.', markersize=0.9)
    plt.plot(x_val, y_val, 'r', markersize=4)
    plt.plot(x_data, y_pred, 'b-', markersize=4)
    plt.fill_between(x_data, y_pred, y_pred + 2 * y_std, color='b', alpha=0.2)
    plt.fill_between(x_data, y_pred - 2 * y_std, y_pred, color='b', alpha=0.2)
    # plt.axis([-12, 12, -2.5, 2.5])
    plt.savefig('predictions.pdf')

    return y_var_epi[-10:].mean(), y_var_ale[-10:].mean(), np.array(train_losses)[-10:].mean(), np.array(val_losses)[-10:].mean()

def backtest(x_data, y_data, lookahead: int = 500):

    var_ale = []
    var_epi = []
    train_losses = []
    val_losses = []
    t = []

    for ids in range(2, (x_data.shape[0]//lookahead)-1):
        # print(ids, ids*lookahead)
        x = x_data[:ids*lookahead]
        y = y_data[:ids*lookahead]
        print(x.shape, y.shape)
        ale, epi, train_loss, val_loss = train(x, y, lookahead)
        t.append(ids*lookahead)
        var_ale.append(ale)
        var_epi.append(epi)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(16, 8))
    plt.plot(t, var_ale, 'r')
    plt.plot(t, var_epi, 'b')
    plt.savefig('variances.pdf')

    plt.figure(figsize=(16, 8))
    plt.plot(t, train_losses, 'r')
    plt.plot(t, val_losses, 'b')
    plt.savefig('losses.pdf')



def main():
    # for _ in range(10):
    #     x_data, y_data = make_data()
    #     plt.plot(x_data, y_data)
    x_data, y_data = make_data(1000)
    plt.plot(x_data, y_data)
    plt.savefig('data.pdf')
    backtest(x_data, y_data, 20)

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import mbrl.util.common as common_utils
import numpy as np
import pandas as pd
import torch
from mbrl import models
from mbrl.util import replay_buffer


def main():

    df = pd.read_csv("data/GBP.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # df = df.set_index("DATETIME")
    # df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

    # print(df.)
    df['delta'] = df['price'] - df['price'].shift(1)
    df = df.dropna()  # Drop the first row which will have NaN after shift
    np_data = df.to_numpy()
    n_data = np_data.shape[0]
    x_data = np.array([idx for idx in range(np_data.shape[0])])
    y_data = np_data[:, 1]
    train_val_ratio = 0.5
    # Split into train and test windows
    n_windows = 30  # Define the number of windows
    windows = []

    window_size = n_data // n_windows
    # print(x_data)

    for idx in range(n_windows):
        start = idx * window_size
        end = start + window_size
        windows.append((x_data[start:end], y_data[start:end]))

    train_window_idx = np.random.choice(list(range(n_windows)), size=int(n_windows*train_val_ratio), replace=False)
    val_window_idx = [idx for idx in range(n_windows) if idx not in train_window_idx]

    x_train, y_train = [], []
    for idx in train_window_idx:
        x_train.append(windows[idx][0])
        y_train.append(windows[idx][1])
    x_train = np.array(x_train).flatten()
    y_train = np.array(y_train).flatten()

    x_val, y_val = [], []
    for idx in val_window_idx:
        x_val.append(windows[idx][0])
        y_val.append(windows[idx][1])
    x_val = np.array(x_val).flatten()
    y_val = np.array(y_val).flatten()

    plt.figure(figsize=(16, 8))
    plt.plot(x_data, y_data, x_train, y_train, '.', x_val, y_val, 'o', markersize=4)
    plt.savefig('data.pdf')

    # Train model
    num_members = 5
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
    wrapper = models.OneDTransitionRewardModel(
        ensemble, target_is_delta=False, normalize=True, learned_rewards=False
    )

    wrapper.update_normalizer(train_buffer.get_all())
    trainer = models.ModelTrainer(wrapper, optim_lr=0.001, weight_decay=5e-5)
    train_losses, val_losses = trainer.train(train_dataset, val_dataset, num_epochs=5000, patience=500)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    ax[0].plot(train_losses)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("train loss (gaussian nll)")
    ax[1].plot(val_losses)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("val loss (mse)")
    plt.savefig('losses.pdf')

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
    plt.plot(x_data, y_data, 'r')
    plt.plot(x_train, y_train, '.', markersize=0.9)
    plt.plot(x_data, y_pred, 'b-', markersize=4)
    plt.fill_between(x_data, y_pred, y_pred + 2 * y_std, color='b', alpha=0.2)
    plt.fill_between(x_data, y_pred - 2 * y_std, y_pred, color='b', alpha=0.2)
    # plt.axis([-12, 12, -2.5, 2.5])
    plt.savefig('predictions.pdf')

if __name__=="__main__":
    main()

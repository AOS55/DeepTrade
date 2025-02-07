from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

import deeptrade.util.common as common_utils
from deeptrade import models
from deeptrade.util import replay_buffer


class Workspace:

    def __init__(self, cfg):

        # Setup dir configs
        self.work_dir = Path.cwd()
        self.log_dir = cfg.log_dir
        self.cfg = cfg

        # Seeding
        np.random.seed(self.cfg.seed)

        # Device
        if self.cfg.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"cfg: {type(cfg)}")
        # Logging
        if self.cfg.use_wandb:
            wandb.init(project="deep_trade",
                       name="test")
            wandb.config.update(OmegaConf.to_container(cfg))

        # Get data
        if self.cfg.data_type == "random":
            self.x_data, self.y_data = self.create_data()
        else:
            print("Data type not supported")  # Space to load real asset instrument data


    def create_data(self):

        x_data = np.linspace(0, self.cfg.n_days, self.cfg.n_days)
        y_data = [0]
        for day in range(1, self.cfg.n_days):
            y_data.append(y_data[-1] + np.random.normal(0, self.cfg.var))
        y_data = np.array(y_data)
        return x_data, y_data


    def run(self):

        for ids in range(2, (self.x_data.shape[0]//self.cfg.n_forecast)-1):
            x = self.x_data[:ids*self.cfg.n_forecast]
            y = self.y_data[:ids*self.cfg.n_forecast]
            x_train, x_val = x[:-self.cfg.n_forecast], x[-self.cfg.n_forecast:]
            y_train, y_val = y[:-self.cfg.n_forecast], y[-self.cfg.n_forecast:]
            train_buffer = replay_buffer.ReplayBuffer(x_train.shape[0], (1,), (0,))
            val_buffer = replay_buffer.ReplayBuffer(x_val.shape[0], (1,), (0,))
            for idx in range(x_train.shape[0]):
                train_buffer.add(x_train[idx], 0, y_train[idx], 0, False, False)
            for idx in range(x_val.shape[0]):
                val_buffer.add(x_val[idx], 0, y_val[idx], 0, False, False)
            train_dataset, _ = common_utils.get_basic_buffer_iterators(
                train_buffer, 2048, 0, ensemble_size=self.cfg.num_members, shuffle_each_epoch=False)
            val_dataset, _ = common_utils.get_basic_buffer_iterators(
                val_buffer, 2048, 0, ensemble_size=1)

            ensemble = models.GaussianMLP(
                1,
                1,
                self.device,
                num_layers=self.cfg.num_layers,
                hid_size=self.cfg.hid_size,
                activation_fn_cfg={"_target_": "torch.nn.SiLU"},
                ensemble_size=self.cfg.num_members
            )

            wrapper = models.OneDTransitionRewardModel(ensemble, target_is_delta=False, normalize=True, learned_rewards=False)

            wrapper.update_normalizer(train_buffer.get_all())
            trainer = models.ModelTrainer(wrapper, optim_lr=0.001, weight_decay=5e-5)
            train_losses , val_losses = trainer.train(train_dataset, val_dataset, num_epochs=self.cfg.num_epochs)

            x_tensor = torch.from_numpy(x).unsqueeze(1).float().to(self.device)
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
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            # ax.figure(figsize=(16, 8))
            ax.plot(x, y, 'k')
            ax.plot(x_train, y_train, '.', markersize=0.9)
            ax.plot(x_val, y_val, 'r', markersize=4)
            ax.plot(x, y_pred, 'b-', markersize=4)
            ax.fill_between(x, y_pred, y_pred + 2 * y_std, color='b', alpha=0.2)
            ax.fill_between(x, y_pred - 2 * y_std, y_pred, color='b', alpha=0.2)
            # plt.axis([-12, 12, -2.5, 2.5])

            wandb.log({"train/prediction": wandb.Image(fig)})

            if self.cfg.use_wandb:
                wandb.log({
                    # "train/train_losses": wandb.Plotly(train_losses),
                    # "train/val_losses": wandb.Plotly(val_losses),
                    "train/step": ids,
                    "train/y_var_epi": y_var_epi[-10:].mean(),
                    "train/y_var_ale": y_var_ale[-10:].mean(),
                    "train/train_losses": np.array(train_losses)[-10:].mean(),
                    "train/val_losses": np.array(val_losses)[-10:].mean()
                })


@hydra.main(config_path='configs/.', config_name='default', version_base="1.1")
def main(cfg):
    # from deeptrade.examples.classifier import Workspace as W
    workspace = Workspace(cfg)
    workspace.run()


if __name__=="__main__":
    main()

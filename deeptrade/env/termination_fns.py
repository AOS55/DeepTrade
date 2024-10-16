import math
import torch

def margin_call(actions: torch.Tensor, next_obs: torch.tensor) -> torch.Tensor:
    margin = next_obs[:, -1]
    done = margin < 0
    return done

def no_termination(actions: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    done = torch.Tensor([False]).repeat(len(next_obs)).bool().to(next_obs.device)
    done = done[:, None]
    return done

def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done
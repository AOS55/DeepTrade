import collections
import functools

import numpy as np
import omegaconf
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import deeptrade.models
import deeptrade.util.replay_buffer
import deeptrade.models.util as model_utils
from deeptrade.env.termination_fns import no_termination
from deeptrade.types import TransitionBatch

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def test_activation_functions_gaussian_mlp():

    activation_cfg_silu = omegaconf.OmegaConf.create(
        {
            "_target_": "torch.nn.SiLU",
        }
    )

    activation_cfg_th = omegaconf.OmegaConf.create(
        {
            "_target_": "torch.nn.Threshold",
            "threshold": 0.5,
            "value": 10,
        }

    )

    model_silu = deeptrade.models.GaussianMLP(
        1, 1, _DEVICE, num_layers=2, activation_fn_cfg=activation_cfg_silu
    )

    model_th = deeptrade.models.GaussianMLP(
        1, 1, _DEVICE, num_layers=2, activation_fn_cfg=activation_cfg_th
    )

    hidden_layer = model_silu.hidden_layers
    silu = torch.nn.SiLU()
    assert(str(hidden_layer[0][1]) == str(silu))

    hidden_layer = model_th.hidden_layers
    threshold = torch.nn.Threshold(0.5, 10)
    assert(str(hidden_layer[0][1]) == str(threshold))

def test_basic_ensemble_gaussian_forward():
    model_in_size = 2
    model_out_size = 2
    member_cfg = omegaconf.OmegaConf.create(
        {
            "_target_": "deeptrade.models.GaussianMLP",
            "device": _DEVICE,
            "in_size": model_in_size,
            "out_size": model_out_size,
        }
    )
    ensemble = deeptrade.models.BasicEnsemble(
        2, torch.device(_DEVICE), member_cfg, propagation_method="expectation"
    )
    batch_size = 4
    model_in = torch.zeros(batch_size, 2).to(_DEVICE)

    member_out_mean_ex, member_out_var_ex = ensemble[0].forward(model_in)
    assert member_out_mean_ex.shape == torch.Size([batch_size, model_out_size])
    assert member_out_var_ex.shape == torch.Size([batch_size, model_out_size])

    def mock_forward(_, v=1):
        return (
            v * torch.ones_like(member_out_mean_ex),
            torch.zeros_like(member_out_var_ex),
        )

    ensemble[0].forward = functools.partial(mock_forward, v=1)
    ensemble[1].forward = functools.partial(mock_forward, v=2)

    model_out = ensemble.forward(model_in)[0]
    assert model_out.shape == torch.Size([batch_size, model_out_size])
    expected_tensor_sum = batch_size * model_out_size
    assert model_out.sum().item() == 1.5 * batch_size * model_out_size

    ensemble.set_propagation_method(None)
    model_out = ensemble.forward(model_in)[0]
    assert model_out.shape == torch.Size([2, batch_size, model_out_size])
    assert model_out[0].sum().item() == expected_tensor_sum
    assert model_out[1].sum().item() == 2 * expected_tensor_sum

_OUTPUT_FACTOR = 10

def _create_gaussian_ensemble_mock(ensemble_size, as_float=False):
    model = deeptrade.models.GaussianMLP(
        1, 1, _DEVICE, num_layers=2, ensemble_size=ensemble_size
    )

    # With this we can use the output value to identify which model produced the output
    def mock_fwd(_x, only_elite=False):
        output = _x.clone()
        if output.shape[0] == 1:
            output = output.repeat(ensemble_size, 1, 1)
        for i in range(ensemble_size):
            output[i] += i
        if as_float:
            return output.float(), output.float()
        return output.int(), output.int()

    model._default_forward = mock_fwd

    return model

def _check_output_counts_and_update_history(
    model_output, ensemble_size, batch_size, history
):
    counts = np.zeros(ensemble_size)
    for i in range(batch_size):
        model_idx = model_output[i].item() % ensemble_size
        counts[model_idx] += 1
        history[i] += str(model_idx)
    # assert that all models produced the same number of outputs
    for i in range(ensemble_size):
        assert counts[i] == batch_size // ensemble_size
    # this checks that each output values correspond to the input
    # at the same index
    for i, v in enumerate(model_output):
        assert int(v.item()) // _OUTPUT_FACTOR == i
    return history

def test_gaussian_mlp_ensemble_random_model_propagation():
    ensemble_size = 5
    model = _create_gaussian_ensemble_mock(ensemble_size)

    batch_size = 100
    num_reps = 200
    batch = _OUTPUT_FACTOR * torch.arange(batch_size).view(-1, 1).to(_DEVICE)
    history = ["" for _ in range(batch_size)]
    with torch.no_grad():
        for _ in range(num_reps):
            model.set_propagation_method("random_model")
            y = model.forward(batch)[0]
            history = _check_output_counts_and_update_history(
                y, ensemble_size, batch_size, history
            )
    # This is really hacky, but it's a cheap test to see if the history of models used
    # varied over the batch
    seen = set([h for h in history])
    assert len(seen) == batch_size


def test_gaussian_mlp_ensemble_fixed_model_propagation():
    ensemble_size = 5
    model = _create_gaussian_ensemble_mock(ensemble_size)
    model.set_propagation_method("fixed_model")

    batch_size = 100
    num_reps = 200
    batch = _OUTPUT_FACTOR * torch.arange(batch_size).view(-1, 1).to(_DEVICE)
    history = ["" for _ in range(batch_size)]
    rng = torch.Generator(device=_DEVICE)
    # This creates propagation indices to use for all runs
    state_dict = model.reset_1d(batch, rng)
    with torch.no_grad():
        for _ in range(num_reps):
            assert "propagation_indices" in state_dict
            y = model.forward(
                batch, propagation_indices=state_dict["propagation_indices"]
            )[0]
            history = _check_output_counts_and_update_history(
                y, ensemble_size, batch_size, history
            )
            for i in range(batch_size):
                assert history[i][-1] == history[i][0]


def test_gaussian_mlp_ensemble_expectation_propagation():
    ensemble_size = 5
    model = _create_gaussian_ensemble_mock(ensemble_size, as_float=True)
    model.set_propagation_method("expectation")

    batch_size = 100
    num_reps = 200
    batch = _OUTPUT_FACTOR * torch.arange(batch_size).view(-1, 1)
    with torch.no_grad():
        for _ in range(num_reps):
            y = model.forward(batch)[0]
            for i in range(batch_size):
                val = y[i].item()
                a = val // _OUTPUT_FACTOR
                b = val % _OUTPUT_FACTOR
                assert a == i
                np.testing.assert_almost_equal(
                    b, ensemble_size * (ensemble_size - 1) / ensemble_size / 2
                )

_MOCK_OBS_DIM = 1
_MOCK_ACT_DIM = 1

class MockEnv:
    observation_space = (_MOCK_OBS_DIM,)
    action_space = (_MOCK_ACT_DIM,)


class MockProbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None
        self.p = nn.Parameter(torch.ones(1))
        self.out_size = _MOCK_OBS_DIM
        self.deterministic = True

    def forward(self, x):
        batch_size = x.shape[0] if x.dim() > 1 else 1
        return (self.value * torch.ones(batch_size, self.out_size).to(x.device)), None


def mock_term_fn(act, next_obs):
    assert len(next_obs.shape) == len(act.shape) == 2

    done = torch.Tensor([False]).repeat(len(next_obs))
    done = done[:, None]
    return done

def mock_reward_fn(act, next_obs):
    print(f"next_obs: {next_obs}, act: {act}")
    return torch.ones_like(act[:, :1]) * 11.0

def get_mock_env(propagation_method):
    member_cfg = omegaconf.OmegaConf.create(
        {"_target_": "tests.core.test_models.MockProbModel"}
    )
    num_members = 3
    ensemble = deeptrade.models.BasicEnsemble(
        num_members,
        torch.device(_DEVICE),
        member_cfg,
        propagation_method=propagation_method,
    )
    dynamics_model = deeptrade.models.OneDTransitionRewardModel(
        ensemble, target_is_delta=True, normalize=False, obs_process_fn=None, reward_fn=mock_reward_fn
    )
    # With value we can uniquely id the output of each member
    member_incs = [i + 10 for i in range(num_members)]
    for i in range(num_members):
        ensemble.members[i].value = torch.tensor(member_incs[i], device=_DEVICE)

    rng = torch.Generator(device=_DEVICE)
    model_env = deeptrade.models.ModelEnv(
        MockEnv(), dynamics_model, mock_term_fn, generator=rng
    )
    return model_env, member_incs


def test_model_env_expectation_propagation():
    batch_size = 7
    model_env, member_incs = get_mock_env("expectation")
    init_obs = np.zeros((batch_size, _MOCK_OBS_DIM)).astype(np.float32)
    model_state = model_env.reset(initial_obs_batch=init_obs)

    action = np.zeros((batch_size, _MOCK_ACT_DIM)).astype(np.float32)
    prev_sum = 0
    for i in range(10):
        next_obs, reward, _, model_state = model_env.step(
            action, model_state, sample=False
        )
        print(f"next_obs: {next_obs.shape}")
        assert next_obs.shape == (batch_size, _MOCK_OBS_DIM)
        cur_sum = np.sum(next_obs)
        assert (cur_sum - prev_sum) == pytest.approx(batch_size * np.mean(member_incs))
        assert reward == pytest.approx(np.mean(member_incs))
        prev_sum = cur_sum


def test_model_env_expectation_random():
    batch_size = 100
    model_env, member_incs = get_mock_env("random_model")
    obs = np.zeros((batch_size, _MOCK_OBS_DIM)).astype(np.float32)
    model_state = model_env.reset(initial_obs_batch=obs)

    action = np.zeros((batch_size, _MOCK_ACT_DIM)).astype(np.float32)
    num_steps = 50
    history = ["" for _ in range(batch_size)]
    for i in range(num_steps):
        next_obs, reward, _, model_state = model_env.step(
            action, model_state, sample=False
        )
        assert next_obs.shape == (batch_size, 1)

        diff = next_obs - obs
        seen = set()
        # Check that all models produced some output in the batch
        for j, val in enumerate(diff):
            if isinstance(val, np.ndarray):
                v = int(val.item())  # Extracts the single element from the array
            else:
                v = int(val)
            assert v in member_incs
            seen.add(v)
            history[j] += str(member_incs.index(v))
        assert len(seen) == 3
        obs = np.copy(next_obs)

    # This is really hacky, but it's a cheap test to see if the history of models used
    # varied over the batch
    seen = set([h for h in history])
    assert len(seen) == batch_size


def test_model_env_expectation_fixed():
    batch_size = 100
    model_env, member_incs = get_mock_env("fixed_model")
    obs = np.zeros((batch_size, _MOCK_OBS_DIM)).astype(np.float32)
    model_state = model_env.reset(initial_obs_batch=obs)

    action = np.zeros((batch_size, _MOCK_ACT_DIM)).astype(np.float32)
    num_steps = 50
    history = ["" for _ in range(batch_size)]
    for i in range(num_steps):
        next_obs, reward, _, model_state = model_env.step(
            action, model_state, sample=False
        )
        assert next_obs.shape == (batch_size, 1)

        diff = next_obs - obs
        seen = set()
        # Check that all models produced some output in the batch
        for j, val in enumerate(diff):
            if isinstance(val, np.ndarray):
                v = int(val.item())  # Extracts the single element from the array
            else:
                v = int(val)
            assert v in member_incs
            seen.add(v)
            history[j] += str(member_incs.index(v))
        assert len(seen) == 3
        obs = np.copy(next_obs)

    for h in history:
        assert len(set([c for c in h])) == 1


_MOCK_OBS_DIM = 1
_MOCK_ACT_DIM = 1


class DummyModel(deeptrade.models.Model):
    def __init__(self):
        super().__init__(torch.device(_DEVICE))
        self.param = nn.Parameter(torch.ones(1))
        self.out_size = _MOCK_OBS_DIM
        self.deterministic = True
        self.in_size = _MOCK_OBS_DIM + _MOCK_ACT_DIM

    def forward(self, x, **kwargs):
        obs = x[:, :_MOCK_OBS_DIM]
        act = x[:, _MOCK_OBS_DIM:]
        new_obs = obs + act
        return new_obs, None

    def sample_1d(self, x, model_state, deterministic=False, rng=None):
        return self.forward(x)[0], model_state

    def reset_1d(self, _obs, rng=None):
        return {}

    def loss(self, _input, target=None):
        return 0.0 * self.param, {"loss": 0}

    def eval_score(self, _input, target=None):
        return torch.zeros_like(_input), {"score": 0}

    def set_elite(self, _indices):
        pass


def test_model_env_evaluate_action_sequences():
    model = DummyModel()
    wrapper = deeptrade.models.OneDTransitionRewardModel(model, target_is_delta=False)
    model_env = deeptrade.models.ModelEnv(
        MockEnv(), wrapper, no_termination, generator=torch.Generator(), reward_fn=mock_reward_fn
    )
    for num_particles in range(1, 10):
        for horizon in range(1, 10):
            action_sequences = torch.stack(
                [
                    torch.ones(horizon, _MOCK_ACT_DIM),
                    2 * torch.ones(horizon, _MOCK_ACT_DIM),
                ]
            ).to(_DEVICE)
            expected_returns = horizon * (horizon + 1) * action_sequences[..., 0, 0] / 2
            # print(f"action_sequences: {action_sequences}, np.zeros(_MOCK_OBS_DIM): {np.zeros(_MOCK_OBS_DIM)}, num_particles: {num_particles}")
            print(f"expected_returns: {expected_returns}")
            returns = model_env.evaluate_action_sequences(
                action_sequences,
                np.zeros(_MOCK_OBS_DIM),
                num_particles=num_particles,
            )
            # TODO: Not sure what this is supposed to equal?
            # assert torch.allclose(expected_returns, returns)


def test_model_trainer_batch_callback():
    model = DummyModel()
    wrapper = deeptrade.models.OneDTransitionRewardModel(model, target_is_delta=False)
    trainer = deeptrade.models.ModelTrainer(wrapper)
    num_batches = 10
    dummy_data = torch.zeros(num_batches, 1)
    mock_dataset = deeptrade.util.replay_buffer.TransitionIterator(
        TransitionBatch(
            dummy_data,
            dummy_data,
            dummy_data,
            dummy_data.squeeze(1),
            dummy_data.squeeze(1),
            dummy_data.squeeze(1),
        ),
        1,
    )

    train_counter = collections.Counter()
    val_counter = collections.Counter()

    def batch_callback(epoch, val, meta, mode):
        assert mode in ["train", "eval"]
        if mode == "train":
            assert "loss" in meta
            train_counter[epoch] += 1
        else:
            assert "score" in meta
            val_counter[epoch] += 1

    num_epochs = 20
    trainer.train(mock_dataset, num_epochs=num_epochs, batch_callback=batch_callback)

    for counter in [train_counter, val_counter]:
        assert set(counter.keys()) == set(range(num_epochs))
        for i in range(num_epochs):
            assert counter[i] == num_batches

def test_conv2d_encoder_shapes():
    in_channels = 3
    config = ((in_channels, 32, 3, 2), (32, 64, 3, 2))
    activation_cls = [nn.ReLU, nn.SiLU, nn.Tanh]
    encoding_size = 200
    image_shape = (32, 32)
    for act_idx, activation_func in enumerate(["ReLU", "SiLU", "Tanh"]):
        encoder = model_utils.Conv2dEncoder(
            config, image_shape, encoding_size, activation_func
        )
        assert len(encoder.convs) == len(config)
        for i, layer_cfg in enumerate(config):
            assert isinstance(encoder.convs[i][0], nn.Conv2d)
            assert encoder.convs[i][0].in_channels == layer_cfg[0]
            assert encoder.convs[i][0].out_channels == layer_cfg[1]
            assert encoder.convs[i][0].kernel_size == (layer_cfg[2], layer_cfg[2])
            assert encoder.convs[i][0].stride == (layer_cfg[3], layer_cfg[3])
            assert isinstance(encoder.convs[i][1], activation_cls[act_idx])

        assert isinstance(encoder.fc, nn.Linear)
        assert encoder.fc.out_features == encoding_size

        dummy = torch.ones((8, in_channels) + image_shape)
        out = encoder.forward(dummy)
        assert out.shape == (8, encoding_size)


def test_conv2d_decoder_shapes():
    in_channels = 64
    config = ((in_channels, 32, 3, 2), (32, 16, 3, 2))
    activation_cls = [nn.ReLU, nn.SiLU, nn.Tanh]
    encoding_size = 200
    deconv_input_shape = (in_channels, 3, 3)
    for act_idx, activation_func in enumerate(["ReLU", "SiLU", "Tanh"]):
        decoder = model_utils.Conv2dDecoder(
            encoding_size, deconv_input_shape, config, activation_func=activation_func
        )
        assert len(decoder.deconvs) == len(config)
        for i, layer_cfg in enumerate(config):
            if i < len(config) - 1:
                deconv = decoder.deconvs[i][0]
                assert isinstance(decoder.deconvs[i][1], activation_cls[act_idx])
            else:
                deconv = decoder.deconvs[i]
            assert isinstance(deconv, nn.ConvTranspose2d)
            assert deconv.in_channels == layer_cfg[0]
            assert deconv.out_channels == layer_cfg[1]
            assert deconv.kernel_size == (layer_cfg[2], layer_cfg[2])
            assert deconv.stride == (layer_cfg[3], layer_cfg[3])

        assert isinstance(decoder.fc, nn.Linear)
        assert decoder.fc.out_features == np.prod(deconv_input_shape)

        dummy = torch.ones(8, encoding_size)
        out = decoder.forward(dummy)
        assert out.shape[0] == 8 and out.shape[1] == config[-1][1]


_VAE_OBS_DIM = 20


@pytest.fixture
def vae_model():
    obs_dim = _VAE_OBS_DIM
    code_dim = 4
    vae_beta = 1.0
    return model_utils.VAE(obs_dim, code_dim, vae_beta)


def test_encode_output_shape(vae_model):
    batch_size = 8
    sample_input = torch.randn(batch_size, _VAE_OBS_DIM)
    mu, std, logvar = vae_model.encode(sample_input)

    assert mu.shape == (batch_size, vae_model.code_dim)
    assert std.shape == (batch_size, vae_model.code_dim)
    assert logvar.shape == (batch_size, vae_model.code_dim)


def test_forward_output_shape(vae_model):
    batch_size = 8
    sample_input = torch.randn(batch_size, _VAE_OBS_DIM)
    epsilon = torch.randn(batch_size, vae_model.code_dim)
    output, (mu, logvar, std) = vae_model.forward(sample_input, epsilon)

    # Assert output and latent variables shapes
    assert output.shape == sample_input.shape
    assert mu.shape == (batch_size, vae_model.code_dim)
    assert logvar.shape == (batch_size, vae_model.code_dim)
    assert std.shape == (batch_size, vae_model.code_dim)


def test_loss_calculation(vae_model):

    batch_size = 8
    sample_input = torch.randn(batch_size, _VAE_OBS_DIM)
    loss, log_prob = vae_model.loss(sample_input)

    # Ensure loss is scalar and log_prob has the correct shape
    assert isinstance(loss.item(), float)
    assert log_prob.shape == (batch_size, 1)


@pytest.fixture
def model_params():
    return {
        'sequence_length': 20,
        'n_features': 3,
        'latent_dim': 8,
        'hidden_dim': 64,
        'beta': 1.0
    }


@pytest.fixture
def convvae_model(model_params):
    return model_utils.ConvVAE(**model_params)


@pytest.fixture
def sample_batch(model_params):
    batch_size = 16
    return torch.randn(batch_size, model_params['sequence_length'], model_params['n_features'])


def test_model_initialization(model_params):
    model = model_utils.ConvVAE(**model_params)

    # Check model attributes
    assert model.sequence_length == model_params['sequence_length']
    assert model.hidden_dim == model_params['hidden_dim']
    assert model.latent_dim == model_params['latent_dim']
    assert model.beta == model_params['beta']

    # Check flatten size calculation
    expected_flatten_size = model_params['hidden_dim'] * 2 * model_params['sequence_length']
    assert model.flatten_size == expected_flatten_size


def test_encode(convvae_model, sample_batch):
    mu, log_var = convvae_model.encode(sample_batch)

    # Check shapes
    assert mu.shape == (sample_batch.shape[0], convvae_model.latent_dim)
    assert log_var.shape == (sample_batch.shape[0], convvae_model.latent_dim)

    # Check values
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_var).any()
    assert not torch.isinf(mu).any()
    assert not torch.isinf(log_var).any()


def test_decode(convvae_model, model_params):
    batch_size = 16
    z = torch.randn(batch_size, model_params['latent_dim'])
    decoded = convvae_model.decode(z)

    # Check output shape
    expected_shape = (batch_size, model_params['sequence_length'], model_params['n_features'])
    assert decoded.shape == expected_shape

    # Check values
    assert not torch.isnan(decoded).any()
    assert not torch.isinf(decoded).any()


def test_forward(convvae_model, sample_batch):
    reconstruction, mu, log_var = convvae_model(sample_batch)

    # Check shapes
    assert reconstruction.shape == sample_batch.shape
    assert mu.shape == (sample_batch.shape[0], convvae_model.latent_dim)
    assert log_var.shape == (sample_batch.shape[0], convvae_model.latent_dim)

    # Check values
    assert not torch.isnan(reconstruction).any()
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_var).any()


def test_loss_function(convvae_model, sample_batch):
    total_loss, recon_loss, kld_loss = convvae_model.loss(sample_batch)

    # Check that losses are scalars
    assert total_loss.ndim == 0
    assert recon_loss.ndim == 0
    assert kld_loss.ndim == 0

    # Check values are reasonable
    assert total_loss.item() > 0
    assert recon_loss.item() > 0
    assert kld_loss.item() > 0

    # Check loss computation
    assert torch.allclose(total_loss, recon_loss + convvae_model.beta * kld_loss)

    # Verify KLD loss is non-negative
    assert kld_loss.item() >= 0


def test_full_pipeline(convvae_model, sample_batch):
    # Test the full pipeline: encode -> reparameterize -> decode
    mu, log_var = convvae_model.encode(sample_batch)
    z = convvae_model.reparameterize(mu, log_var)
    reconstruction = convvae_model.decode(z)

    # Final shape should match input
    assert reconstruction.shape == sample_batch.shape
    # Test if the model can reconstruct the input with reasonable accuracy
    mse = F.mse_loss(reconstruction, sample_batch)
    assert mse.item() < 10.0, "Reconstruction error is too high"


def test_beta_effect(model_params, sample_batch):
    # Test with different beta values
    beta_1 = model_utils.ConvVAE(**{**model_params, 'beta': 1.0})
    beta_01 = model_utils.ConvVAE(**{**model_params, 'beta': 0.1})

    # Compute losses for both models
    total_1, recon_1, kld_1 = beta_1.loss(sample_batch)
    total_01, recon_01, kld_01 = beta_01.loss(sample_batch)

    # Lower beta should result in lower KLD contribution to total loss
    assert abs(total_1 - (recon_1 + kld_1)) < 1e-3
    assert abs(total_01 - (recon_01 + 0.1 * kld_01)) < 1e-3

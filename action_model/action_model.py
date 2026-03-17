from action_model.models import DiT
import torch
from torch import nn
from torchdiffeq import odeint


def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)


def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


DiT_models = {"DiT-S": DiT_S, "DiT-B": DiT_B, "DiT-L": DiT_L}


class ActionModel(nn.Module):
    def __init__(
        self,
        token_size,
        model_type,
        in_channels,
        Ta=16,
        train_venc=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.net = DiT_models[model_type](
            token_size=token_size,
            in_channels=in_channels,
            inj_channels=32,
            class_dropout_prob=0.1,
            obs_dropout_prob=0 if train_venc else 1,
            future_action_window_size=Ta - 1,
            use_per_attn=True if train_venc else False,
        )
        self.train_venc = train_venc

    def loss(self, x1, z, obs):
        x0 = torch.randn_like(x1)
        t = sample_t_beta(x1.size(0), alpha=1.5, beta=1.0, device=x1.device, dtype=x1.dtype)
        t_broadcast = t.view(-1, *([1] * (x1.dim() - 1)))
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
        u_t = x1 - x0
        x = self.net(x_t, t, z, obs)
        act_loss = ((x - u_t) ** 2).mean()

        return act_loss

    @torch.no_grad()
    def inference(self, z, num_steps=10):
        assert not self.net.training, "Model should be in eval mode for inference."

        B = z.shape[0]
        Ta = self.net.future_action_window_size + 1
        x = torch.randn(B, Ta, self.in_channels, device=z.device, dtype=z.dtype)  # initial x_0

        t_steps = torch.linspace(0.0, 1.0, num_steps, device=z.device, dtype=z.dtype)

        for i in range(1, num_steps):
            t_prev = t_steps[i - 1]
            t_curr = t_steps[i]
            dt = t_curr - t_prev

            t_batch = torch.ones(B, device=z.device, dtype=z.dtype) * t_prev
            dx = self.net(x, t_batch, z)

            x = x + dx * dt

        return x


# for action sample
def sample_t_beta(batch_size, alpha=1.5, beta=1.0, eps=1e-3, device=None, dtype=None):
    t = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device=device, dtype=dtype)
    t = t * (1 - 2 * eps) + eps
    return t

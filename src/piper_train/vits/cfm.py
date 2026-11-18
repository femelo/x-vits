"""Optimal-transport conditional flow matching (OT-CFM) in the VITS latent space.

This replaces the normalizing flow (``ResidualCouplingBlock``) as the prior
model.  The estimator predicts the velocity field that transports Gaussian
noise to the posterior latent ``z`` (from ``PosteriorEncoder``), conditioned on
the text-derived, duration-expanded mean ``mu`` and the speaker embedding.

All tensors are ``[batch, channels, frames]`` and masks ``[batch, 1, frames]``.
"""

import math
from typing import Optional

import torch
from torch import nn

from . import modules


def sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1000.0):
    """t: [b] in [0, 1] -> [b, dim]"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = (t.float() * scale).unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class CFMEstimator(nn.Module):
    """Velocity-field estimator v(x_t, t | mu, g).

    A non-causal dilated WaveNet-style stack (reusing ``modules.WN``).  The time
    embedding (and speaker embedding, if any) enter through WN's global
    conditioning path, so they modulate every layer.  The output projection is
    zero-initialised so the untrained estimator predicts v = 0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        n_layers: int = 8,
        kernel_size: int = 5,
        dilation_rate: int = 2,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        # cond vector = time embedding (+ speaker embedding projected to same size)
        if gin_channels != 0:
            self.spk_proj = nn.Conv1d(gin_channels, hidden_channels, 1)

        self.pre = nn.Conv1d(2 * in_channels, hidden_channels, 1)
        self.wn = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=hidden_channels,
            p_dropout=p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, in_channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x_t, mask, mu, t, g=None):
        """x_t, mu: [b, c, T]; mask: [b, 1, T]; t: [b]; g: [b, gin, 1] or None"""
        cond = self.time_mlp(sinusoidal_embedding(t, self.hidden_channels))
        cond = cond.unsqueeze(-1)  # [b, h, 1]
        if g is not None:
            cond = cond + self.spk_proj(g)
        h = self.pre(torch.cat([x_t, mu], dim=1)) * mask
        h = self.wn(h, mask, g=cond)
        return self.proj(h) * mask


class OTCFM(nn.Module):
    """OT-CFM training loss and ODE sampler around a :class:`CFMEstimator`."""

    def __init__(self, estimator: CFMEstimator, sigma_min: float = 1e-4):
        super().__init__()
        self.estimator = estimator
        self.sigma_min = sigma_min

    def compute_loss(self, x1, mask, mu, g=None, t: Optional[torch.Tensor] = None):
        """Returns (loss, x_t). ``x1`` is the target latent (should be detached).

        Path: x_t = (1 - (1 - s) t) x0 + t x1,  target u = x1 - (1 - s) x0.
        """
        b = x1.size(0)
        if t is None:
            t = torch.rand(b, device=x1.device, dtype=x1.dtype)
        t_ = t.view(b, 1, 1)
        x0 = torch.randn_like(x1)
        x_t = (1 - (1 - self.sigma_min) * t_) * x0 + t_ * x1
        u = x1 - (1 - self.sigma_min) * x0
        v = self.estimator(x_t, mask, mu, t, g=g)
        # mean over valid elements
        loss = torch.sum(((v - u) ** 2) * mask) / (torch.sum(mask) * x1.size(1))
        return loss, x_t

    @torch.no_grad()
    def sample(
        self,
        mu,
        mask,
        g=None,
        n_steps: int = 4,
        temperature: float = 1.0,
        cosine_schedule: bool = True,
        noise: Optional[torch.Tensor] = None,
        solver: str = "euler",
    ):
        """ODE solve from noise (t=0) to data (t=1).

        ``solver``: euler (1 estimator call/step), midpoint | heun (2),
        rk4 = Simpson 1/3 (4), simpson38 = Simpson 3/8 rule (4).
        """
        x = noise if noise is not None else torch.randn_like(mu)
        x = x * temperature * mask
        t_span = torch.linspace(0, 1, n_steps + 1, device=mu.device, dtype=mu.dtype)
        if cosine_schedule:
            t_span = 1 - torch.cos(0.5 * math.pi * t_span)
        b = mu.size(0)

        def f(x_, t_):
            return self.estimator(x_, mask, mu, t_.expand(b), g=g) * mask

        for i in range(n_steps):
            t0, t1 = t_span[i], t_span[i + 1]
            dt = t1 - t0
            if solver == "euler":
                x = x + dt * f(x, t0)
            elif solver == "midpoint":
                k1 = f(x, t0)
                x = x + dt * f(x + 0.5 * dt * k1, t0 + 0.5 * dt)
            elif solver == "heun":
                k1 = f(x, t0)
                k2 = f(x + dt * k1, t1)
                x = x + 0.5 * dt * (k1 + k2)
            elif solver == "rk4":
                k1 = f(x, t0)
                k2 = f(x + 0.5 * dt * k1, t0 + 0.5 * dt)
                k3 = f(x + 0.5 * dt * k2, t0 + 0.5 * dt)
                k4 = f(x + dt * k3, t1)
                x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            elif solver == "simpson38":
                k1 = f(x, t0)
                k2 = f(x + dt / 3.0 * k1, t0 + dt / 3.0)
                k3 = f(x + dt * (-k1 / 3.0 + k2), t0 + 2.0 * dt / 3.0)
                k4 = f(x + dt * (k1 - k2 + k3), t1)
                x = x + dt / 8.0 * (k1 + 3 * k2 + 3 * k3 + k4)
            else:
                raise ValueError(f"unknown solver {solver!r}")
            x = x * mask
        return x

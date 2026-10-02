"""A learned hedger in the spirit of Buehler et al. (2019).

The network is given no Black-Scholes formula, no implied volatility and no
knowledge of the true volatility state. It observes only what a trader sees at
the decision time (moneyness, time, current position) and outputs the position
to hold, so any edge over the classical delta is learned rather than supplied.

Training minimises the mean of the worst tail of the profit-and-loss
distribution. A mean-variance objective would leave the tail unmanaged, and the
seller's problem is precisely the tail.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# A one-month at-the-money log-moneyness rarely leaves about +/- 0.1, so feeding
# it raw leaves the first layer reading near-identical inputs and slows the fit.
# Rescaling to roughly unit size is a constant, not model knowledge, and hides
# nothing from the comparison.
MONEYNESS_SCALE = 20.0


def _features(spot, j, dt, T, K, pos):
    """Observable state at one decision point.

    The current position is part of the state so the network can learn
    band-like behaviour, holding still when already close to the target. That is
    the same margin Sepp's hand-built bands chase, and leaving it out would make
    the comparison favour the classical strategy by construction.
    """
    log_moneyness = torch.log(spot / K) * MONEYNESS_SCALE
    tau = torch.full_like(log_moneyness, T - j * dt)
    return torch.stack([log_moneyness, tau / T, pos], dim=-1)


class HedgeNet(nn.Module):
    """Small feed-forward policy, applied identically at each time step."""

    def __init__(self, hidden: int = 64, depth: int = 2, in_dim: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ELU()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        # Start near an at-the-money delta of one half. A zero initial output
        # would spend early epochs rediscovering the sign and scale of the
        # exposure before any risk reduction could happen.
        last = layers[-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.constant_(last.bias, 0.5)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def hedge_pnl(paths, net, K, kind, premium, cost, dt, T):
    """Roll the network over paths and return terminal cash P&L.

    Gradients are kept through every step because the position today changes the
    state tomorrow; detaching the rollout would hide the cost of a trade from
    the loss that is supposed to discourage it.
    """
    n_steps = paths.shape[1] - 1
    pos = torch.zeros(paths.shape[0], device=paths.device, dtype=paths.dtype)
    cash = torch.full(
        (paths.shape[0],), premium, device=paths.device, dtype=paths.dtype
    )
    for j in range(n_steps):
        spot = paths[:, j]
        new_pos = net(_features(spot, j, dt, T, K, pos))
        traded = new_pos - pos
        cash = cash - traded * spot - cost * torch.abs(traded) * spot
        pos = new_pos

    terminal = paths[:, -1]
    if kind == "call":
        payoff = torch.clamp(terminal - K, min=0.0)
    elif kind == "put":
        payoff = torch.clamp(K - terminal, min=0.0)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    close_cost = cost * torch.abs(pos) * terminal
    return cash + pos * terminal - close_cost - payoff


def cvar_loss(pnl, alpha: float = 0.05):
    """Mean of the worst ``alpha`` fraction of outcomes.

    The finite-sample form needs no auxiliary optimisation variable, and its
    gradient flows only to the paths that breach the tail, which keeps training
    focused on the losses a seller actually cares about.
    """
    losses = -pnl
    k = max(1, int(alpha * losses.shape[0]))
    worst, _ = torch.topk(losses, k)
    return worst.mean()


def train_hedger(
    paths,
    K,
    kind,
    premium,
    cost,
    dt,
    T,
    *,
    epochs: int = 60,
    batch_size: int = 1024,
    lr: float = 1e-3,
    hidden: int = 64,
    depth: int = 2,
    alpha: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
):
    """Fit the hedger on a fixed path set.

    Paths are shuffled but never regenerated, so the only source of
    generalisation is the strategy the network learns rather than fresh noise
    every epoch. That keeps the in-sample and out-of-sample gap meaningful.
    """
    torch.manual_seed(seed)
    net = HedgeNet(hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = paths.shape[0]
    history: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pnl = hedge_pnl(paths[idx], net, K, kind, premium, cost, dt, T)
            loss = cvar_loss(pnl, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.detach())
            batches += 1
        history.append(epoch_loss / max(batches, 1))
        if verbose and (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:>3}  cvar {history[-1]: .6f}")
    return net, history


def evaluate_network(paths, net, K, kind, premium, cost, dt, T, batch_size=16384):
    """Forward-only P&L, batched so large evaluation sets stay in memory."""
    net.eval()
    out = np.empty(paths.shape[0])
    with torch.no_grad():
        for start in range(0, paths.shape[0], batch_size):
            chunk = paths[start : start + batch_size]
            pnl = hedge_pnl(chunk, net, K, kind, premium, cost, dt, T)
            out[start : start + batch_size] = pnl.detach().cpu().numpy()
    net.train()
    return out

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

# The default state, and the order the columns are stacked in. The set is a
# parameter rather than a literal so the ablation in scripts/explore_features.py
# can drop one observable at a time and measure what each contributes. Keeping
# the default tuple identical to the historical columns means every existing
# caller (bench, the replication scripts) is byte-for-byte unchanged.
DEFAULT_FEATURES = ("log_moneyness", "time", "pos")


def _features(spot, j, dt, T, K, pos, features=DEFAULT_FEATURES):
    """Observable state at one decision point.

    The current position is part of the state so the network can learn
    band-like behaviour, holding still when already close to the target. That is
    the same margin Sepp's hand-built bands chase, and leaving it out would make
    the comparison favour the classical strategy by construction.
    """
    parts = []
    if "log_moneyness" in features:
        parts.append(torch.log(spot / K) * MONEYNESS_SCALE)
    if "time" in features:
        tau = torch.full_like(spot, T - j * dt)
        parts.append(tau / T)
    if "pos" in features:
        parts.append(pos)
    return torch.stack(parts, dim=-1)


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


def hedge_pnl(paths, net, K, kind, premium, cost, dt, T, features=DEFAULT_FEATURES):
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
        new_pos = net(_features(spot, j, dt, T, K, pos, features))
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
    features=DEFAULT_FEATURES,
):
    """Fit the hedger on a fixed path set.

    Paths are shuffled but never regenerated, so the only source of
    generalisation is the strategy the network learns rather than fresh noise
    every epoch. That keeps the in-sample and out-of-sample gap meaningful.
    """
    torch.manual_seed(seed)
    net = HedgeNet(hidden=hidden, depth=depth, in_dim=len(features)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = paths.shape[0]
    history: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pnl = hedge_pnl(
                paths[idx], net, K, kind, premium, cost, dt, T, features
            )
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


def evaluate_network(
    paths, net, K, kind, premium, cost, dt, T, batch_size=16384,
    features=DEFAULT_FEATURES,
):
    """Forward-only P&L, batched so large evaluation sets stay in memory."""
    net.eval()
    out = np.empty(paths.shape[0])
    with torch.no_grad():
        for start in range(0, paths.shape[0], batch_size):
            chunk = paths[start : start + batch_size]
            pnl = hedge_pnl(chunk, net, K, kind, premium, cost, dt, T, features)
            out[start : start + batch_size] = pnl.detach().cpu().numpy()
    net.train()
    return out


def hedge_positions(paths, net, K, dt, T, features=DEFAULT_FEATURES):
    """The positions the policy holds, one column per decision point.

    Training only ever shows the network through its P&L, so the frictionless
    check -- does a cost-free hedger rediscover Black-Scholes delta? -- needs a
    way to read the positions directly. Inferring the answer from P&L alone
    would be weaker: several quite different policies produce nearly the same
    P&L when the premium is large relative to the hedging error.
    """
    net.eval()
    n_steps = paths.shape[1] - 1
    pos = torch.zeros(paths.shape[0], device=paths.device, dtype=paths.dtype)
    out = []
    with torch.no_grad():
        for j in range(n_steps):
            pos = net(_features(paths[:, j], j, dt, T, K, pos, features))
            out.append(pos.cpu().numpy())
    net.train()
    return np.stack(out, axis=1)

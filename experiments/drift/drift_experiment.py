"""Reconstructed support module for make_fig_drift.py, NOT the missing original.
Geometry and bound formulas follow exp_drift_v2.py.
SIGMAS/TRIALS/T follow that script; SPEEDS and BURN are explicit new defaults.
No experiments or file writes are triggered on import.
"""

import numpy as np

D = 80
K_SAT = 60.5
ALPHA = 0.05
T = 4000
TRIALS = 20
SIGMAS = (3e-6, 1.2e-5, 5e-5)
SPEEDS = (1.0, 100.0, 10000.0)  # Reconstruction choice; verify against paper.
FIXED_W = (1, 16, 256)
BURN = 256  # Reconstruction choice: common warm-up for window comparisons.


def _reaction(target):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = (lo + hi) / 2
        theta = (1 + (D - 1) * t) ** 2 / (1 + (D - 1) * t * t)
        if theta < target:
            lo = t
        else:
            hi = t
    v = np.r_[1.0, np.full(D - 1, (lo + hi) / 2)]
    return 0.5 * v / v.sum()


_GRID = np.linspace(55, 66, 4001)
_PATH = np.array([_reaction(u) for u in _GRID])
_ARC = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(_PATH, axis=0), axis=1))]
S_TOT = float(_ARC[-1])


def theta_of_t(tp):
    """tp contains the true participation values returned by path_at."""
    return np.asarray(tp)


def _theta(x):
    x = np.asarray(x)
    return x.sum(axis=-1) ** 2 / np.sum(x * x, axis=-1)


def path_at(nu, phase):
    s = np.mod(phase + nu * np.arange(1, T + 1), 2 * S_TOT)
    s = np.where(s > S_TOT, 2 * S_TOT - s, s)
    i = np.searchsorted(_ARC, s).clip(1, len(_ARC) - 1)
    w = (s - _ARC[i - 1]) / (_ARC[i] - _ARC[i - 1])
    om = _PATH[i - 1] * (1 - w[:, None]) + _PATH[i] * w[:, None]
    return om, _theta(om)


def classify(x):
    """Binary SAT-side classifier for this fixed-mass orthogonal path only.

    This is not a replacement for the general regime classifier.
    """
    return _theta(np.maximum(x, 1e-300)) > K_SAT


def window_means(y, W):
    """Causal means, with shortened windows during initial warm-up."""
    if not 1 <= W <= len(y):
        raise ValueError("W must be between 1 and the trajectory length")
    cs = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y, axis=0)])
    end = np.arange(1, len(y) + 1)
    start = np.maximum(0, end - W)
    return (cs[end] - cs[start]) / (end - start)[:, None]


def _jac_bound(v, eps=1e-7):
    def f(x):
        return np.array([x.sum(), _theta(x), 1.0])

    J = []
    for j in range(D):
        e = np.zeros(D)
        e[j] = eps
        J.append((f(v + e) - f(np.maximum(v - e, 0))) / (2 * eps))
    return np.abs(J).max()


C = max(_jac_bound(v) for v in _PATH[::200])
LZ = 1 + np.sqrt(D)
ETA = 0.3 / (C * (1 + LZ))
_LOG = np.log(2 / ALPHA)


def nu_max(sigma):
    return 4 * ETA**3 / (27 * (8 * D * sigma**2) * _LOG)


def w_star(sigma, nu):
    return min(
        max(1, int(np.ceil((2 * (8 * D * sigma**2) * _LOG / nu**2) ** (1 / 3)))), T // 4
    )


def d_theory(ratio):
    # Continuous-window formula, not corrected for rounding or clipping W*.
    ratio = np.asarray(ratio)
    return 0.3 * ratio ** (1 / 3)


def d99(d, ok):
    for edge in np.logspace(-3, 1, 17):
        mask = d >= edge
        if mask.sum() > 200 and ok[mask].mean() >= 0.99:
            return float(edge)
    return np.nan

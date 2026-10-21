"""Core library for 'Operational regimes in non-convex optimization' (Sections 2-4).
Features Xi-hat=(Lambda/g_ref, vartheta, I), Algorithm 1 on features, margins m_e (Definition 4).
"""

import numpy as np

INF = float("inf")
THETA0 = dict(eta=0.05, ksrd=1.4, ksat=60.5, gI=1.3)


def kappa_sat_rule(q, csat=0.6):
    """Table-2 rule with q (number of active constraints) in place of n-p; None if q<=2 (sat undefined)."""
    if q <= 2:
        return INF
    return min(max(2.5, np.floor(csat * q) + 0.5), q - 0.5)


def features_from_reaction(lt, F_norm):
    """lt: scaled reactions lambda~_i>0 (active inequalities); F_norm=||sum_i lambda_i grad g_i||_2 (=||A^T lt||)."""
    if lt.size == 0:
        return 0.0, 1.0, 1.0
    L = lt.sum()
    s = (lt**2).sum()
    th = L * L / s
    I = np.sqrt(s) / F_norm if F_norm > 0 else INF
    return L, th, I


def classify_features(Lh, th, I, theta, eps=1e-3):
    """Algorithm 1 evaluated on dimensionless features (Lh=Lambda/g_ref). Returns label."""
    eta, ks, kt, gI = theta["eta"], theta["ksrd"], theta["ksat"], theta["gI"]
    if Lh < eta - eps:
        return "unc"
    if abs(Lh - eta) <= eps:
        return "hyb"
    if abs(th - ks) <= eps:
        return "hyb"
    if th < ks - eps:
        return "srd"
    if th < kt - eps:
        return "int"
    if I > gI + eps:
        return "int"
    if th > kt + eps and I < gI - eps:
        return "sat"
    return "hyb"


def classify_exact(Lh, th, I, theta):
    return classify_features(Lh, th, I, theta, eps=0.0)


def margin(Lh, th, I, theta, label):
    eta, ks, kt, gI = theta["eta"], theta["ksrd"], theta["ksat"], theta["gI"]
    s = dict(
        L=abs(Lh - eta),
        srd=abs(th - ks),
        sat=abs(th - kt) if np.isfinite(kt) else INF,
        I=abs(I - gI),
    )
    T = dict(
        unc=["L"], srd=["L", "srd"], sat=["L", "sat", "I"], int=["L", "srd", "sat", "I"]
    )
    if label not in T:
        return 0.0
    return min(s[k] for k in T[label])

"""
regimes_lib.py
Core library for Regime Classification in Non-Convex / Convex Optimization
Based on: "Operational Regimes in Non-Convex Optimization: A Multiplier-Based Taxonomy"
Parameters: (Lambda, vartheta, I)
"""

import numpy as np

# مقادیر پیش‌فرض Admissible مطابق با جدول ۲ و بخش ۲ مقاله
DEFAULT_PARAMS = {
    "delta_Lambda": 1e-3,  # Unconstrained threshold
    "kappa_srd": 1.35,  # Single-Resource Dominant threshold (1 < kappa_srd < 2)
    "kappa_sat": 2.45,  # Saturation threshold (2 < kappa_sat < n - p)
    "gamma_I": 1.25,  # Geometric interference threshold (> 1)
    "eps_0": 1e-12,  # Floating-point guard (Remark 20 / Algorithm 1)
}


def compute_regime_features(
    lambda_raw, grad_g_norms=None, grad_matrix=None, eps_0=1e-12
):
    """
    محاسبه ویژگی‌های سه‌گانه: Lambda, vartheta, I

    ورودی‌ها:
    - lambda_raw: بردار ضرایب لاگرانژ خام بدست‌آمده از حل‌کننده KKT (اندازه m)
    - grad_g_norms: نرم دو گرادیان قیدها (||∇g_i(x*)||_2). در صورت None بودن، فرض بر ۱ است.
    - grad_matrix: ماتریس گرادیان قیدهای فعال برای محاسبه تداخل I.
    """
    lambda_raw = np.asarray(lambda_raw, dtype=float).flatten()

    # ۱. فیلتر کردن قیدهای فعال
    active_mask = lambda_raw > eps_0
    if not np.any(active_mask):
        return {"Lambda": 0.0, "vartheta": 1.0, "I": 1.0, "active_mask": active_mask}

    lam_act = lambda_raw[active_mask]

    # ۲. مقیاس‌بندی ناوردا: tilde_lambda_i = lambda_i * ||∇g_i||_2
    if grad_g_norms is not None:
        g_norms = np.asarray(grad_g_norms, dtype=float).flatten()[active_mask]
        tilde_lambda = lam_act * g_norms
    else:
        tilde_lambda = lam_act

    # ۳. توده حساسیت (Sensitivity Mass)
    # Lambda = ||tilde_lambda||_1
    Lambda = np.sum(np.abs(tilde_lambda))

    if Lambda < eps_0:
        return {"Lambda": 0.0, "vartheta": 1.0, "I": 1.0, "active_mask": active_mask}

    # ۴. ضریب مشارکت مؤثر (Effective Participation)
    # vartheta = Lambda^2 / ||tilde_lambda||_2^2
    norm_sq = np.sum(tilde_lambda**2)
    vartheta = (Lambda**2) / max(norm_sq, eps_0)

    # ۵. شاخص تداخل هندسی (Interference Index)
    # I = ||tilde_lambda||_2 / ||F||_2 که در آن F = A^T tilde_lambda
    if grad_matrix is not None:
        A_act = np.asarray(grad_matrix)[:, active_mask]
        F = A_act @ tilde_lambda
        norm_F = np.linalg.norm(F)
        I_geom = np.sqrt(norm_sq) / max(norm_F, eps_0)
    else:
        # در غیاب ماتریس گرام، مقدار پیش‌فرض حالت گرادیان‌های متعامد I = 1 است
        I_geom = 1.0

    return {
        "Lambda": Lambda,
        "vartheta": vartheta,
        "I": I_geom,
        "tilde_lambda": tilde_lambda,
        "active_mask": active_mask,
    }


def classify_regime(features, params=None):
    """
    پیاده‌سازی دقیق الگوریتم ۱ (Algorithm 1) و تعریف ۱ (Definition 1)
    خروجی: نام رژیم ('unc', 'srd', 'int', 'sat', 'hyb')
    """
    if params is None:
        params = DEFAULT_PARAMS

    Lambda = features["Lambda"]
    vartheta = features["vartheta"]
    I_geom = features["I"]

    delta_L = params["delta_Lambda"]
    k_srd = params["kappa_srd"]
    k_sat = params["kappa_sat"]
    g_I = params["gamma_I"]
    eps = params.get("eps_0", 1e-12)

    # بررسی مرزهای هایبرید (Lemma 5)
    if (
        np.isclose(Lambda, delta_L, atol=eps)
        or (Lambda > delta_L and np.isclose(vartheta, k_srd, atol=eps))
        or (
            Lambda > delta_L and np.isclose(vartheta, k_sat, atol=eps) and I_geom <= g_I
        )
        or (Lambda > delta_L and vartheta > k_sat and np.isclose(I_geom, g_I, atol=eps))
    ):
        return "hyb"

    # رژیم‌های اصلی طبق Definition 1
    if Lambda < delta_L:
        return "unc"  # Unconstrained
    elif vartheta < k_srd:
        return "srd"  # Single-Resource-Dominant
    elif vartheta > k_sat and I_geom < g_I:
        return "sat"  # Saturated / Distributed
    else:
        return "int"  # Intermediate / Strongly-Coupled

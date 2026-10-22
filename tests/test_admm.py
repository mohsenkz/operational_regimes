"""Regression tests for the actual ADMM state and convergence certificate."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from operational_regimes.optimization.admm_mimo import (
    dual_residual,
    proj_soc,
    rescale_dual,
    run,
    stopping_metrics,
)


def toy():
    # min theta^2 subject to (theta,1) in SOC: unique optimum theta=1.
    a = np.array([[1.0], [0.0]])
    return SimpleNamespace(A=[a], G=[a.T @ a], Z=np.eye(1), dk=np.array([0.0, 1.0]))


class ADMMTests(unittest.TestCase):
    def test_dual_residual_block_cancellation(self):
        a = [np.eye(2), np.eye(2)]
        zn = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
        self.assertEqual(dual_residual(a, [1.0, 1.0], zn, [np.zeros(2)] * 2), 0.0)

    def test_scaled_dual_preserved(self):
        u = [np.array([1.0, -3.0]), np.array([2.0, 7.0])]
        old, new = [2.0, 5.0], [10.0, 0.5]
        for uk, un, ro, rn in zip(u, rescale_dual(u, old, new), old, new):
            np.testing.assert_allclose(uk * ro, un * rn)

    def test_soc_projection_all_branches(self):
        for v, expected in [
            ([2.0, 1.0], [2.0, 1.0]),
            ([-2.0, 1.0], [0.0, 0.0]),
            ([0.0, 2.0], [1.0, 1.0]),
        ]:
            np.testing.assert_allclose(proj_soc(np.array(v)), expected)

    def test_last_iteration_convergence_and_zero_budget(self):
        p = toy()
        reference = run(p, "fixed", 1.0, maxit=1000, tol=1e-8)
        self.assertTrue(reference["converged"])
        final = run(p, "fixed", 1.0, maxit=reference["it"], tol=1e-8)
        self.assertTrue(final["conv"])
        self.assertEqual(final["it"], reference["it"])
        self.assertFalse(
            run(p, "fixed", 1.0, maxit=reference["it"] - 1, tol=1e-8)["conv"]
        )
        with self.assertRaises(ValueError):
            run(p, "fixed", 1.0, maxit=0)

    def test_tolerances_include_dimensions_and_rhs(self):
        p = toy()
        m = stopping_metrics(
            p.A,
            p.dk,
            np.zeros(1),
            [np.zeros(2)],
            [np.zeros(2)],
            [1.0],
            [np.zeros(2)],
            1e-3,
            1e-2,
        )
        self.assertAlmostEqual(m["eps_pri"], np.sqrt(2) * 1e-3 + 1e-2)
        self.assertAlmostEqual(m["eps_dual"], 1e-3)

    def test_fixed_balanced_and_relaxed_reach_same_solution(self):
        for strategy, alpha in [
            ("fixed", 1.0),
            ("balance", 1.0),
            ("balance_normalized", 1.0),
            ("history", 1.0),
            ("fixed", 1.6),
            ("balance", 1.6),
        ]:
            out = run(toy(), strategy, 3.0, maxit=2000, tol=1e-8, alpha=alpha)
            self.assertTrue(out["conv"], (strategy, alpha))
            self.assertLess(abs(out["objective"] - 1), 1e-6)
            self.assertLess(out["feasibility"], 1e-6)
            for key, eps in [
                ("r", "eps_pri"),
                ("s", "eps_dual"),
                ("stationarity", "eps_dual"),
            ]:
                self.assertLessEqual(out[key], out[eps])

    def test_non_relaxed_stationarity_equals_dual_residual(self):
        out = run(toy(), "balance", 20.0, maxit=80, track=True)
        for point in out["history"]:
            self.assertAlmostEqual(point["s"], point["stationarity"], places=10)

    def test_shuffled_requires_independent_pool_and_unknown_rejected(self):
        for st in ["shuffled", "typo"]:
            with self.assertRaises(ValueError):
                run(toy(), st, 1.0)


if __name__ == "__main__":
    unittest.main()

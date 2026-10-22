import importlib.util
from pathlib import Path
import unittest
import numpy as np

P = Path(__file__).resolve().parents[1] / "experiments/rl_bcd/benchmark_rl.py"
spec = importlib.util.spec_from_file_location("rl", P)
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


class TestRL(unittest.TestCase):
    def test_descent_guard(self):
        env = rl.Env(417, rl.TRAIN_CFGS[0])
        old = env.f
        direction = env.grad[:2] / max(np.linalg.norm(env.grad[:2]), 1e-12)
        env.step(0, env.x[:2] + 0.01 * direction)
        self.assertGreater(env.backtracks, 0)
        self.assertLessEqual(env.f, old + 1e-12)

    def test_projection_retry(self):
        from unittest.mock import patch

        def disk(b, z):
            return np.array([z @ z - 1.0]), np.array([2 * z])

        proj = rl.Projector(disk)
        with patch.object(
            rl.bcd.Projector,
            "__call__",
            side_effect=RuntimeError("forced initial failure"),
        ):
            z = proj(0, np.array([2.0, 0.0]), np.zeros(2))
        np.testing.assert_allclose(z, [1.0, 0.0], atol=1e-7)

    def test_policy_gradient_direction(self):
        policy = rl.Policy(41)
        rng = np.random.default_rng(9)
        F = rng.normal(size=(4, rl.D))
        prob, cache, value = policy.forward(F)
        p0 = prob[2]
        # Positive return for action 2 should increase its probability.
        policy.update([(cache, prob, 2, 2.0, value)])
        self.assertGreater(policy.forward(F)[0][2], p0)

    def test_masked_regime_and_permutation(self):
        env = rl.Env(417, rl.TRAIN_CFGS[0])
        rng = np.random.default_rng(7)
        # Go to an active boundary to obtain nontrivial geometry.
        for _ in range(8):
            b, p = env.gs()
            env.step(b, p)
        base = env.features("base", rng)
        reg = env.features("regime", rng)
        shuffle = env.features("shuffled", rng)
        np.testing.assert_array_equal(base[:, 7:], 0)
        np.testing.assert_allclose(base[:, :7], reg[:, :7])
        self.assertEqual(
            sorted(map(tuple, reg[:, 7:])), sorted(map(tuple, shuffle[:, 7:]))
        )

    def test_reproducible_training(self):
        a = rl.Policy(12)
        b = rl.Policy(12)
        for variant in ["base", "regime", "shuffled"]:
            x = rl.episode(961, rl.TRAIN_CFGS[0], variant, a, 51, limit=16, train=True)
            y = rl.episode(961, rl.TRAIN_CFGS[0], variant, b, 51, limit=16, train=True)
            np.testing.assert_array_equal(x["x"], y["x"])
            for k in a.p:
                np.testing.assert_array_equal(a.p[k], b.p[k])
            self.assertLessEqual(x["objective"], x["initial_objective"] + 1e-8)

    def test_policy_gradient_finite_difference(self):
        p = rl.Policy(16)
        rng = np.random.default_rng(21)
        F = rng.normal(size=(4, rl.D))
        prob, (F, h, _), _ = p.forward(F)
        a = 2
        dz = np.eye(4)[a] - prob
        analytic = F.T @ (dz[:, None] * p.p["v"][None, :] * (1 - h * h))
        eps = 1e-6
        for i, j in [(0, 0), (7, 3), (2, 11)]:
            old = p.p["W"][i, j]
            p.p["W"][i, j] = old + eps
            plus = np.log(p.forward(F)[0][a])
            p.p["W"][i, j] = old - eps
            minus = np.log(p.forward(F)[0][a])
            p.p["W"][i, j] = old
            self.assertAlmostEqual((plus - minus) / (2 * eps), analytic[i, j], places=7)


if __name__ == "__main__":
    unittest.main()

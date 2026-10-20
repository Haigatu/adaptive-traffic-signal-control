import unittest
import numpy as np
import torch
from atsc.data import Windows
from atsc.models import Forecaster
from atsc.control import Replay
from atsc.environment import Environment
import libsumo as sim


class CoreTests(unittest.TestCase):
    def test_split_targets_never_cross_end(self):
        x = np.arange(100, dtype=np.float32)[:, None]
        s = Windows(x, 60, 80, 0, 1)
        self.assertEqual(s.indices[0], 60)
        self.assertEqual(s.indices[-1] + 2, 79)
        inputs, target = s[0]
        self.assertLess(float(inputs.max()), float(target.min()))

    def test_tgcn_node_permutation_equivariance(self):
        torch.manual_seed(1)
        m = Forecaster("tgcn", 8)
        x = torch.randn(2, 12, 4)
        a = torch.tensor(
            [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.6, 0.4], [0, 0, 0.4, 0.6]]
        )
        p = torch.tensor([2, 0, 3, 1])
        y = m(x, a)
        torch.testing.assert_close(
            m(x[:, :, p], a[p][:, p]), y[:, :, p], atol=1e-6, rtol=1e-5
        )

    def test_forecaster_all_horizons_and_nodes(self):
        for kind in ["tgcn", "gru", "lstm", "mlp"]:
            m = Forecaster(kind, 8)
            x = torch.randn(3, 12, 4)
            self.assertEqual(m(x, torch.eye(4).to_sparse()).shape, (3, 3, 4))

    def test_prioritized_replay_weights(self):
        r = Replay(capacity=100)
        for i in range(100):
            r.add(np.zeros(4), i % 4, float(i), np.ones(4), 0.0, 10.0)
        r.priority[99] = 100
        seen = 0
        for _ in range(50):
            ids, fields, w = r.sample(16, 0.5)
            seen += 99 in ids
            self.assertTrue(torch.isfinite(w).all())
            self.assertLessEqual(float(w.max()), 1.0)
        self.assertGreater(seen, 35)

    def test_sumo_horizon_yellow_and_shape(self):
        e = Environment(duration=23, warmup=300)
        e.reset(9999, 0.4)
        try:
            first = e.phase
            s, r, d, i = e.step(first)
            self.assertEqual(i["duration"], 10)
            self.assertFalse(d)
            second = (first + 1) % 4
            s, r, d, i = e.step(second)
            self.assertEqual(i["duration"], 13)
            self.assertTrue(d)
            self.assertEqual(e.elapsed, 23)
            self.assertEqual(s.shape, (52,))
            state = sim.trafficlight.getRedYellowGreenState("J")
            self.assertEqual(
                state, "".join("G" if j // 3 == second else "r" for j in range(12))
            )
            self.assertEqual(e.metrics()["teleports"], 0)
        finally:
            e.close()


if __name__ == "__main__":
    unittest.main()

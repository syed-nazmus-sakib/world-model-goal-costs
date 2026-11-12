"""Metro-WM-style sub-goals (code/lewm_explorations/paa/metro_graph.py) on a toy line world.  Needs torch and scipy; skipped without."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "lewm_explorations" / "paa"))
try:
    import scipy  # noqa: F401
    import torch

    from metro_graph import MetroGraph
except ImportError:  # the default test environment has numpy only
    torch = None


def line_world():
    """Episode 0 walks x = 0..10 at y = 0, episode 1 walks x = 10..20 at y = 0.1, episode 2 is far away; nodes 5 frames apart."""
    xy, ep, step = [], [], []
    for e, (x0, y) in enumerate(((0, 0.0), (10, 0.1), (100, 0.0))):
        for i in range(11 if e < 2 else 4):
            xy.append((x0 + i, y)); ep.append(e); step.append(5 * i)
    return torch.tensor(xy, dtype=torch.float32), np.array(ep), np.array(step)


@unittest.skipIf(torch is None, "torch or scipy not installed")
class MetroGraphTests(unittest.TestCase):
    def setUp(self):
        self.emb, self.ep, self.step = line_world()

    def graph(self, budget):
        return MetroGraph(self.emb, self.ep, self.step, k=4, alpha=0.25, budget=budget)

    def sub_x(self, g, start, goal):
        node, u0, j0 = g.subgoal(torch.tensor(start), g.goal_tree(torch.tensor(goal)))
        return (None if node is None else float(self.emb[node, 0])), j0

    def test_calibration_reads_gaps_as_frames(self):
        g = self.graph(10)
        np.testing.assert_allclose(g.lat[:3], [0.0, 1.0, 2.0], atol=1e-5)   # consecutive nodes are 1 apart, 5 frames
        self.assertAlmostEqual(g.eps, 2.0, places=4)                         # radius = Lambda(budget)
        np.testing.assert_allclose(g.cost(np.array([0.0, 1.0, 1.5, 100.0])), [1.0, 5.0, 7.5, 10.0], atol=1e-4)

    def test_bridges_join_other_episodes_only(self):
        g = self.graph(10)
        G = g._top[:, :-1].T.tocoo()   # back to the forward graph, without the goal column
        same = self.ep[G.row] == self.ep[G.col]
        self.assertTrue((G.data[same] == 5.0).all() and (self.step[G.col[same]] - self.step[G.row[same]] == 5).all())
        self.assertGreater(g.n_bridge, 0)
        self.assertEqual(int((~same).sum()), 2 * g.n_bridge)
        self.assertFalse(((self.ep[G.row] == 2) ^ (self.ep[G.col] == 2)).any())   # the far episode stays unconnected

    def test_subgoal_is_within_the_budget_and_moves_with_it(self):
        near, j_near = self.sub_x(self.graph(10), (0.0, 0.0), (20.0, 0.1))
        far, j_far = self.sub_x(self.graph(20), (0.0, 0.0), (20.0, 0.1))
        self.assertTrue(2.0 <= near <= 3.0, near)   # 10 frames = two nodes; the first move is always taken
        self.assertTrue(4.0 <= far <= 5.0, far)
        self.assertTrue(95.0 <= j_near <= 110.0, j_near)   # 20 nodes of 5 frames across the bridge

    def test_goal_within_budget_unreachable_goal_and_direction(self):
        g = self.graph(10)
        self.assertEqual(self.sub_x(g, (19.0, 0.1), (20.0, 0.1))[0], None)        # goal within one budget
        x, j = self.sub_x(g, (0.0, 0.0), (103.0, 0.0))                            # goal in the unconnected episode
        self.assertTrue(x is None and not np.isfinite(j))
        x, j = self.sub_x(g, (20.0, 0.1), (0.0, 0.0))                             # temporal edges do not run backward
        self.assertTrue(x is None and not np.isfinite(j))


if __name__ == "__main__":
    unittest.main()

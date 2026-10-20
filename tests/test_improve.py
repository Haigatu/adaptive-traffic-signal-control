import unittest
import numpy as np
from atsc.improve import teacher_scores
from atsc.guarded import state_masks


class ImprovementTests(unittest.TestCase):
    def test_teacher_respects_approach_order_and_causal_forecast(self):
        state = np.zeros((1, 56), dtype=np.float32)
        state[0, 36] = 1
        state[0, 3:6] = .5
        self.assertEqual(teacher_scores(state, 5).argmax(), 1)
        changed = state.copy()
        changed[0, [42, 46, 50]] = 2
        self.assertGreater(teacher_scores(changed, 5)[0, 2],
                           teacher_scores(state, 5)[0, 2])
        self.assertEqual(teacher_scores(changed, 5)[0, 1],
                         teacher_scores(state, 5)[0, 1])

    def test_deadline_mask_overrides_largest_queue(self):
        state = np.zeros((1, 56), dtype=np.float32)
        state[0, 36] = 1
        state[0, :3] = 10
        state[0, -4:] = np.array([0, 110, 80, 50])/120
        scores = teacher_scores(state, 30)
        self.assertEqual(scores.argmax(), 0)
        action = np.where(state_masks(state).numpy(), scores, -np.inf).argmax()
        self.assertNotEqual(action, 0)

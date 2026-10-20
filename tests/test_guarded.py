import unittest
import numpy as np
from atsc.guarded import feasible_actions, ClockEnvironment, MAX_RED, state_masks


class GuardTests(unittest.TestCase):
    def test_mask_reserves_time_for_all_approaches(self):
        # Keeping phase zero would strand phase one; switching to it is feasible.
        mask = feasible_actions([0, 110, 80, 50], 0)[0]
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])

    def test_adversarial_valid_choices_preserve_bound(self):
        rng = np.random.default_rng(7)
        ages = np.array([0.0, 93.0, 63.0, 33.0])
        phase = 0
        for _ in range(20000):
            mask = feasible_actions(ages, phase)[0]
            self.assertTrue(mask.any())
            action = int(rng.choice(np.flatnonzero(mask)))
            if action != phase:
                ages += 3
                self.assertLessEqual(ages.max(), MAX_RED)
            ages[action] = 0
            ages += 10
            ages[action] = 0
            self.assertLessEqual(ages.max(), MAX_RED)
            phase = action

    def test_sumo_guard_includes_yellow_and_matches_state_mask(self):
        env = ClockEnvironment(duration=360, warmup=3600, enforce=True)
        s = env.reset(9001, 0.4, "train")
        try:
            self.assertEqual(s.shape, (56,))
            while True:
                mask = env.action_mask()
                np.testing.assert_array_equal(state_masks(s)[0].numpy(), mask)
                # Repeatedly favor phase zero; guard must still serve all phases.
                action = int(np.flatnonzero(mask)[0])
                s, _, done, _ = env.step(action)
                if done:
                    break
            self.assertLessEqual(env.metrics()["max_without_green_s"], MAX_RED)
            self.assertGreater(env.metrics()["constrained_decisions"], 0)
            self.assertEqual(env.elapsed, 360)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

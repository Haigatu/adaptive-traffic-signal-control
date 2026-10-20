import tempfile
import unittest
from atsc.environment import Environment
from atsc.traci_cache import CachedTraCI


class CacheTests(unittest.TestCase):
    def test_subscriptions_preserve_states_and_metrics(self):
        runs = []
        for cached in (False, True):
            with tempfile.TemporaryDirectory(dir=".", ignore_cleanup_errors=True) as folder:
                env = Environment(with_forecast=False, backend="traci",
                                  warmup=90, duration=90, run_dir=folder)
                if cached:
                    env.sim = CachedTraCI(env.sim)
                try:
                    state = env.reset(73, .7, "test")
                    states = [state.tolist()]
                    for action in [1, 1, 2, 3, 0, 2, 1, 3]:
                        state, _, done, _ = env.step(action)
                        states.append(state.tolist())
                        if done:
                            break
                    positions = [(v, env.sim.vehicle.getPosition(v),
                                  env.sim.vehicle.getAngle(v))
                                 for v in env.sim.vehicle.getIDList()]
                    runs.append((states, env.metrics(), positions))
                finally:
                    env.close()
        self.assertEqual(runs[0], runs[1])

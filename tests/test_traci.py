import tempfile
import unittest
from pathlib import Path

from atsc.environment import Environment


class TraCITests(unittest.TestCase):
    def test_external_control_and_yellow_timing(self):
        Path("data/runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="data/runs", ignore_cleanup_errors=True) as folder:
            env = Environment(backend="traci", run_dir=Path(folder),
                              duration=23, warmup=0)
            try:
                state = env.reset(9999, 0.4, "test")
                self.assertEqual(state.shape, (52,))
                _, _, done, info = env.step(0)
                self.assertEqual(info["duration"], 10)
                self.assertFalse(done)
                _, _, done, info = env.step(1)
                self.assertEqual(info["duration"], 13)
                self.assertTrue(done)
                self.assertEqual(env.sim.trafficlight.getProgram("J"), "online")
                self.assertEqual(env.sim.trafficlight.getRedYellowGreenState("J"),
                                 "rrrGGGrrrrrr")
            finally:
                env.close()

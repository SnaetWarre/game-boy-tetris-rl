import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gb_tetris_rl.training import train_dqn_agent


class TinyTrainingEnvironment(gym.Env):
    def __init__(
        self,
        rom_path: str | Path,
        *,
        render_mode: str | None = None,
        display_emulator_window: bool | None = None,
    ) -> None:
        del rom_path
        del render_mode
        del display_emulator_window
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(180,),
            dtype=np.uint8,
        )
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(180, dtype=np.uint8), {}

    def step(self, action):
        del action
        self._step_count += 1
        observation = np.zeros(180, dtype=np.uint8)
        terminated = self._step_count >= 3
        return observation, 0.0, terminated, False, {}


@unittest.skipUnless(find_spec("stable_baselines3"), "training extra is not installed")
class TrainingSmokeTests(unittest.TestCase):
    def test_trains_and_saves_a_tiny_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "tiny-model"
            with patch("gb_tetris_rl.training.TetrisEnvironment", TinyTrainingEnvironment):
                saved_model_path = train_dqn_agent(
                    "unused.gb",
                    output_path,
                    total_timesteps=12,
                    seed=7,
                    device="cpu",
                )

            self.assertEqual(saved_model_path, output_path.with_suffix(".zip"))
            self.assertTrue(saved_model_path.is_file())

    def test_rejects_zero_parallel_environments(self) -> None:
        with self.assertRaisesRegex(ValueError, "environment_count"):
            train_dqn_agent(
                "unused.gb",
                "unused-model",
                total_timesteps=12,
                seed=7,
                device="cpu",
                environment_count=0,
            )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gb_tetris_rl.agent.training import TrainingConfig, train_agent
from gb_tetris_rl.game.contracts import AGENT_ACTION_COUNT, AGENT_OBSERVATION_SHAPE


class TinyTrainingEnvironment(gym.Env):
    def __init__(
        self,
        rom_path: str | Path,
        *,
        render_mode: str | None = None,
        display_emulator_window: bool | None = None,
        emulation_speed: int = 0,
    ) -> None:
        del rom_path, render_mode, display_emulator_window, emulation_speed
        self.action_space = spaces.Discrete(AGENT_ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=AGENT_OBSERVATION_SHAPE,
            dtype=np.uint8,
        )
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(AGENT_OBSERVATION_SHAPE, dtype=np.uint8), {}

    def step(self, action):
        del action
        self._step_count += 1
        observation = np.zeros(AGENT_OBSERVATION_SHAPE, dtype=np.uint8)
        terminated = self._step_count >= 3
        return observation, 0.0, terminated, False, {}


@unittest.skipUnless(find_spec("stable_baselines3"), "training dependencies are not installed")
class TrainingSmokeTests(unittest.TestCase):
    def test_trains_and_saves_a_tiny_model(self) -> None:
        training_config = TrainingConfig(
            total_timesteps=12,
            seed=7,
            device="cpu",
            environment_count=1,
            demonstration_count=0,
            imitation_epoch_count=0,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "tiny-run"
            with patch(
                "gb_tetris_rl.agent.training.TetrisEnvironment",
                TinyTrainingEnvironment,
            ):
                training_artifacts = train_agent(
                    "unused.gbc",
                    run_directory,
                    training_config,
                )

            self.assertIsNone(training_artifacts.imitation_model_path)
            self.assertEqual(
                training_artifacts.dqn_model_path,
                run_directory.resolve() / "dqn-final.zip",
            )
            self.assertTrue(training_artifacts.dqn_model_path.is_file())

    def test_rejects_zero_parallel_environments(self) -> None:
        invalid_config = TrainingConfig(
            total_timesteps=12,
            environment_count=0,
            demonstration_count=0,
            imitation_epoch_count=0,
        )
        with self.assertRaisesRegex(ValueError, "environment_count"):
            train_agent("unused.gbc", "unused-run", invalid_config)


if __name__ == "__main__":
    unittest.main()

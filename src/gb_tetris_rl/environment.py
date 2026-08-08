from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pyboy import PyBoy

from gb_tetris_rl.actions import PYBOY_BUTTON_BY_ACTION, TetrisAction
from gb_tetris_rl.game_adapters import TETRIS_BOARD_SHAPE, create_game_adapter
from gb_tetris_rl.rewards import TetrisSnapshot, calculate_transition_reward, create_snapshot
from gb_tetris_rl.roms import validate_tetris_rom

TETRIS_OBSERVATION_SHAPE = (TETRIS_BOARD_SHAPE[0] * TETRIS_BOARD_SHAPE[1],)


class TetrisEnvironment(gym.Env[NDArray[np.uint8], int]):
    """Gymnasium environment backed by a supported PyBoy game adapter."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        rom_path: str | Path,
        *,
        render_mode: str | None = None,
        frames_per_action: int = 2,
        maximum_episode_steps: int = 20_000,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"unsupported render mode: {render_mode}")
        if frames_per_action < 1:
            raise ValueError("frames_per_action must be at least 1")
        if maximum_episode_steps < 1:
            raise ValueError("maximum_episode_steps must be at least 1")

        validated_rom = validate_tetris_rom(rom_path)
        window_backend = "SDL2" if render_mode == "human" else "null"
        pyboy_options: dict[str, str | bool] = {
            "window": window_backend,
            "sound_emulated": False,
        }
        symbols_path = validated_rom.path.with_suffix(".sym")
        if symbols_path.is_file():
            pyboy_options["symbols"] = str(symbols_path)
        self._pyboy = PyBoy(str(validated_rom.path), **pyboy_options)
        self._pyboy.set_emulation_speed(1 if render_mode == "human" else 0)
        self._game_adapter = create_game_adapter(validated_rom.game, self._pyboy)

        self.render_mode = render_mode
        self._frames_per_action = frames_per_action
        self._maximum_episode_steps = maximum_episode_steps
        self._episode_step_count = 0
        self._previous_snapshot = self._capture_snapshot()
        self._is_closed = False

        self.action_space = spaces.Discrete(len(TetrisAction))
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=TETRIS_OBSERVATION_SHAPE,
            dtype=np.uint8,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.uint8], dict[str, int]]:
        del options
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self._game_adapter.reset(seed)
        self._episode_step_count = 0
        self._previous_snapshot = self._capture_snapshot()
        return self._observe(), self._build_info(self._previous_snapshot)

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, int]]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid Tetris action: {action!r}")

        tetris_action = TetrisAction(action)
        pyboy_button = PYBOY_BUTTON_BY_ACTION[tetris_action]
        if pyboy_button is not None:
            self._pyboy.button(pyboy_button)

        should_render_frame = self.render_mode is not None
        emulator_is_running = self._pyboy.tick(
            self._frames_per_action,
            render=should_render_frame,
            sound=False,
        )
        self._game_adapter.update_after_tick()
        self._episode_step_count += 1

        game_is_over = self._game_is_over() or not emulator_is_running
        current_snapshot = self._capture_snapshot()
        reward = calculate_transition_reward(
            self._previous_snapshot,
            current_snapshot,
            game_is_over=game_is_over,
        )
        self._previous_snapshot = current_snapshot

        episode_was_truncated = self._episode_step_count >= self._maximum_episode_steps
        return (
            self._observe(),
            reward,
            game_is_over,
            episode_was_truncated,
            self._build_info(current_snapshot),
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        screen_rgba = np.asarray(self._pyboy.screen.ndarray)
        return np.array(screen_rgba[:, :, :3], dtype=np.uint8, copy=True)

    def close(self) -> None:
        if not self._is_closed:
            self._pyboy.stop(save=False)
            self._is_closed = True

    def _observe(self) -> NDArray[np.uint8]:
        return self._read_board().reshape(TETRIS_OBSERVATION_SHAPE)

    def _read_board(self) -> NDArray[np.uint8]:
        return self._game_adapter.read_board()

    def _capture_snapshot(self) -> TetrisSnapshot:
        return create_snapshot(
            score=self._game_adapter.score,
            cleared_lines=self._game_adapter.cleared_lines,
            board=self._read_board(),
        )

    def _game_is_over(self) -> bool:
        return self._game_adapter.game_is_over

    def _build_info(self, snapshot: TetrisSnapshot) -> dict[str, int]:
        return {
            "score": snapshot.score,
            "cleared_lines": snapshot.cleared_lines,
            "level": self._game_adapter.level,
            "aggregate_height": snapshot.board.aggregate_height,
            "holes": snapshot.board.holes,
            "bumpiness": snapshot.board.bumpiness,
            "episode_steps": self._episode_step_count,
        }

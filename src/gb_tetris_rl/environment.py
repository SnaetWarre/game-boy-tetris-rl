from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pyboy import PyBoy

from gb_tetris_rl.actions import PYBOY_BUTTON_BY_ACTION, TetrisAction
from gb_tetris_rl.rewards import TetrisSnapshot, calculate_transition_reward, create_snapshot
from gb_tetris_rl.roms import validate_tetris_rom

TETRIS_BOARD_SHAPE = (18, 10)
TETRIS_OBSERVATION_SHAPE = (TETRIS_BOARD_SHAPE[0] * TETRIS_BOARD_SHAPE[1],)


class TetrisEnvironment(gym.Env[NDArray[np.uint8], int]):
    """Gymnasium environment backed by PyBoy's official Tetris wrapper."""

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
        self._pyboy = PyBoy(
            str(validated_rom.path),
            window=window_backend,
            sound_emulated=False,
        )
        self._pyboy.set_emulation_speed(1 if render_mode == "human" else 0)
        self._tetris_wrapper = self._pyboy.game_wrapper
        self._tetris_wrapper.game_area_mapping(
            self._tetris_wrapper.mapping_minimal,
            0,
        )
        self._tetris_wrapper.start_game(timer_div=0)

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
        timer_divider = None if seed is None else seed % 256
        self._tetris_wrapper.reset_game(timer_div=timer_divider)
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
        board = np.asarray(self._tetris_wrapper.game_area(), dtype=np.uint8)
        if board.shape != TETRIS_BOARD_SHAPE:
            raise RuntimeError(
                f"PyBoy returned board shape {board.shape}; expected {TETRIS_BOARD_SHAPE}"
            )
        return np.array(board, dtype=np.uint8, copy=True)

    def _capture_snapshot(self) -> TetrisSnapshot:
        return create_snapshot(
            score=int(self._tetris_wrapper.score),
            cleared_lines=int(self._tetris_wrapper.lines),
            board=self._read_board(),
        )

    def _game_is_over(self) -> bool:
        game_over_value = self._tetris_wrapper.game_over
        return bool(game_over_value() if callable(game_over_value) else game_over_value)

    def _build_info(self, snapshot: TetrisSnapshot) -> dict[str, int]:
        return {
            "score": snapshot.score,
            "cleared_lines": snapshot.cleared_lines,
            "level": int(self._tetris_wrapper.level),
            "aggregate_height": snapshot.board.aggregate_height,
            "holes": snapshot.board.holes,
            "bumpiness": snapshot.board.bumpiness,
            "episode_steps": self._episode_step_count,
        }

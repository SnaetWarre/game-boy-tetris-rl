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
from gb_tetris_rl.roms import SupportedGame, validate_tetris_rom

TETRIS_OBSERVATION_SHAPE = (TETRIS_BOARD_SHAPE[0] * TETRIS_BOARD_SHAPE[1],)
TETROMINO_TYPE_COUNT = 7
PLACEMENT_CONTEXT_SIZE = TETROMINO_TYPE_COUNT * 2
PLACEMENT_OBSERVATION_SHAPE = (
    TETRIS_OBSERVATION_SHAPE[0] + PLACEMENT_CONTEXT_SIZE,
)
PLACEMENT_ROTATION_COUNT = 4
PLACEMENT_COLUMN_COUNT = 10
PLACEMENT_ACTION_COUNT = PLACEMENT_ROTATION_COUNT * PLACEMENT_COLUMN_COUNT
HELD_PIECE_TYPE_COUNT = TETROMINO_TYPE_COUNT + 1
HOLD_PLACEMENT_OBSERVATION_SHAPE = (
    PLACEMENT_OBSERVATION_SHAPE[0] + HELD_PIECE_TYPE_COUNT,
)
HOLD_PLACEMENT_ACTION_COUNT = PLACEMENT_ACTION_COUNT * 2


class TetrisEnvironment(gym.Env[NDArray[np.uint8], int]):
    """Gymnasium environment backed by a supported PyBoy game adapter."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        rom_path: str | Path,
        *,
        render_mode: str | None = None,
        display_emulator_window: bool | None = None,
        emulation_speed: int | None = None,
        control_mode: str = "buttons",
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

        if display_emulator_window is None:
            display_emulator_window = render_mode == "human"
        if display_emulator_window and render_mode != "human":
            raise ValueError("display_emulator_window requires render_mode='human'")
        if emulation_speed is None:
            emulation_speed = 1 if display_emulator_window else 0
        if emulation_speed < 0:
            raise ValueError("emulation_speed cannot be negative")
        if control_mode not in {"buttons", "placements", "placements-hold"}:
            raise ValueError(f"unsupported control mode: {control_mode}")

        validated_rom = validate_tetris_rom(rom_path)
        if (
            control_mode in {"placements", "placements-hold"}
            and validated_rom.game is not SupportedGame.PANDORAS_BLOCKS
        ):
            raise ValueError("placement controls currently require Pandora's Blocks")
        window_backend = "SDL2" if display_emulator_window else "null"
        pyboy_options: dict[str, str | bool] = {
            "window": window_backend,
            "sound_emulated": False,
        }
        symbols_path = validated_rom.path.with_suffix(".sym")
        if symbols_path.is_file():
            pyboy_options["symbols"] = str(symbols_path)
        self._pyboy = PyBoy(str(validated_rom.path), **pyboy_options)
        self._pyboy.set_emulation_speed(emulation_speed)
        self._game_adapter = create_game_adapter(validated_rom.game, self._pyboy)

        self.render_mode = render_mode
        self._should_render_frames = display_emulator_window or render_mode == "rgb_array"
        self._control_mode = control_mode
        self._frames_per_action = frames_per_action
        self._maximum_episode_steps = maximum_episode_steps
        self._episode_step_count = 0
        self._previous_snapshot = self._capture_snapshot()
        self._is_closed = False

        action_count_by_control_mode = {
            "buttons": len(TetrisAction),
            "placements": PLACEMENT_ACTION_COUNT,
            "placements-hold": HOLD_PLACEMENT_ACTION_COUNT,
        }
        observation_shape_by_control_mode = {
            "buttons": TETRIS_OBSERVATION_SHAPE,
            "placements": PLACEMENT_OBSERVATION_SHAPE,
            "placements-hold": HOLD_PLACEMENT_OBSERVATION_SHAPE,
        }
        action_count = action_count_by_control_mode[control_mode]
        observation_shape = observation_shape_by_control_mode[control_mode]
        self.action_space = spaces.Discrete(action_count)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=observation_shape,
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

        if self._control_mode in {"placements", "placements-hold"}:
            use_hold, placement_action = (
                divmod(action, PLACEMENT_ACTION_COUNT)
                if self._control_mode == "placements-hold"
                else (0, action)
            )
            target_rotation, right_moves_from_left_wall = divmod(
                placement_action,
                PLACEMENT_COLUMN_COUNT,
            )
            emulator_is_running = self._game_adapter.place_piece(
                target_rotation,
                right_moves_from_left_wall,
                use_hold=bool(use_hold),
                render_frames=self._should_render_frames,
            )
        else:
            tetris_action = TetrisAction(action)
            pyboy_button = PYBOY_BUTTON_BY_ACTION[tetris_action]
            if pyboy_button is not None:
                self._pyboy.button(pyboy_button)

            emulator_is_running = self._pyboy.tick(
                self._frames_per_action,
                render=self._should_render_frames,
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
        flattened_board = self._read_board().reshape(TETRIS_OBSERVATION_SHAPE)
        if self._control_mode == "buttons":
            return flattened_board

        piece_context = np.zeros(PLACEMENT_CONTEXT_SIZE, dtype=np.uint8)
        current_piece = self._game_adapter.current_piece
        next_piece = self._game_adapter.next_piece
        if 0 <= current_piece < TETROMINO_TYPE_COUNT:
            piece_context[current_piece] = 1
        if 0 <= next_piece < TETROMINO_TYPE_COUNT:
            piece_context[TETROMINO_TYPE_COUNT + next_piece] = 1
        placement_observation = np.concatenate((flattened_board, piece_context))
        if self._control_mode == "placements":
            return placement_observation

        held_piece_context = np.zeros(HELD_PIECE_TYPE_COUNT, dtype=np.uint8)
        held_piece = self._game_adapter.held_piece
        if 0 <= held_piece < HELD_PIECE_TYPE_COUNT:
            held_piece_context[held_piece] = 1
        return np.concatenate((placement_observation, held_piece_context))

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
            "current_piece": self._game_adapter.current_piece,
            "next_piece": self._game_adapter.next_piece,
            "held_piece": self._game_adapter.held_piece,
        }

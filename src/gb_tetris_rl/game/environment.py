from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pyboy import PyBoy

from gb_tetris_rl.game.contracts import (
    AGENT_ACTION_COUNT,
    AGENT_OBSERVATION_SHAPE,
    AgentObservation,
    EpisodeInfo,
    decode_agent_action,
    encode_agent_observation,
)
from gb_tetris_rl.game.pandoras_blocks import PandorasBlocksAdapter
from gb_tetris_rl.game.rewards import TetrisSnapshot, calculate_transition_reward, create_snapshot
from gb_tetris_rl.game.rom import validate_pandoras_blocks_rom


class TetrisEnvironment(gym.Env[AgentObservation, int]):
    """The single Gymnasium contract used by both training and evaluation."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        rom_path: str | Path,
        *,
        render_mode: str | None = None,
        display_emulator_window: bool | None = None,
        emulation_speed: int | None = None,
        maximum_episode_steps: int = 20_000,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"unsupported render mode: {render_mode}")
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

        validated_rom = validate_pandoras_blocks_rom(rom_path)
        pyboy_options: dict[str, str | bool] = {
            "window": "SDL2" if display_emulator_window else "null",
            "sound_emulated": False,
        }
        symbols_path = validated_rom.path.with_suffix(".sym")
        if symbols_path.is_file():
            pyboy_options["symbols"] = str(symbols_path)

        self._pyboy = PyBoy(str(validated_rom.path), **pyboy_options)
        self._pyboy.set_emulation_speed(emulation_speed)
        self._game = PandorasBlocksAdapter(self._pyboy)

        self.render_mode = render_mode
        self._should_render_frames = display_emulator_window or render_mode == "rgb_array"
        self._maximum_episode_steps = maximum_episode_steps
        self._episode_step_count = 0
        self._emulator_is_running = True
        self._previous_snapshot = self._capture_snapshot()
        self._is_closed = False

        self.action_space = spaces.Discrete(AGENT_ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=AGENT_OBSERVATION_SHAPE,
            dtype=np.uint8,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[AgentObservation, EpisodeInfo]:
        del options
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self._game.reset(seed)
        self._episode_step_count = 0
        self._previous_snapshot = self._capture_snapshot()
        return self._observe(), self._build_episode_info(self._previous_snapshot)

    def step(
        self,
        action: int,
    ) -> tuple[AgentObservation, float, bool, bool, EpisodeInfo]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid Tetris action: {action!r}")

        placement = decode_agent_action(action)
        self._emulator_is_running = self._game.place_piece(
            placement.rotation,
            placement.column_from_left_wall,
            use_hold=placement.uses_hold,
            render_frames=self._should_render_frames,
        )
        self._episode_step_count += 1

        game_is_over = self._game.game_is_over or not self._emulator_is_running
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
            self._build_episode_info(current_snapshot),
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        screen_rgba = np.asarray(self._pyboy.screen.ndarray)
        return np.array(screen_rgba[:, :, :3], dtype=np.uint8, copy=True)

    @property
    def emulator_is_running(self) -> bool:
        return self._emulator_is_running

    def close(self) -> None:
        if not self._is_closed:
            self._pyboy.stop(save=False)
            self._is_closed = True

    def _observe(self) -> AgentObservation:
        return encode_agent_observation(
            self._game.read_board(),
            self._game.current_piece,
            self._game.next_piece,
            self._game.held_piece,
        )

    def _capture_snapshot(self) -> TetrisSnapshot:
        return create_snapshot(
            score=self._game.score,
            cleared_lines=self._game.cleared_lines,
            board=self._game.read_board(),
        )

    def _build_episode_info(self, snapshot: TetrisSnapshot) -> EpisodeInfo:
        return {
            "score": snapshot.score,
            "cleared_lines": snapshot.cleared_lines,
            "level": self._game.level,
            "aggregate_height": snapshot.board.aggregate_height,
            "holes": snapshot.board.holes,
            "bumpiness": snapshot.board.bumpiness,
            "episode_steps": self._episode_step_count,
            "current_piece": self._game.current_piece,
            "next_piece": self._game.next_piece,
            "held_piece": self._game.held_piece,
        }

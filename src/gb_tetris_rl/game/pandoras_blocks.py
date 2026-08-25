from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from pyboy import PyBoy

from gb_tetris_rl.game.contracts import BOARD_SHAPE


class PandorasBlocksAdapter:
    """Translates Pandora's Blocks memory and controls into game-level operations."""

    _SHADOW_FIELD_ADDRESS = 0xCA9F
    _SHADOW_FIELD_ROWS = 24
    _SHADOW_FIELD_ROW_STRIDE = 14
    _SHADOW_FIELD_LEFT_BORDER_WIDTH = 2
    _EMPTY_TILE = 108
    _GHOST_TILE = 107

    _SCORE_ADDRESS = 0xFFB8
    _SCORE_DIGIT_COUNT = 8
    _LINE_CLEAR_COUNT_ADDRESS = 0xFFB0
    _STALE_PIECE_ADDRESS = 0xFFB5
    _LEVEL_ADDRESS = 0xFF90
    _LEVEL_DIGIT_COUNT = 4
    _RNG_SEED_ADDRESS = 0xFFCA

    _GAME_STATE_ADDRESS = 0xFFFE
    _GAMEPLAY_STATE = 3
    _MODE_ADDRESS = 0xFFE5
    _CURRENT_PIECE_ADDRESS = 0xFFDF
    _NEXT_PIECE_ADDRESS = 0xFFD2
    _HELD_PIECE_ADDRESS = 0xFFE3
    _HOLD_SPENT_ADDRESS = 0xFFE4
    _HOLD_SPENT_VALUE = 0xFF
    _PIECE_IN_MOTION_MODE = 15
    _GAME_OVER_MODES = frozenset({21, 24})

    _DROP_MODE_ADDRESS = 0xCF3A
    _HARD_DROP_MODE = 2
    _SPEED_CURVE_ADDRESS = 0xCF3B
    _CHILL_SPEED_CURVE = 5

    _LEFT_WALL_MOVE_COUNT = 10
    _MAXIMUM_STARTUP_FRAMES = 600
    _MAXIMUM_SPAWN_WAIT_FRAMES = 240
    _MAXIMUM_HOLD_WAIT_FRAMES = 120
    _HOLD_RETRY_INTERVAL_FRAMES = 8

    def __init__(self, pyboy: PyBoy) -> None:
        self._pyboy = pyboy
        self._cleared_line_total = 0
        self._current_clear_was_counted = False
        self._boot_into_gameplay()

        initial_state_buffer = BytesIO()
        self._pyboy.save_state(initial_state_buffer)
        self._initial_state_bytes = initial_state_buffer.getvalue()

    def reset(self, seed: int | None) -> None:
        self._pyboy.load_state(BytesIO(self._initial_state_bytes))
        if seed is not None:
            normalized_seed = seed & 0xFFFFFFFF
            for byte_offset in range(4):
                seed_byte = (normalized_seed >> (byte_offset * 8)) & 0xFF
                self._pyboy.memory[self._RNG_SEED_ADDRESS + byte_offset] = seed_byte

        self._cleared_line_total = 0
        self._current_clear_was_counted = False

    def read_board(self) -> NDArray[np.uint8]:
        """Read locked blocks without including the active or ghost piece."""
        shadow_field_cell_count = self._SHADOW_FIELD_ROWS * self._SHADOW_FIELD_ROW_STRIDE
        shadow_field_tiles = np.asarray(
            self._pyboy.memory[
                self._SHADOW_FIELD_ADDRESS : self._SHADOW_FIELD_ADDRESS + shadow_field_cell_count
            ],
            dtype=np.uint8,
        ).reshape(self._SHADOW_FIELD_ROWS, self._SHADOW_FIELD_ROW_STRIDE)
        settled_field_tiles = shadow_field_tiles[
            :,
            self._SHADOW_FIELD_LEFT_BORDER_WIDTH : self._SHADOW_FIELD_LEFT_BORDER_WIDTH
            + BOARD_SHAPE[1],
        ]
        visible_field_tiles = settled_field_tiles[-BOARD_SHAPE[0] :]
        occupied_cells = (
            (visible_field_tiles != self._EMPTY_TILE)
            & (visible_field_tiles != self._GHOST_TILE)
            & (visible_field_tiles != 0)
        )
        return occupied_cells.astype(np.uint8)

    def update_after_tick(self) -> None:
        current_line_clear_count = int(self._pyboy.memory[self._LINE_CLEAR_COUNT_ADDRESS])
        if current_line_clear_count > 0 and not self._current_clear_was_counted:
            self._cleared_line_total += current_line_clear_count
            self._current_clear_was_counted = True
        elif current_line_clear_count == 0:
            self._current_clear_was_counted = False

    def place_piece(
        self,
        target_rotation: int,
        right_moves_from_left_wall: int,
        *,
        use_hold: bool,
        render_frames: bool,
    ) -> bool:
        if not self._piece_is_in_motion:
            raise RuntimeError("cannot place a piece outside piece-in-motion mode")

        emulator_is_running = True
        if use_hold:
            emulator_is_running = self._apply_hold(render_frames)
            if self.game_is_over or not emulator_is_running:
                return emulator_is_running

        if int(self._pyboy.memory[self._STALE_PIECE_ADDRESS]) == 0:
            emulator_is_running = self._tick_frame(render_frames)

        for _ in range(target_rotation):
            emulator_is_running = self._press_button("b", render_frames) and emulator_is_running

        for _ in range(self._LEFT_WALL_MOVE_COUNT):
            emulator_is_running = self._press_button("left", render_frames) and emulator_is_running

        for _ in range(right_moves_from_left_wall):
            emulator_is_running = self._press_button("right", render_frames) and emulator_is_running

        self._pyboy.button("up")
        emulator_is_running = self._tick_frame(render_frames) and emulator_is_running
        piece_has_locked = not self._piece_is_in_motion

        for _ in range(self._MAXIMUM_SPAWN_WAIT_FRAMES):
            if self.game_is_over or not emulator_is_running:
                return emulator_is_running
            if piece_has_locked and self._piece_is_in_motion:
                return emulator_is_running

            emulator_is_running = self._tick_frame(render_frames)
            piece_has_locked = piece_has_locked or not self._piece_is_in_motion

        raise RuntimeError("Pandora's Blocks did not spawn the next piece after a hard drop")

    @property
    def score(self) -> int:
        return self._read_decimal_digits(self._SCORE_ADDRESS, self._SCORE_DIGIT_COUNT)

    @property
    def cleared_lines(self) -> int:
        return self._cleared_line_total

    @property
    def level(self) -> int:
        return self._read_decimal_digits(self._LEVEL_ADDRESS, self._LEVEL_DIGIT_COUNT)

    @property
    def game_is_over(self) -> bool:
        return int(self._pyboy.memory[self._MODE_ADDRESS]) in self._GAME_OVER_MODES

    @property
    def current_piece(self) -> int:
        return int(self._pyboy.memory[self._CURRENT_PIECE_ADDRESS])

    @property
    def next_piece(self) -> int:
        return int(self._pyboy.memory[self._NEXT_PIECE_ADDRESS])

    @property
    def held_piece(self) -> int:
        return int(self._pyboy.memory[self._HELD_PIECE_ADDRESS])

    @property
    def _piece_is_in_motion(self) -> bool:
        return int(self._pyboy.memory[self._MODE_ADDRESS]) == self._PIECE_IN_MOTION_MODE

    def _boot_into_gameplay(self) -> None:
        self._pyboy.tick(180, render=True, sound=False)
        self._pyboy.memory[self._DROP_MODE_ADDRESS] = self._HARD_DROP_MODE
        self._pyboy.memory[self._SPEED_CURVE_ADDRESS] = self._CHILL_SPEED_CURVE
        self._pyboy.button("start")

        for _ in range(self._MAXIMUM_STARTUP_FRAMES):
            self._pyboy.tick(1, render=True, sound=False)
            game_state = int(self._pyboy.memory[self._GAME_STATE_ADDRESS])
            if game_state == self._GAMEPLAY_STATE and self._piece_is_in_motion:
                return
        raise RuntimeError("Pandora's Blocks did not reach gameplay during startup")

    def _apply_hold(self, render_frames: bool) -> bool:
        emulator_is_running = True
        for hold_wait_frame in range(self._MAXIMUM_HOLD_WAIT_FRAMES):
            hold_was_applied = (
                int(self._pyboy.memory[self._HOLD_SPENT_ADDRESS]) == self._HOLD_SPENT_VALUE
            )
            if hold_was_applied and self._piece_is_in_motion:
                return emulator_is_running
            if self.game_is_over or not emulator_is_running:
                return emulator_is_running

            should_retry_hold = (
                hold_wait_frame % self._HOLD_RETRY_INTERVAL_FRAMES == 0 and self._piece_is_in_motion
            )
            if should_retry_hold:
                emulator_is_running = self._press_button("select", render_frames)
            else:
                emulator_is_running = self._tick_frame(render_frames)

        raise RuntimeError("Pandora's Blocks did not finish the hold action")

    def _read_decimal_digits(self, start_address: int, digit_count: int) -> int:
        digits = self._pyboy.memory[start_address : start_address + digit_count]
        return sum(
            int(digit) * (10 ** (digit_count - digit_index - 1))
            for digit_index, digit in enumerate(digits)
        )

    def _tick_frame(self, render_frame: bool) -> bool:
        emulator_is_running = self._pyboy.tick(1, render=render_frame, sound=False)
        self.update_after_tick()
        return emulator_is_running

    def _press_button(self, button_name: str, render_frames: bool) -> bool:
        self._pyboy.button(button_name)
        press_frame_is_running = self._tick_frame(render_frames)
        release_frame_is_running = self._tick_frame(render_frames)
        return press_frame_is_running and release_frame_is_running

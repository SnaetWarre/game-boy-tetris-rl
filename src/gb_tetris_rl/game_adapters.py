from io import BytesIO
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pyboy import PyBoy

from gb_tetris_rl.roms import SupportedGame

TETRIS_BOARD_SHAPE = (18, 10)


class TetrisGameAdapter(Protocol):
    def reset(self, seed: int | None) -> None: ...

    def read_board(self) -> NDArray[np.uint8]: ...

    def update_after_tick(self) -> None: ...

    def place_piece(
        self,
        target_rotation: int,
        right_moves_from_left_wall: int,
        *,
        use_hold: bool,
        render_frames: bool,
    ) -> bool: ...

    @property
    def score(self) -> int: ...

    @property
    def cleared_lines(self) -> int: ...

    @property
    def level(self) -> int: ...

    @property
    def game_is_over(self) -> bool: ...

    @property
    def current_piece(self) -> int: ...

    @property
    def next_piece(self) -> int: ...

    @property
    def held_piece(self) -> int: ...


class NintendoTetrisAdapter:
    def __init__(self, pyboy: PyBoy) -> None:
        self._wrapper = pyboy.game_wrapper
        self._wrapper.game_area_mapping(self._wrapper.mapping_minimal, 0)
        self._wrapper.start_game(timer_div=0)

    def reset(self, seed: int | None) -> None:
        timer_divider = None if seed is None else seed % 256
        self._wrapper.reset_game(timer_div=timer_divider)

    def read_board(self) -> NDArray[np.uint8]:
        board = np.asarray(self._wrapper.game_area(), dtype=np.uint8)
        if board.shape != TETRIS_BOARD_SHAPE:
            raise RuntimeError(
                f"PyBoy returned board shape {board.shape}; expected {TETRIS_BOARD_SHAPE}"
            )
        return np.array(board, dtype=np.uint8, copy=True)

    def update_after_tick(self) -> None:
        pass

    def place_piece(
        self,
        target_rotation: int,
        right_moves_from_left_wall: int,
        *,
        use_hold: bool,
        render_frames: bool,
    ) -> bool:
        del target_rotation, right_moves_from_left_wall, use_hold, render_frames
        raise RuntimeError("placement controls require Pandora's Blocks")

    @property
    def score(self) -> int:
        return int(self._wrapper.score)

    @property
    def cleared_lines(self) -> int:
        return int(self._wrapper.lines)

    @property
    def level(self) -> int:
        return int(self._wrapper.level)

    @property
    def game_is_over(self) -> bool:
        game_over_value = self._wrapper.game_over
        return bool(game_over_value() if callable(game_over_value) else game_over_value)

    @property
    def current_piece(self) -> int:
        return 0

    @property
    def next_piece(self) -> int:
        return 0

    @property
    def held_piece(self) -> int:
        return 7


class PandorasBlocksAdapter:
    """Reads the GPL game's named state from its source-matched WRAM layout."""

    _FIELD_ADDRESS = 0xC8A3
    _FIELD_CELL_COUNT = 24 * 10
    _SHADOW_FIELD_ADDRESS = 0xCA9F
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
    _MAXIMUM_SPAWN_WAIT_FRAMES = 240
    _MAXIMUM_HOLD_WAIT_FRAMES = 120

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
                self._pyboy.memory[self._RNG_SEED_ADDRESS + byte_offset] = (
                    normalized_seed >> (byte_offset * 8)
                ) & 0xFF
        self._cleared_line_total = 0
        self._current_clear_was_counted = False

    def read_board(self) -> NDArray[np.uint8]:
        # Pandora's collision shadow contains only locked blocks. The regular
        # field is unsuitable for placement-level observations because it also
        # contains the active falling piece on rendered frames.
        shadow_field_tiles = np.asarray(
            self._pyboy.memory[
                self._SHADOW_FIELD_ADDRESS : self._SHADOW_FIELD_ADDRESS
                + (24 * self._SHADOW_FIELD_ROW_STRIDE)
            ],
            dtype=np.uint8,
        ).reshape(24, self._SHADOW_FIELD_ROW_STRIDE)
        settled_field_tiles = shadow_field_tiles[
            :,
            self._SHADOW_FIELD_LEFT_BORDER_WIDTH : self._SHADOW_FIELD_LEFT_BORDER_WIDTH
            + TETRIS_BOARD_SHAPE[1],
        ]
        visible_field_tiles = settled_field_tiles[-TETRIS_BOARD_SHAPE[0] :]
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
        if int(self._pyboy.memory[self._MODE_ADDRESS]) != self._PIECE_IN_MOTION_MODE:
            raise RuntimeError("cannot place a piece outside piece-in-motion mode")

        emulator_is_running = True
        if use_hold:
            for hold_wait_frame in range(self._MAXIMUM_HOLD_WAIT_FRAMES):
                hold_was_applied = (
                    int(self._pyboy.memory[self._HOLD_SPENT_ADDRESS])
                    == self._HOLD_SPENT_VALUE
                )
                piece_is_in_motion = (
                    int(self._pyboy.memory[self._MODE_ADDRESS])
                    == self._PIECE_IN_MOTION_MODE
                )
                if hold_was_applied and piece_is_in_motion:
                    break
                if self.game_is_over or not emulator_is_running:
                    return emulator_is_running
                should_retry_hold_input = hold_wait_frame % 8 == 0 and piece_is_in_motion
                if should_retry_hold_input:
                    emulator_is_running = self._press_button("select", render_frames)
                else:
                    emulator_is_running = self._tick_frame(render_frames)
            else:
                raise RuntimeError("Pandora's Blocks did not finish the hold action")

        if int(self._pyboy.memory[self._STALE_PIECE_ADDRESS]) == 0:
            emulator_is_running = self._tick_frame(render_frames)

        for _ in range(target_rotation):
            emulator_is_running = (
                self._press_button("b", render_frames) and emulator_is_running
            )

        for _ in range(self._LEFT_WALL_MOVE_COUNT):
            emulator_is_running = (
                self._press_button("left", render_frames) and emulator_is_running
            )

        for _ in range(right_moves_from_left_wall):
            emulator_is_running = (
                self._press_button("right", render_frames) and emulator_is_running
            )

        self._pyboy.button("up")
        emulator_is_running = self._tick_frame(render_frames) and emulator_is_running
        piece_has_locked = int(self._pyboy.memory[self._MODE_ADDRESS]) != self._PIECE_IN_MOTION_MODE

        for _ in range(self._MAXIMUM_SPAWN_WAIT_FRAMES):
            if self.game_is_over or not emulator_is_running:
                return emulator_is_running
            if (
                piece_has_locked
                and int(self._pyboy.memory[self._MODE_ADDRESS]) == self._PIECE_IN_MOTION_MODE
            ):
                return emulator_is_running

            emulator_is_running = self._tick_frame(render_frames)
            piece_has_locked = piece_has_locked or (
                int(self._pyboy.memory[self._MODE_ADDRESS]) != self._PIECE_IN_MOTION_MODE
            )

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

    def _boot_into_gameplay(self) -> None:
        self._pyboy.tick(180, render=True, sound=False)
        self._pyboy.memory[self._DROP_MODE_ADDRESS] = self._HARD_DROP_MODE
        self._pyboy.memory[self._SPEED_CURVE_ADDRESS] = self._CHILL_SPEED_CURVE
        self._pyboy.button("start")

        for _ in range(600):
            self._pyboy.tick(1, render=True, sound=False)
            game_state = int(self._pyboy.memory[self._GAME_STATE_ADDRESS])
            gameplay_mode = int(self._pyboy.memory[self._MODE_ADDRESS])
            if (
                game_state == self._GAMEPLAY_STATE
                and gameplay_mode == self._PIECE_IN_MOTION_MODE
            ):
                return
        raise RuntimeError("Pandora's Blocks did not reach gameplay during startup")

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


def create_game_adapter(game: SupportedGame, pyboy: PyBoy) -> TetrisGameAdapter:
    if game is SupportedGame.PANDORAS_BLOCKS:
        return PandorasBlocksAdapter(pyboy)
    return NintendoTetrisAdapter(pyboy)

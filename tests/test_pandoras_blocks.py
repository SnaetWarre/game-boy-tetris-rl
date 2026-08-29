import unittest

import numpy as np

from gb_tetris_rl.game.contracts import BOARD_SHAPE
from gb_tetris_rl.game.pandoras_blocks import PandorasBlocksAdapter


class FakeMemoryPyBoy:
    def __init__(self) -> None:
        self.memory = bytearray(0x10000)


class TopOutPyBoy(FakeMemoryPyBoy):
    def __init__(self) -> None:
        super().__init__()
        self.pressed_buttons: set[str] = set()
        self.button_events: list[tuple[str, str]] = []
        self.tick_gravity_values: list[int] = []

    def button_press(self, button_name: str) -> None:
        self.pressed_buttons.add(button_name)
        self.button_events.append(("press", button_name))

    def button_release(self, button_name: str) -> None:
        self.pressed_buttons.discard(button_name)
        self.button_events.append(("release", button_name))

    def tick(self, frame_count: int, *, render: bool, sound: bool) -> bool:
        del frame_count, render, sound
        self.tick_gravity_values.append(
            int(self.memory[PandorasBlocksAdapter._INTEGER_GRAVITY_ADDRESS])
        )
        if "up" in self.pressed_buttons:
            self.memory[PandorasBlocksAdapter._MODE_ADDRESS] = 24
        return True


class MovingPiecePyBoy(FakeMemoryPyBoy):
    def __init__(self) -> None:
        super().__init__()
        self.pressed_buttons: set[str] = set()
        self.button_events: list[tuple[str, str]] = []

    def button_press(self, button_name: str) -> None:
        self.pressed_buttons.add(button_name)
        self.button_events.append(("press", button_name))

    def button_release(self, button_name: str) -> None:
        self.pressed_buttons.discard(button_name)
        self.button_events.append(("release", button_name))

    def tick(self, frame_count: int, *, render: bool, sound: bool) -> bool:
        del frame_count, render, sound
        if "left" in self.pressed_buttons:
            self.memory[PandorasBlocksAdapter._CURRENT_PIECE_X_ADDRESS] -= 1
        if "right" in self.pressed_buttons:
            self.memory[PandorasBlocksAdapter._CURRENT_PIECE_X_ADDRESS] += 1
        return True


def create_adapter_for_memory_tests() -> tuple[PandorasBlocksAdapter, FakeMemoryPyBoy]:
    fake_pyboy = FakeMemoryPyBoy()
    adapter = PandorasBlocksAdapter.__new__(PandorasBlocksAdapter)
    adapter._pyboy = fake_pyboy
    adapter._cleared_line_total = 0
    adapter._current_clear_was_counted = False
    return adapter, fake_pyboy


class PandorasBlocksMemoryTests(unittest.TestCase):
    def test_converts_source_field_tiles_to_binary_board(self) -> None:
        adapter, fake_pyboy = create_adapter_for_memory_tests()
        shadow_start = adapter._SHADOW_FIELD_ADDRESS
        shadow_size = adapter._SHADOW_FIELD_ROWS * adapter._SHADOW_FIELD_ROW_STRIDE
        fake_pyboy.memory[shadow_start : shadow_start + shadow_size] = bytes(
            [adapter._EMPTY_TILE] * shadow_size
        )
        first_visible_cell = (
            shadow_start
            + (6 * adapter._SHADOW_FIELD_ROW_STRIDE)
            + adapter._SHADOW_FIELD_LEFT_BORDER_WIDTH
        )
        fake_pyboy.memory[first_visible_cell] = 48
        fake_pyboy.memory[first_visible_cell + 1] = adapter._GHOST_TILE

        board = adapter.read_board()

        self.assertEqual(board.shape, BOARD_SHAPE)
        self.assertEqual(board.dtype, np.uint8)
        self.assertEqual(board[0, 0], 1)
        self.assertEqual(board[0, 1], 0)
        self.assertEqual(int(board.sum()), 1)

    def test_counts_each_line_clear_event_once(self) -> None:
        adapter, fake_pyboy = create_adapter_for_memory_tests()
        line_clear_address = adapter._LINE_CLEAR_COUNT_ADDRESS

        fake_pyboy.memory[line_clear_address] = 2
        adapter.update_after_tick()
        adapter.update_after_tick()
        fake_pyboy.memory[line_clear_address] = 0
        adapter.update_after_tick()
        fake_pyboy.memory[line_clear_address] = 1
        adapter.update_after_tick()

        self.assertEqual(adapter.cleared_lines, 3)

    def test_reads_decimal_score_digits_and_game_over_mode(self) -> None:
        adapter, fake_pyboy = create_adapter_for_memory_tests()
        fake_pyboy.memory[
            adapter._SCORE_ADDRESS : adapter._SCORE_ADDRESS + adapter._SCORE_DIGIT_COUNT
        ] = bytes([0, 0, 0, 1, 2, 3, 4, 5])
        fake_pyboy.memory[adapter._MODE_ADDRESS] = 21

        self.assertEqual(adapter.score, 12_345)
        self.assertTrue(adapter.game_is_over)

    def test_releases_hard_drop_button_when_the_piece_tops_out(self) -> None:
        fake_pyboy = TopOutPyBoy()
        adapter = PandorasBlocksAdapter.__new__(PandorasBlocksAdapter)
        adapter._pyboy = fake_pyboy
        adapter._cleared_line_total = 0
        adapter._current_clear_was_counted = False
        fake_pyboy.memory[adapter._MODE_ADDRESS] = adapter._PIECE_IN_MOTION_MODE
        fake_pyboy.memory[adapter._STALE_PIECE_ADDRESS] = 1

        emulator_is_running = adapter.place_piece(
            target_rotation=0,
            target_left_column=0,
            use_hold=False,
            render_frames=False,
        )

        self.assertTrue(emulator_is_running)
        self.assertNotIn("up", fake_pyboy.pressed_buttons)
        self.assertIn(("release", "up"), fake_pyboy.button_events)

    def test_moves_directly_from_the_spawn_column_to_the_target(self) -> None:
        fake_pyboy = MovingPiecePyBoy()
        adapter = PandorasBlocksAdapter.__new__(PandorasBlocksAdapter)
        adapter._pyboy = fake_pyboy
        adapter._cleared_line_total = 0
        adapter._current_clear_was_counted = False
        fake_pyboy.memory[adapter._MODE_ADDRESS] = adapter._PIECE_IN_MOTION_MODE
        fake_pyboy.memory[adapter._CURRENT_PIECE_ADDRESS] = 0
        fake_pyboy.memory[adapter._CURRENT_PIECE_X_ADDRESS] = 5

        emulator_is_running = adapter._position_piece(
            target_rotation=1,
            target_left_column=9,
            render_frames=False,
        )

        self.assertTrue(emulator_is_running)
        self.assertEqual(fake_pyboy.memory[adapter._CURRENT_PIECE_X_ADDRESS], 9)
        self.assertEqual(fake_pyboy.button_events.count(("press", "b")), 1)
        self.assertEqual(fake_pyboy.button_events.count(("press", "right")), 4)
        self.assertLess(
            fake_pyboy.button_events.index(("press", "b")),
            fake_pyboy.button_events.index(("release", "right")),
        )

    def test_restores_gravity_after_placement_input(self) -> None:
        fake_pyboy = TopOutPyBoy()
        adapter = PandorasBlocksAdapter.__new__(PandorasBlocksAdapter)
        adapter._pyboy = fake_pyboy
        adapter._cleared_line_total = 0
        adapter._current_clear_was_counted = False
        fake_pyboy.memory[adapter._MODE_ADDRESS] = adapter._PIECE_IN_MOTION_MODE
        fake_pyboy.memory[adapter._STALE_PIECE_ADDRESS] = 1
        fake_pyboy.memory[adapter._CURRENT_PIECE_X_ADDRESS] = 5
        fake_pyboy.memory[adapter._CURRENT_PIECE_Y_ADDRESS] = 12
        fake_pyboy.memory[adapter._INTEGER_GRAVITY_ADDRESS] = 3
        fake_pyboy.memory[adapter._FRACTIONAL_GRAVITY_ADDRESS] = 17

        adapter.place_piece(
            target_rotation=1,
            target_left_column=9,
            use_hold=False,
            render_frames=False,
        )

        self.assertEqual(fake_pyboy.memory[adapter._INTEGER_GRAVITY_ADDRESS], 3)
        self.assertEqual(fake_pyboy.memory[adapter._FRACTIONAL_GRAVITY_ADDRESS], 17)
        self.assertIn(0, fake_pyboy.tick_gravity_values)
        self.assertEqual(fake_pyboy.tick_gravity_values[-1], 3)
        self.assertEqual(
            fake_pyboy.memory[adapter._CURRENT_PIECE_Y_ADDRESS],
            adapter._PIECE_SPAWN_Y,
        )


if __name__ == "__main__":
    unittest.main()

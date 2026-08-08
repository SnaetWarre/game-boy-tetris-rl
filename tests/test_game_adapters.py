import unittest

import numpy as np

from gb_tetris_rl.game_adapters import TETRIS_BOARD_SHAPE, PandorasBlocksAdapter


class FakeMemoryPyBoy:
    def __init__(self) -> None:
        self.memory = bytearray(0x10000)


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
        field_start = adapter._FIELD_ADDRESS
        field_end = field_start + adapter._FIELD_CELL_COUNT
        fake_pyboy.memory[field_start:field_end] = bytes(
            [adapter._EMPTY_TILE] * adapter._FIELD_CELL_COUNT
        )
        first_visible_cell = field_start + (6 * TETRIS_BOARD_SHAPE[1])
        fake_pyboy.memory[first_visible_cell] = 48
        fake_pyboy.memory[first_visible_cell + 1] = adapter._GHOST_TILE

        board = adapter.read_board()

        self.assertEqual(board.shape, TETRIS_BOARD_SHAPE)
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


if __name__ == "__main__":
    unittest.main()

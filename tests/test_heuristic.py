import unittest

import numpy as np

from gb_tetris_rl.heuristic import (
    PANDORAS_PIECE_ROTATIONS,
    choose_placement_action,
    enumerate_placements,
    simulate_hard_drop,
)


class PlacementHeuristicTests(unittest.TestCase):
    def test_horizontal_i_piece_clears_bottom_row(self) -> None:
        board = np.zeros((18, 10), dtype=np.uint8)
        board[-1, :6] = 1

        board_after_drop = simulate_hard_drop(
            board,
            PANDORAS_PIECE_ROTATIONS[0][0],
            6,
        )

        self.assertIsNotNone(board_after_drop)
        assert board_after_drop is not None
        self.assertEqual(int(board_after_drop.sum()), 0)

    def test_enumeration_removes_duplicate_piece_rotations(self) -> None:
        board = np.zeros((18, 10), dtype=np.uint8)

        i_piece_placements = enumerate_placements(board, 0)
        o_piece_placements = enumerate_placements(board, 5)

        self.assertEqual(len(i_piece_placements), 17)
        self.assertEqual(len(o_piece_placements), 9)

    def test_planner_takes_available_line_clear(self) -> None:
        board = np.zeros((18, 10), dtype=np.uint8)
        board[-1, :6] = 1

        action = choose_placement_action(board, current_piece=0)

        self.assertEqual(action, 6)


if __name__ == "__main__":
    unittest.main()

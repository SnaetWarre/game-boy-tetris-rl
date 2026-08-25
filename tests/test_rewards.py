import unittest

import numpy as np

from gb_tetris_rl.game.rewards import (
    TetrisSnapshot,
    calculate_transition_reward,
    create_snapshot,
    measure_board,
)


class BoardMeasurementTests(unittest.TestCase):
    def test_empty_board_has_zero_measurements(self) -> None:
        empty_board = np.zeros((18, 10), dtype=np.uint8)

        measurements = measure_board(empty_board)

        self.assertEqual(measurements.aggregate_height, 0)
        self.assertEqual(measurements.holes, 0)
        self.assertEqual(measurements.bumpiness, 0)
        self.assertEqual(measurements.completed_rows, 0)

    def test_measurements_capture_holes_height_bumpiness_and_full_rows(self) -> None:
        board = np.zeros((6, 4), dtype=np.uint8)
        board[-1, :] = 1
        board[-2, 0] = 1
        board[-3, 0] = 1
        board[-3, 2] = 1

        measurements = measure_board(board)

        self.assertEqual(measurements.aggregate_height, 8)
        self.assertEqual(measurements.holes, 1)
        self.assertEqual(measurements.bumpiness, 6)
        self.assertEqual(measurements.completed_rows, 1)

    def test_non_matrix_board_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            measure_board(np.zeros(10, dtype=np.uint8))


class TransitionRewardTests(unittest.TestCase):
    def test_clearing_lines_is_rewarded(self) -> None:
        board = np.zeros((18, 10), dtype=np.uint8)
        previous_snapshot = create_snapshot(score=0, cleared_lines=0, board=board)
        current_snapshot = create_snapshot(score=1200, cleared_lines=4, board=board)

        reward = calculate_transition_reward(
            previous_snapshot,
            current_snapshot,
            game_is_over=False,
        )

        self.assertGreater(reward, 8.0)

    def test_game_over_has_a_strong_penalty(self) -> None:
        board = np.zeros((18, 10), dtype=np.uint8)
        snapshot: TetrisSnapshot = create_snapshot(score=0, cleared_lines=0, board=board)

        reward = calculate_transition_reward(snapshot, snapshot, game_is_over=True)

        self.assertLess(reward, -4.9)


if __name__ == "__main__":
    unittest.main()

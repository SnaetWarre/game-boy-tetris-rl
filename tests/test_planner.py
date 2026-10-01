import unittest

import numpy as np
import torch

from gb_tetris_rl.agent.planner import (
    LOOKAHEAD_WEIGHT,
    PANDORAS_PIECE_ROTATIONS,
    TOP_OUT_LOOKAHEAD_QUALITY,
    boards_to_columns,
    choose_agent_action,
    choose_agent_actions,
    columns_to_boards,
    drop_pieces,
    enumerate_agent_afterstates,
    score_boards,
)
from gb_tetris_rl.game.contracts import EMPTY_HOLD_SLOT
from gb_tetris_rl.game.rewards import measure_board

HIDDEN_ROWS = 4


def reference_hard_drop(board, piece_shape, left_column):
    """Readable scalar reference: the piece falls from hidden rows above the field."""
    row_count, column_count = board.shape
    piece_width = max(column_offset for _, column_offset in piece_shape) + 1
    if left_column + piece_width > column_count:
        return None
    padded_board = np.vstack((np.zeros((HIDDEN_ROWS, column_count), np.uint8), board))

    def overlaps(top_row):
        return any(
            top_row + row_offset >= padded_board.shape[0]
            or padded_board[top_row + row_offset, left_column + column_offset]
            for row_offset, column_offset in piece_shape
        )

    landing_row = 0
    while not overlaps(landing_row + 1):
        landing_row += 1
    if landing_row < HIDDEN_ROWS:
        return None
    for row_offset, column_offset in piece_shape:
        padded_board[landing_row + row_offset, left_column + column_offset] = 1
    board_after_drop = padded_board[HIDDEN_ROWS:]
    uncleared_rows = board_after_drop[~board_after_drop.all(axis=1)]
    cleared_line_count = row_count - len(uncleared_rows)
    return (
        np.vstack((np.zeros((cleared_line_count, column_count), np.uint8), uncleared_rows)),
        cleared_line_count,
    )


def reference_placements(board, piece):
    placements = []
    seen_shapes = set()
    for rotation, piece_shape in enumerate(PANDORAS_PIECE_ROTATIONS[piece]):
        if piece_shape in seen_shapes:
            continue
        seen_shapes.add(piece_shape)
        for left_column in range(board.shape[1]):
            result = reference_hard_drop(board, piece_shape, left_column)
            if result is not None:
                placements.append((rotation * board.shape[1] + left_column, *result))
    return placements


def reference_quality(board, cleared_line_count):
    measurements = measure_board(board)
    return (
        0.760666 * cleared_line_count
        - 0.510066 * measurements.aggregate_height
        - 0.35663 * measurements.holes
        - 0.184483 * measurements.bumpiness
    )


def reference_best(board, piece, next_piece):
    best_action, best_quality = 0, float("-inf")
    for action, board_after, cleared_line_count in reference_placements(board, piece):
        quality = reference_quality(board_after, cleared_line_count)
        if next_piece is not None:
            next_qualities = [
                reference_quality(next_board, next_cleared)
                for _, next_board, next_cleared in reference_placements(board_after, next_piece)
            ]
            quality += LOOKAHEAD_WEIGHT * max(next_qualities, default=TOP_OUT_LOOKAHEAD_QUALITY)
        if quality > best_quality:
            best_action, best_quality = action, quality
    return best_action, best_quality


def reference_agent_action(board, current_piece, next_piece, held_piece, use_lookahead):
    lookahead_piece = next_piece if use_lookahead and held_piece != EMPTY_HOLD_SLOT else None
    direct_action, direct_quality = reference_best(board, current_piece, lookahead_piece)
    piece_after_hold = next_piece if held_piece == EMPTY_HOLD_SLOT else held_piece
    hold_action, hold_quality = reference_best(board, piece_after_hold, lookahead_piece)
    return 40 + hold_action if hold_quality > direct_quality else direct_action


def pack(boards):
    return boards_to_columns(torch.as_tensor(np.asarray(boards) != 0))


def random_boards(board_count, seed):
    random_generator = np.random.default_rng(seed)
    boards = np.zeros((board_count, 18, 10), dtype=np.uint8)
    for board in boards:
        stack_height = int(random_generator.integers(0, 19))
        fill_rate = random_generator.uniform(0.3, 0.95)
        if stack_height:
            board[-stack_height:] = random_generator.random((stack_height, 10)) < fill_rate
        # Settled boards never keep a full row.
        board[board.all(axis=1), int(random_generator.integers(10))] = 0
    return boards


class VectorizedPlacementTests(unittest.TestCase):
    def test_matches_scalar_reference_for_every_piece_and_action(self) -> None:
        boards = random_boards(200, seed=1)
        for piece in range(7):
            with self.subTest(piece=piece):
                dropped_columns, cleared_lines, valid = drop_pieces(
                    pack(boards),
                    torch.full((len(boards),), piece),
                )
                for board_index, board in enumerate(boards):
                    expected = {
                        action: (board_after, cleared)
                        for action, board_after, cleared in reference_placements(board, piece)
                    }
                    self.assertEqual(
                        set(np.flatnonzero(valid[board_index].numpy())),
                        set(expected),
                    )
                    for action, (board_after, cleared) in expected.items():
                        np.testing.assert_array_equal(
                            columns_to_boards(dropped_columns[board_index, action]).numpy(),
                            board_after != 0,
                        )
                        self.assertEqual(int(cleared_lines[board_index, action]), cleared)

    def test_bitboards_round_trip_dense_boards(self) -> None:
        boards = random_boards(50, seed=4) != 0

        np.testing.assert_array_equal(columns_to_boards(pack(boards)).numpy(), boards)

    def test_board_score_matches_reference_measurements(self) -> None:
        boards = random_boards(100, seed=2)
        cleared_lines = np.arange(len(boards)) % 5
        scores = score_boards(pack(boards), torch.as_tensor(cleared_lines))

        for board, cleared, score in zip(boards, cleared_lines, scores, strict=True):
            self.assertEqual(float(score), reference_quality(board, int(cleared)))

    def test_batched_planner_matches_scalar_reference(self) -> None:
        random_generator = np.random.default_rng(3)
        boards = random_boards(120, seed=3)
        current_pieces = random_generator.integers(0, 7, len(boards))
        next_pieces = random_generator.integers(0, 7, len(boards))
        held_pieces = random_generator.integers(0, 8, len(boards))

        for use_lookahead in (False, True):
            with self.subTest(use_lookahead=use_lookahead):
                actions = choose_agent_actions(
                    pack(boards),
                    torch.as_tensor(current_pieces),
                    torch.as_tensor(next_pieces),
                    torch.as_tensor(held_pieces),
                    use_lookahead=use_lookahead,
                )
                expected_actions = [
                    reference_agent_action(
                        board, int(current), int(upcoming), int(held), use_lookahead
                    )
                    for board, current, upcoming, held in zip(
                        boards, current_pieces, next_pieces, held_pieces, strict=True
                    )
                ]
                self.assertEqual(actions.tolist(), expected_actions)

    def test_afterstates_cover_direct_and_hold_actions(self) -> None:
        board = pack(np.zeros((1, 18, 10), dtype=np.uint8))

        afterstates = enumerate_agent_afterstates(
            board,
            torch.tensor([5]),
            torch.tensor([0]),
            torch.tensor([EMPTY_HOLD_SLOT]),
        )

        self.assertEqual(tuple(afterstates.columns.shape), (1, 80, 10))
        self.assertEqual(int(afterstates.valid[0, :40].sum()), 9)
        self.assertEqual(int(afterstates.valid[0, 40:].sum()), 17)


class PlacementHeuristicTests(unittest.TestCase):
    def test_horizontal_i_piece_clears_bottom_row(self) -> None:
        board = np.zeros((1, 18, 10), dtype=np.bool_)
        board[0, -1, :6] = True

        dropped_columns, cleared_lines, valid = drop_pieces(pack(board), torch.tensor([0]))

        self.assertTrue(valid[0, 6])
        self.assertEqual(int(cleared_lines[0, 6]), 1)
        self.assertEqual(int(dropped_columns[0, 6].sum()), 0)

    def test_planner_takes_available_line_clear(self) -> None:
        board = np.zeros((18, 10), dtype=np.uint8)
        board[-1, :6] = 1

        agent_action = choose_agent_action(board, 0, 0, EMPTY_HOLD_SLOT)

        self.assertEqual(agent_action, 6)

    def test_hold_planner_can_choose_the_held_piece(self) -> None:
        board = np.zeros((18, 10), dtype=np.uint8)

        agent_action = choose_agent_action(
            board,
            current_piece=1,
            next_piece=0,
            held_piece=0,
        )

        self.assertGreaterEqual(agent_action, 40)
        self.assertLess(agent_action, 80)


if __name__ == "__main__":
    unittest.main()

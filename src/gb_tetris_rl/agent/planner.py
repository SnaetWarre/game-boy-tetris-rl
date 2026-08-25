from dataclasses import dataclass

import numpy as np

from gb_tetris_rl.game.contracts import (
    BOARD_COLUMNS,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    Board,
)
from gb_tetris_rl.game.rewards import measure_board

PieceShape = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SimulatedPlacement:
    placement_action: int
    board: Board
    cleared_line_count: int
    quality: float


# These states follow Pandora's GPL sPieceRotationStates table exactly. Empty
# padding from each 4x4 source state is removed because a left-wall slam makes
# the leftmost occupied cell column zero for every state.
PANDORAS_PIECE_ROTATIONS: tuple[tuple[PieceShape, ...], ...] = (
    (  # I
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
    ),
    (  # Z
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
    (  # S
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
    ),
    (  # J
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ),
    (  # L
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ),
    (  # O
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ),
    (  # T
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 0)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
    ),
)


def simulate_hard_drop(board: Board, piece_shape: PieceShape, left_column: int) -> Board | None:
    row_count, column_count = board.shape
    piece_width = max(column_offset for _, column_offset in piece_shape) + 1
    piece_height = max(row_offset for row_offset, _ in piece_shape) + 1
    if left_column < 0 or left_column + piece_width > column_count:
        return None

    landing_row = 0
    if _piece_overlaps_board(board, piece_shape, landing_row, left_column):
        return None

    while landing_row + piece_height < row_count and not _piece_overlaps_board(
        board,
        piece_shape,
        landing_row + 1,
        left_column,
    ):
        landing_row += 1

    board_after_drop = np.array(board, dtype=np.uint8, copy=True)
    for row_offset, column_offset in piece_shape:
        board_after_drop[landing_row + row_offset, left_column + column_offset] = 1

    completed_rows = board_after_drop.all(axis=1)
    completed_row_count = int(np.count_nonzero(completed_rows))
    if completed_row_count == 0:
        return board_after_drop

    uncleared_rows = board_after_drop[~completed_rows]
    empty_rows = np.zeros((completed_row_count, column_count), dtype=np.uint8)
    return np.concatenate((empty_rows, uncleared_rows), axis=0)


def choose_placement_action(
    board: Board,
    current_piece: int,
    next_piece: int | None = None,
    *,
    lookahead_weight: float = 0.65,
) -> int:
    action, _ = _choose_action_with_quality(
        board,
        current_piece,
        next_piece,
        lookahead_weight=lookahead_weight,
    )
    return action


def choose_agent_action(
    board: Board,
    current_piece: int,
    next_piece: int,
    held_piece: int,
    *,
    use_lookahead: bool = True,
) -> int:
    comparable_normal_lookahead = (
        next_piece if use_lookahead and held_piece != EMPTY_HOLD_SLOT else None
    )
    normal_action, normal_quality = _choose_action_with_quality(
        board,
        current_piece,
        comparable_normal_lookahead,
    )

    piece_after_hold = next_piece if held_piece == EMPTY_HOLD_SLOT else held_piece
    hold_lookahead_piece = next_piece if use_lookahead and held_piece != EMPTY_HOLD_SLOT else None
    hold_action, hold_quality = _choose_action_with_quality(
        board,
        piece_after_hold,
        hold_lookahead_piece,
    )
    if hold_quality > normal_quality:
        return DIRECT_PLACEMENT_ACTION_COUNT + hold_action
    return normal_action


def _choose_action_with_quality(
    board: Board,
    piece: int,
    next_piece: int | None,
    *,
    lookahead_weight: float = 0.65,
) -> tuple[int, float]:
    current_placements = enumerate_placements(board, piece)
    if not current_placements:
        return 0, float("-inf")

    best_action = current_placements[0].placement_action
    best_action_quality = float("-inf")
    for current_placement in current_placements:
        action_quality = current_placement.quality
        if next_piece is not None and 0 <= next_piece < len(PANDORAS_PIECE_ROTATIONS):
            next_placements = enumerate_placements(current_placement.board, next_piece)
            if next_placements:
                action_quality += lookahead_weight * max(
                    next_placement.quality for next_placement in next_placements
                )
        if action_quality > best_action_quality:
            best_action = current_placement.placement_action
            best_action_quality = action_quality
    return best_action, best_action_quality


def enumerate_placements(board: Board, piece: int) -> list[SimulatedPlacement]:
    if not 0 <= piece < len(PANDORAS_PIECE_ROTATIONS):
        raise ValueError(f"unsupported Pandora's Blocks piece: {piece}")

    placements: list[SimulatedPlacement] = []
    seen_shapes: set[PieceShape] = set()
    for rotation, piece_shape in enumerate(PANDORAS_PIECE_ROTATIONS[piece]):
        if piece_shape in seen_shapes:
            continue
        seen_shapes.add(piece_shape)
        piece_width = max(column_offset for _, column_offset in piece_shape) + 1
        for left_column in range(board.shape[1] - piece_width + 1):
            board_after_drop = simulate_hard_drop(board, piece_shape, left_column)
            if board_after_drop is None:
                continue
            # Counting occupied rows is not a reliable line count, so compare
            # occupied cells: every cleared row removes exactly ten cells.
            cleared_line_count = (
                int(np.count_nonzero(board))
                + len(piece_shape)
                - int(np.count_nonzero(board_after_drop))
            ) // board.shape[1]
            placements.append(
                SimulatedPlacement(
                    placement_action=rotation * BOARD_COLUMNS + left_column,
                    board=board_after_drop,
                    cleared_line_count=cleared_line_count,
                    quality=_evaluate_board(board_after_drop, cleared_line_count),
                )
            )
    return placements


def _piece_overlaps_board(
    board: Board,
    piece_shape: PieceShape,
    top_row: int,
    left_column: int,
) -> bool:
    return any(
        board[top_row + row_offset, left_column + column_offset] != 0
        for row_offset, column_offset in piece_shape
    )


def _evaluate_board(board: Board, cleared_line_count: int) -> float:
    measurements = measure_board(board)
    return (
        0.760666 * cleared_line_count
        - 0.510066 * measurements.aggregate_height
        - 0.35663 * measurements.holes
        - 0.184483 * measurements.bumpiness
    )

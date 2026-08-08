from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Board = NDArray[np.uint8]


@dataclass(frozen=True)
class BoardMeasurements:
    aggregate_height: int
    holes: int
    bumpiness: int
    completed_rows: int


@dataclass(frozen=True)
class TetrisSnapshot:
    score: int
    cleared_lines: int
    board: BoardMeasurements


def measure_board(board: Board) -> BoardMeasurements:
    if board.ndim != 2:
        raise ValueError(f"expected a two-dimensional board, received shape {board.shape}")

    occupied_cells = board != 0
    row_count = occupied_cells.shape[0]
    occupied_columns = occupied_cells.any(axis=0)
    first_occupied_rows = np.argmax(occupied_cells, axis=0)
    column_heights = np.where(occupied_columns, row_count - first_occupied_rows, 0)

    cells_below_a_block = np.maximum.accumulate(occupied_cells, axis=0)
    hole_count = int(np.count_nonzero(cells_below_a_block & ~occupied_cells))
    height_differences = np.diff(column_heights)

    return BoardMeasurements(
        aggregate_height=int(column_heights.sum()),
        holes=hole_count,
        bumpiness=int(np.abs(height_differences).sum()),
        completed_rows=int(np.count_nonzero(occupied_cells.all(axis=1))),
    )


def create_snapshot(score: int, cleared_lines: int, board: Board) -> TetrisSnapshot:
    return TetrisSnapshot(
        score=score,
        cleared_lines=cleared_lines,
        board=measure_board(board),
    )


def calculate_transition_reward(
    previous_snapshot: TetrisSnapshot,
    current_snapshot: TetrisSnapshot,
    *,
    game_is_over: bool,
) -> float:
    cleared_line_count = max(0, current_snapshot.cleared_lines - previous_snapshot.cleared_lines)
    score_gain = max(0, current_snapshot.score - previous_snapshot.score)

    line_clear_reward_by_count = (0.0, 1.0, 3.0, 5.0, 8.0)
    line_clear_reward = line_clear_reward_by_count[min(cleared_line_count, 4)]

    removed_holes = previous_snapshot.board.holes - current_snapshot.board.holes
    reduced_height = (
        previous_snapshot.board.aggregate_height - current_snapshot.board.aggregate_height
    )
    reduced_bumpiness = previous_snapshot.board.bumpiness - current_snapshot.board.bumpiness

    board_quality_reward = 0.08 * removed_holes + 0.004 * reduced_height + 0.002 * reduced_bumpiness
    survival_reward = 0.001
    game_over_penalty = -5.0 if game_is_over else 0.0

    return float(
        line_clear_reward
        + 0.0005 * score_gain
        + board_quality_reward
        + survival_reward
        + game_over_penalty
    )

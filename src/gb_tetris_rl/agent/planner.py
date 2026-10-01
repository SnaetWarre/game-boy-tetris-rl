"""Bitboard Pandora's Blocks simulator and deterministic placement planner.

Boards are stored as one int32 per column: bit ``r`` is set when row ``r``
(counted from the top) is occupied. Every operation is batched, so the same
code labels demonstrations on a GPU and plans single emulator moves on a CPU.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from gb_tetris_rl.game.contracts import (
    BOARD_CELL_COUNT,
    BOARD_COLUMNS,
    BOARD_ROWS,
    BOARD_SHAPE,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    TETROMINO_TYPE_COUNT,
    Board,
)

PieceShape = tuple[tuple[int, int], ...]

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

LOOKAHEAD_WEIGHT = 0.65
# A placement that leaves the known next piece nowhere to go is a top-out, so
# its lookahead term must be far worse than any reachable board.
TOP_OUT_LOOKAHEAD_QUALITY = -10_000.0
# Lowest piece row for board columns a placement does not touch. It keeps
# those columns out of the landing-row minimum.
_UNTOUCHED_COLUMN_BOTTOM = -2 * BOARD_ROWS
_MAXIMUM_CLEARED_LINES = 4
COLUMN_DTYPE = torch.int32


@dataclass(frozen=True)
class AgentAfterstates:
    """Boards left behind by all 80 agent actions, after line clears."""

    columns: torch.Tensor  # [N, 80, columns] column bitboards
    cleared_lines: torch.Tensor  # [N, 80]
    valid: torch.Tensor  # [N, 80] bool: canonical and fits from above


@dataclass(frozen=True)
class ObservationBatch:
    """Agent observations decoded into bitboards and piece indices."""

    columns: torch.Tensor  # [N, columns]
    current_pieces: torch.Tensor  # [N]
    next_pieces: torch.Tensor  # [N]
    held_pieces: torch.Tensor  # [N]


@dataclass(frozen=True)
class _PlacementTables:
    canonical: torch.Tensor  # [pieces, 40] bool
    piece_columns: torch.Tensor  # [pieces, 40, columns] piece bits at landing row zero
    column_bottoms: torch.Tensor  # [pieces, 40, columns] lowest piece row per column
    top_rows: torch.Tensor  # [2**rows] first occupied row of a column, or the row count
    popcounts: torch.Tensor  # [2**rows] occupied cells in a column
    row_bits: torch.Tensor  # [rows] single-row masks


@cache
def _placement_tables(device: torch.device) -> _PlacementTables:
    action_shape = (TETROMINO_TYPE_COUNT, DIRECT_PLACEMENT_ACTION_COUNT)
    canonical = np.zeros(action_shape, dtype=np.bool_)
    piece_columns = np.zeros((*action_shape, BOARD_COLUMNS), dtype=np.int32)
    column_bottoms = np.full(
        (*action_shape, BOARD_COLUMNS),
        _UNTOUCHED_COLUMN_BOTTOM,
        dtype=np.int32,
    )
    for piece_type, piece_rotations in enumerate(PANDORAS_PIECE_ROTATIONS):
        seen_piece_shapes: set[PieceShape] = set()
        for rotation, piece_shape in enumerate(piece_rotations):
            piece_width = max(column_offset for _, column_offset in piece_shape) + 1
            for left_column in range(BOARD_COLUMNS):
                action = rotation * BOARD_COLUMNS + left_column
                if left_column + piece_width > BOARD_COLUMNS:
                    # Out-of-bounds actions borrow action zero's in-bounds
                    # geometry so batched indexing stays safe; they are masked.
                    piece_columns[piece_type, action] = piece_columns[piece_type, 0]
                    column_bottoms[piece_type, action] = column_bottoms[piece_type, 0]
                    continue
                canonical[piece_type, action] = piece_shape not in seen_piece_shapes
                for row_offset, column_offset in piece_shape:
                    board_column = left_column + column_offset
                    piece_columns[piece_type, action, board_column] |= 1 << row_offset
                    column_bottoms[piece_type, action, board_column] = max(
                        column_bottoms[piece_type, action, board_column],
                        row_offset,
                    )
            seen_piece_shapes.add(piece_shape)

    column_values = np.arange(1 << BOARD_ROWS, dtype=np.int64)
    lowest_bits = column_values & -column_values
    top_rows = np.where(
        column_values == 0,
        BOARD_ROWS,
        np.log2(np.maximum(lowest_bits, 1)).astype(np.int64),
    )
    popcounts = np.zeros(1 << BOARD_ROWS, dtype=np.int64)
    for row in range(BOARD_ROWS):
        popcounts += (column_values >> row) & 1
    return _PlacementTables(
        canonical=torch.as_tensor(canonical, device=device),
        piece_columns=torch.as_tensor(piece_columns, device=device),
        column_bottoms=torch.as_tensor(column_bottoms, device=device),
        top_rows=torch.as_tensor(top_rows, dtype=COLUMN_DTYPE, device=device),
        popcounts=torch.as_tensor(popcounts, dtype=COLUMN_DTYPE, device=device),
        row_bits=torch.as_tensor(1 << np.arange(BOARD_ROWS), dtype=COLUMN_DTYPE, device=device),
    )


def boards_to_columns(boards: torch.Tensor) -> torch.Tensor:
    """Pack dense [..., rows, columns] boards into [..., columns] bitboards."""
    row_bits = _placement_tables(boards.device).row_bits
    return (boards.to(COLUMN_DTYPE) * row_bits[:, None]).sum(dim=-2, dtype=COLUMN_DTYPE)


def columns_to_boards(columns: torch.Tensor) -> torch.Tensor:
    """Unpack [..., columns] bitboards into dense [..., rows, columns] bool boards."""
    row_bits = _placement_tables(columns.device).row_bits
    return (columns.unsqueeze(-2) & row_bits[:, None]) != 0


def decode_observations(observations: torch.Tensor) -> ObservationBatch:
    """Decode [N, 202] agent observations of any numeric dtype on any device."""
    next_piece_start = BOARD_CELL_COUNT + TETROMINO_TYPE_COUNT
    held_piece_start = next_piece_start + TETROMINO_TYPE_COUNT
    boards = observations[:, :BOARD_CELL_COUNT].reshape(-1, *BOARD_SHAPE) != 0
    return ObservationBatch(
        columns=boards_to_columns(boards),
        current_pieces=observations[:, BOARD_CELL_COUNT:next_piece_start].argmax(dim=1),
        next_pieces=observations[:, next_piece_start:held_piece_start].argmax(dim=1),
        held_pieces=observations[:, held_piece_start:].argmax(dim=1),
    )


def canonical_direct_action_masks(pieces: torch.Tensor) -> torch.Tensor:
    """Return [N, 40] masks of unique, in-bounds placements for each piece."""
    return _placement_tables(pieces.device).canonical[pieces]


def drop_pieces(
    columns: torch.Tensor,
    pieces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard-drop one piece per board at every rotation and column.

    Pieces fall from above the visible field, so a placement is valid only when
    it is canonical and every column it touches has room above the stack.
    Input boards must be settled, without full rows, as every game board is.
    Returns boards after line clears [N, 40, columns], cleared line counts
    [N, 40], and validity [N, 40].
    """
    tables = _placement_tables(columns.device)
    column_tops = tables.top_rows[columns]
    landing_rows = (column_tops[:, None, :] - 1 - tables.column_bottoms[pieces]).amin(dim=-1)
    valid = tables.canonical[pieces] & (landing_rows >= 0)
    placed_columns = columns[:, None, :] | (
        tables.piece_columns[pieces] << landing_rows.clamp(min=0)[..., None]
    )
    cleared_columns, cleared_lines = _clear_full_rows(placed_columns, tables)
    return cleared_columns, cleared_lines, valid


def enumerate_agent_afterstates(
    columns: torch.Tensor,
    current_pieces: torch.Tensor,
    next_pieces: torch.Tensor,
    held_pieces: torch.Tensor,
) -> AgentAfterstates:
    """Simulate all 80 actions: 40 direct placements, then 40 after hold."""
    pieces_after_hold = torch.where(held_pieces == EMPTY_HOLD_SLOT, next_pieces, held_pieces)
    board_count = columns.shape[0]
    dropped_columns, cleared_lines, valid = drop_pieces(
        columns.repeat_interleave(2, dim=0),
        torch.stack((current_pieces, pieces_after_hold), dim=1).reshape(-1),
    )
    return AgentAfterstates(
        columns=dropped_columns.view(board_count, -1, BOARD_COLUMNS),
        cleared_lines=cleared_lines.view(board_count, -1),
        valid=valid.view(board_count, -1),
    )


def score_boards(columns: torch.Tensor, cleared_lines: torch.Tensor) -> torch.Tensor:
    """Linear planner heuristic, evaluated in float64 for stable tie-breaking."""
    tables = _placement_tables(columns.device)
    column_heights = BOARD_ROWS - tables.top_rows[columns]
    aggregate_height = column_heights.sum(dim=-1)
    # Every empty cell below a column's top block is a hole.
    holes = aggregate_height - tables.popcounts[columns].sum(dim=-1)
    bumpiness = column_heights.diff(dim=-1).abs().sum(dim=-1)
    return (
        0.760666 * cleared_lines.double()
        - 0.510066 * aggregate_height.double()
        - 0.35663 * holes.double()
        - 0.184483 * bumpiness.double()
    )


def choose_agent_actions(
    columns: torch.Tensor,
    current_pieces: torch.Tensor,
    next_pieces: torch.Tensor,
    held_pieces: torch.Tensor,
    *,
    use_lookahead: bool = True,
    afterstates: AgentAfterstates | None = None,
) -> torch.Tensor:
    """Choose the planner's agent action for every board in a batch.

    With a held piece, both options are scored with a next-piece lookahead.
    With an empty hold slot, holding consumes the known next piece, so neither
    option uses lookahead and the two stay comparable. Ties prefer the lowest
    action index and direct placement over hold.
    """
    if afterstates is None:
        afterstates = enumerate_agent_afterstates(
            columns,
            current_pieces,
            next_pieces,
            held_pieces,
        )
    action_quality = score_boards(afterstates.columns, afterstates.cleared_lines)

    if use_lookahead:
        lookahead_pairs = afterstates.valid & (held_pieces != EMPTY_HOLD_SLOT)[:, None]
        state_indices, action_indices = lookahead_pairs.nonzero(as_tuple=True)
        if state_indices.numel():
            next_columns, next_cleared_lines, next_valid = drop_pieces(
                afterstates.columns[state_indices, action_indices],
                next_pieces[state_indices],
            )
            best_next_quality = (
                score_boards(next_columns, next_cleared_lines)
                .masked_fill(~next_valid, -torch.inf)
                .amax(dim=-1)
                .clamp(min=TOP_OUT_LOOKAHEAD_QUALITY)
            )
            action_quality[state_indices, action_indices] += LOOKAHEAD_WEIGHT * best_next_quality

    action_quality = action_quality.masked_fill(~afterstates.valid, -torch.inf)
    direct_quality = action_quality[:, :DIRECT_PLACEMENT_ACTION_COUNT]
    hold_quality = action_quality[:, DIRECT_PLACEMENT_ACTION_COUNT:]
    best_direct_quality, best_direct_actions = _first_maximum(direct_quality)
    best_hold_quality, best_hold_actions = _first_maximum(hold_quality)
    return torch.where(
        best_hold_quality > best_direct_quality,
        best_hold_actions + DIRECT_PLACEMENT_ACTION_COUNT,
        best_direct_actions,
    )


def choose_agent_action(
    board: Board,
    current_piece: int,
    next_piece: int,
    held_piece: int,
    *,
    use_lookahead: bool = True,
) -> int:
    def piece_tensor(piece: int) -> torch.Tensor:
        return torch.tensor([piece], dtype=torch.int64)

    return int(
        choose_agent_actions(
            boards_to_columns(torch.as_tensor(np.asarray(board) != 0)[None]),
            piece_tensor(current_piece),
            piece_tensor(next_piece),
            piece_tensor(held_piece),
            use_lookahead=use_lookahead,
        )[0]
    )


def _first_maximum(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    best_indices = values.argmax(dim=1)
    return values.gather(1, best_indices[:, None]).squeeze(1), best_indices


def _clear_full_rows(
    columns: torch.Tensor,
    tables: _PlacementTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_rows = columns[..., 0]
    for column in range(1, BOARD_COLUMNS):
        full_rows = full_rows & columns[..., column]
    cleared_lines = tables.popcounts[full_rows]
    if not bool(full_rows.any()):
        return columns, cleared_lines

    # Remove the topmost full row each pass: rows above it fall by one and
    # rows below it, including any other full rows, keep their index.
    for _ in range(_MAXIMUM_CLEARED_LINES):
        topmost_full_row = full_rows & -full_rows
        rows_above = torch.where(topmost_full_row != 0, topmost_full_row - 1, 0).unsqueeze(-1)
        rows_below = ~(rows_above | topmost_full_row.unsqueeze(-1))
        columns = ((columns & rows_above) << 1) | (columns & rows_below)
        full_rows = full_rows & (full_rows - 1)
    return columns, cleared_lines

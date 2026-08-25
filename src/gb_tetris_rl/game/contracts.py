from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

BOARD_ROWS = 18
BOARD_COLUMNS = 10
BOARD_SHAPE = (BOARD_ROWS, BOARD_COLUMNS)
BOARD_CELL_COUNT = BOARD_ROWS * BOARD_COLUMNS

TETROMINO_TYPE_COUNT = 7
EMPTY_HOLD_SLOT = TETROMINO_TYPE_COUNT
HELD_PIECE_TYPE_COUNT = TETROMINO_TYPE_COUNT + 1

PLACEMENT_ROTATION_COUNT = 4
DIRECT_PLACEMENT_ACTION_COUNT = PLACEMENT_ROTATION_COUNT * BOARD_COLUMNS
AGENT_ACTION_COUNT = DIRECT_PLACEMENT_ACTION_COUNT * 2

PIECE_CONTEXT_SIZE = TETROMINO_TYPE_COUNT * 2 + HELD_PIECE_TYPE_COUNT
AGENT_OBSERVATION_SHAPE = (BOARD_CELL_COUNT + PIECE_CONTEXT_SIZE,)

Board = NDArray[np.uint8]
AgentObservation = NDArray[np.uint8]


class EpisodeInfo(TypedDict):
    score: int
    cleared_lines: int
    level: int
    aggregate_height: int
    holes: int
    bumpiness: int
    episode_steps: int
    current_piece: int
    next_piece: int
    held_piece: int


@dataclass(frozen=True)
class PlacementDecision:
    rotation: int
    column_from_left_wall: int
    uses_hold: bool


def decode_agent_action(agent_action: int) -> PlacementDecision:
    if not 0 <= agent_action < AGENT_ACTION_COUNT:
        raise ValueError(f"agent action must be between 0 and {AGENT_ACTION_COUNT - 1}")

    hold_group, direct_placement_action = divmod(
        agent_action,
        DIRECT_PLACEMENT_ACTION_COUNT,
    )
    rotation, column_from_left_wall = divmod(
        direct_placement_action,
        BOARD_COLUMNS,
    )
    return PlacementDecision(
        rotation=rotation,
        column_from_left_wall=column_from_left_wall,
        uses_hold=bool(hold_group),
    )


def encode_agent_observation(
    board: Board,
    current_piece: int,
    next_piece: int,
    held_piece: int,
) -> AgentObservation:
    if board.shape != BOARD_SHAPE:
        raise ValueError(f"expected board shape {BOARD_SHAPE}, received {board.shape}")
    _validate_piece_type(current_piece, "current piece", allow_empty=False)
    _validate_piece_type(next_piece, "next piece", allow_empty=False)
    _validate_piece_type(held_piece, "held piece", allow_empty=True)

    observation = np.zeros(AGENT_OBSERVATION_SHAPE, dtype=np.uint8)
    observation[:BOARD_CELL_COUNT] = board.reshape(-1)
    observation[BOARD_CELL_COUNT + current_piece] = 1
    observation[BOARD_CELL_COUNT + TETROMINO_TYPE_COUNT + next_piece] = 1
    held_piece_offset = BOARD_CELL_COUNT + (2 * TETROMINO_TYPE_COUNT)
    observation[held_piece_offset + held_piece] = 1
    return observation


def board_from_observation(observation: AgentObservation) -> Board:
    if observation.shape != AGENT_OBSERVATION_SHAPE:
        raise ValueError(
            f"expected observation shape {AGENT_OBSERVATION_SHAPE}, received {observation.shape}"
        )
    return observation[:BOARD_CELL_COUNT].reshape(BOARD_SHAPE)


def _validate_piece_type(piece_type: int, role: str, *, allow_empty: bool) -> None:
    maximum_piece_type = EMPTY_HOLD_SLOT if allow_empty else TETROMINO_TYPE_COUNT - 1
    if not 0 <= piece_type <= maximum_piece_type:
        raise ValueError(f"{role} must be between 0 and {maximum_piece_type}")

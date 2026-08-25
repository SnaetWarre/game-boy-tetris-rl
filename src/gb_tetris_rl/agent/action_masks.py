from functools import cache

import numpy as np
from numpy.typing import NDArray

from gb_tetris_rl.agent.planner import PANDORAS_PIECE_ROTATIONS
from gb_tetris_rl.game.contracts import (
    AGENT_ACTION_COUNT,
    BOARD_COLUMNS,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    AgentObservation,
    piece_context_from_observation,
)

ActionMask = NDArray[np.bool_]


@cache
def _canonical_direct_action_mask(piece_type: int) -> ActionMask:
    action_mask = np.zeros(DIRECT_PLACEMENT_ACTION_COUNT, dtype=np.bool_)
    seen_piece_shapes = set()
    for rotation, piece_shape in enumerate(PANDORAS_PIECE_ROTATIONS[piece_type]):
        if piece_shape in seen_piece_shapes:
            continue
        seen_piece_shapes.add(piece_shape)
        piece_width = max(column_offset for _, column_offset in piece_shape) + 1
        for left_column in range(BOARD_COLUMNS - piece_width + 1):
            action_mask[rotation * BOARD_COLUMNS + left_column] = True
    action_mask.setflags(write=False)
    return action_mask


def canonical_agent_action_mask(observation: AgentObservation) -> ActionMask:
    piece_context = piece_context_from_observation(observation)
    piece_after_hold = (
        piece_context.next_piece
        if piece_context.held_piece == EMPTY_HOLD_SLOT
        else piece_context.held_piece
    )
    action_mask = np.zeros(AGENT_ACTION_COUNT, dtype=np.bool_)
    action_mask[:DIRECT_PLACEMENT_ACTION_COUNT] = _canonical_direct_action_mask(
        piece_context.current_piece
    )
    action_mask[DIRECT_PLACEMENT_ACTION_COUNT:] = _canonical_direct_action_mask(piece_after_hold)
    return action_mask


def canonical_agent_action_masks(observations: NDArray[np.generic]) -> ActionMask:
    observation_batch = np.asarray(observations)
    if observation_batch.ndim == 1:
        return canonical_agent_action_mask(observation_batch)
    return np.stack(
        [canonical_agent_action_mask(observation) for observation in observation_batch],
        axis=0,
    )

import numpy as np
import torch
from numpy.typing import NDArray

from gb_tetris_rl.agent.planner import canonical_direct_action_masks
from gb_tetris_rl.game.contracts import (
    AGENT_OBSERVATION_SHAPE,
    BOARD_CELL_COUNT,
    EMPTY_HOLD_SLOT,
    TETROMINO_TYPE_COUNT,
    AgentObservation,
)

ActionMask = NDArray[np.bool_]


def canonical_agent_action_mask_tensor(
    current_pieces: torch.Tensor,
    next_pieces: torch.Tensor,
    held_pieces: torch.Tensor,
) -> torch.Tensor:
    """Return [N, 80] masks of unique, in-bounds direct and hold placements."""
    pieces_after_hold = torch.where(held_pieces == EMPTY_HOLD_SLOT, next_pieces, held_pieces)
    return torch.cat(
        (
            canonical_direct_action_masks(current_pieces),
            canonical_direct_action_masks(pieces_after_hold),
        ),
        dim=1,
    )


def canonical_agent_action_mask(observation: AgentObservation) -> ActionMask:
    return canonical_agent_action_masks(observation)


def canonical_agent_action_masks(observations: NDArray[np.generic]) -> ActionMask:
    observation_batch = np.asarray(observations)
    is_single_observation = observation_batch.ndim == 1
    observation_batch = np.atleast_2d(observation_batch)
    if observation_batch.shape[1:] != AGENT_OBSERVATION_SHAPE:
        raise ValueError(
            f"expected observation shape {AGENT_OBSERVATION_SHAPE}, "
            f"received {observation_batch.shape[1:]}"
        )

    next_piece_start = BOARD_CELL_COUNT + TETROMINO_TYPE_COUNT
    held_piece_start = next_piece_start + TETROMINO_TYPE_COUNT
    piece_indices = []
    for context_name, start, stop in (
        ("current piece", BOARD_CELL_COUNT, next_piece_start),
        ("next piece", next_piece_start, held_piece_start),
        ("held piece", held_piece_start, AGENT_OBSERVATION_SHAPE[0]),
    ):
        encoded_values = observation_batch[:, start:stop]
        if np.any(np.count_nonzero(encoded_values, axis=1) != 1):
            raise ValueError(f"observation must encode exactly one {context_name}")
        piece_indices.append(torch.as_tensor(encoded_values.argmax(axis=1)))

    action_masks = canonical_agent_action_mask_tensor(*piece_indices).numpy()
    return action_masks[0] if is_single_observation else action_masks

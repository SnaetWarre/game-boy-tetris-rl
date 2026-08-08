from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gb_tetris_rl.environment import (
    PLACEMENT_CONTEXT_SIZE,
    TETRIS_OBSERVATION_SHAPE,
    TETROMINO_TYPE_COUNT,
)
from gb_tetris_rl.heuristic import choose_placement_action, enumerate_placements


@dataclass(frozen=True)
class ExpertDataset:
    observations: NDArray[np.uint8]
    placement_actions: NDArray[np.int64]


@dataclass(frozen=True)
class ExpertTrainingMetrics:
    final_training_loss: float
    validation_accuracy: float


def generate_expert_dataset(
    sample_count: int,
    *,
    seed: int,
    maximum_episode_pieces: int = 40,
) -> ExpertDataset:
    if sample_count < 1:
        raise ValueError("expert sample count must be at least 1")
    if maximum_episode_pieces < 1:
        raise ValueError("maximum expert episode pieces must be at least 1")

    random_generator = np.random.default_rng(seed)
    observation_width = TETRIS_OBSERVATION_SHAPE[0] + PLACEMENT_CONTEXT_SIZE
    observations = np.zeros((sample_count, observation_width), dtype=np.uint8)
    placement_actions = np.zeros(sample_count, dtype=np.int64)
    board = np.zeros((18, 10), dtype=np.uint8)
    current_piece = int(random_generator.integers(TETROMINO_TYPE_COUNT))
    next_piece = int(random_generator.integers(TETROMINO_TYPE_COUNT))
    episode_piece_count = 0

    for sample_index in range(sample_count):
        legal_placements = enumerate_placements(board, current_piece)
        if not legal_placements or episode_piece_count >= maximum_episode_pieces:
            board.fill(0)
            episode_piece_count = 0
            legal_placements = enumerate_placements(board, current_piece)

        observations[sample_index, : TETRIS_OBSERVATION_SHAPE[0]] = board.reshape(-1)
        observations[sample_index, TETRIS_OBSERVATION_SHAPE[0] + current_piece] = 1
        observations[
            sample_index,
            TETRIS_OBSERVATION_SHAPE[0] + TETROMINO_TYPE_COUNT + next_piece,
        ] = 1

        expert_action = choose_placement_action(board, current_piece)
        placement_actions[sample_index] = expert_action
        board = next(
            placement.board for placement in legal_placements if placement.action == expert_action
        )
        current_piece = next_piece
        next_piece = int(random_generator.integers(TETROMINO_TYPE_COUNT))
        episode_piece_count += 1

    return ExpertDataset(observations=observations, placement_actions=placement_actions)


def pretrain_dqn_policy(
    model,
    expert_dataset: ExpertDataset,
    *,
    epoch_count: int,
    batch_size: int,
    seed: int,
) -> ExpertTrainingMetrics:
    if epoch_count < 1:
        raise ValueError("expert epoch count must be at least 1")
    if batch_size < 1:
        raise ValueError("expert batch size must be at least 1")

    import torch
    from torch.nn import functional as torch_functional

    sample_count = expert_dataset.observations.shape[0]
    if sample_count < 2:
        raise ValueError("expert training requires at least two samples")

    random_generator = np.random.default_rng(seed)
    shuffled_indices = random_generator.permutation(sample_count)
    validation_sample_count = max(1, sample_count // 10)
    validation_indices = shuffled_indices[:validation_sample_count]
    training_indices = shuffled_indices[validation_sample_count:]
    optimizer = torch.optim.AdamW(model.q_net.parameters(), lr=3e-4, weight_decay=1e-5)
    model.policy.set_training_mode(True)
    final_training_loss = 0.0

    for epoch_index in range(epoch_count):
        epoch_training_indices = random_generator.permutation(training_indices)
        epoch_loss_total = 0.0
        batch_count = 0
        for batch_start in range(0, len(epoch_training_indices), batch_size):
            batch_indices = epoch_training_indices[batch_start : batch_start + batch_size]
            observation_batch = torch.as_tensor(
                expert_dataset.observations[batch_indices],
                device=model.device,
            )
            action_batch = torch.as_tensor(
                expert_dataset.placement_actions[batch_indices],
                device=model.device,
            )
            predicted_action_values = model.q_net(observation_batch)
            classification_loss = torch_functional.cross_entropy(
                predicted_action_values,
                action_batch,
            )
            optimizer.zero_grad(set_to_none=True)
            classification_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.q_net.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss_total += float(classification_loss.detach().cpu())
            batch_count += 1

        final_training_loss = epoch_loss_total / max(batch_count, 1)
        print(
            f"Expert epoch {epoch_index + 1}/{epoch_count}: "
            f"loss={final_training_loss:.4f}"
        )

    model.q_net_target.load_state_dict(model.q_net.state_dict())
    model.policy.set_training_mode(False)
    with torch.no_grad():
        validation_observations = torch.as_tensor(
            expert_dataset.observations[validation_indices],
            device=model.device,
        )
        validation_actions = torch.as_tensor(
            expert_dataset.placement_actions[validation_indices],
            device=model.device,
        )
        predicted_validation_actions = model.q_net(validation_observations).argmax(dim=1)
        validation_accuracy = float(
            (predicted_validation_actions == validation_actions).float().mean().cpu()
        )

    return ExpertTrainingMetrics(
        final_training_loss=final_training_loss,
        validation_accuracy=validation_accuracy,
    )

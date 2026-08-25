from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gb_tetris_rl.agent.action_masks import canonical_agent_action_masks
from gb_tetris_rl.agent.planner import choose_agent_action, enumerate_placements
from gb_tetris_rl.game.contracts import (
    AGENT_OBSERVATION_SHAPE,
    BOARD_SHAPE,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    TETROMINO_TYPE_COUNT,
    Board,
    encode_agent_observation,
)


@dataclass(frozen=True)
class DemonstrationDataset:
    observations: NDArray[np.uint8]
    agent_actions: NDArray[np.int64]


@dataclass(frozen=True)
class ImitationTrainingMetrics:
    final_training_loss: float
    validation_accuracy: float


def generate_planner_demonstrations(
    sample_count: int,
    *,
    seed: int,
    maximum_episode_pieces: int = 200,
    use_lookahead: bool = False,
) -> DemonstrationDataset:
    """Generate synthetic boards labelled by the deterministic planner."""
    if sample_count < 1:
        raise ValueError("demonstration sample count must be at least 1")
    if maximum_episode_pieces < 1:
        raise ValueError("maximum demonstration episode pieces must be at least 1")

    random_generator = np.random.default_rng(seed)
    observations = np.zeros((sample_count, *AGENT_OBSERVATION_SHAPE), dtype=np.uint8)
    agent_actions = np.zeros(sample_count, dtype=np.int64)

    board: Board = np.zeros(BOARD_SHAPE, dtype=np.uint8)
    current_piece = int(random_generator.integers(TETROMINO_TYPE_COUNT))
    next_piece = int(random_generator.integers(TETROMINO_TYPE_COUNT))
    held_piece = EMPTY_HOLD_SLOT
    episode_piece_count = 0

    for sample_index in range(sample_count):
        legal_placements = enumerate_placements(board, current_piece)
        episode_needs_reset = not legal_placements or episode_piece_count >= maximum_episode_pieces
        if episode_needs_reset:
            board.fill(0)
            episode_piece_count = 0
            held_piece = EMPTY_HOLD_SLOT

        observations[sample_index] = encode_agent_observation(
            board,
            current_piece,
            next_piece,
            held_piece,
        )
        planner_action = choose_agent_action(
            board,
            current_piece,
            next_piece,
            held_piece,
            use_lookahead=use_lookahead,
        )
        agent_actions[sample_index] = planner_action

        direct_placement_action = planner_action
        piece_to_place = current_piece
        if planner_action >= DIRECT_PLACEMENT_ACTION_COUNT:
            direct_placement_action -= DIRECT_PLACEMENT_ACTION_COUNT
            piece_to_place = next_piece if held_piece == EMPTY_HOLD_SLOT else held_piece
            previous_held_piece = held_piece
            held_piece = current_piece
            if previous_held_piece == EMPTY_HOLD_SLOT:
                next_piece = int(random_generator.integers(TETROMINO_TYPE_COUNT))

        legal_placements = enumerate_placements(board, piece_to_place)
        board = next(
            placement.board
            for placement in legal_placements
            if placement.placement_action == direct_placement_action
        )
        current_piece = next_piece
        next_piece = int(random_generator.integers(TETROMINO_TYPE_COUNT))
        episode_piece_count += 1

    return DemonstrationDataset(observations=observations, agent_actions=agent_actions)


def pretrain_policy_from_demonstrations(
    dqn_agent,
    demonstration_dataset: DemonstrationDataset,
    *,
    epoch_count: int,
    batch_size: int,
    seed: int,
) -> ImitationTrainingMetrics:
    """Teach the DQN network to copy planner actions with cross-entropy loss."""
    if epoch_count < 1:
        raise ValueError("imitation epoch count must be at least 1")
    if batch_size < 1:
        raise ValueError("imitation batch size must be at least 1")

    import torch
    from torch.nn import functional as torch_functional

    sample_count = demonstration_dataset.observations.shape[0]
    if sample_count < 2:
        raise ValueError("imitation training requires at least two samples")

    random_generator = np.random.default_rng(seed)
    shuffled_indices = random_generator.permutation(sample_count)
    validation_sample_count = max(1, sample_count // 10)
    validation_indices = shuffled_indices[:validation_sample_count]
    training_indices = shuffled_indices[validation_sample_count:]
    optimizer = torch.optim.AdamW(
        dqn_agent.q_net.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )
    dqn_agent.policy.set_training_mode(True)
    final_training_loss = 0.0

    for epoch_index in range(epoch_count):
        epoch_training_indices = random_generator.permutation(training_indices)
        epoch_loss_total = 0.0
        batch_count = 0
        for batch_start in range(0, len(epoch_training_indices), batch_size):
            batch_indices = epoch_training_indices[batch_start : batch_start + batch_size]
            observation_batch = torch.as_tensor(
                demonstration_dataset.observations[batch_indices],
                device=dqn_agent.device,
            )
            action_batch = torch.as_tensor(
                demonstration_dataset.agent_actions[batch_indices],
                device=dqn_agent.device,
            )
            predicted_action_values = dqn_agent.q_net(observation_batch)
            if getattr(dqn_agent.policy, "uses_canonical_action_masks", False):
                valid_action_masks = canonical_agent_action_masks(
                    demonstration_dataset.observations[batch_indices]
                )
                valid_action_mask_tensor = torch.as_tensor(
                    valid_action_masks,
                    device=dqn_agent.device,
                )
                predicted_action_values.masked_fill_(~valid_action_mask_tensor, -torch.inf)
            classification_loss = torch_functional.cross_entropy(
                predicted_action_values,
                action_batch,
            )
            optimizer.zero_grad(set_to_none=True)
            classification_loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn_agent.q_net.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss_total += float(classification_loss.detach().cpu())
            batch_count += 1

        final_training_loss = epoch_loss_total / max(batch_count, 1)
        print(f"Imitation epoch {epoch_index + 1}/{epoch_count}: loss={final_training_loss:.4f}")

    dqn_agent.q_net_target.load_state_dict(dqn_agent.q_net.state_dict())
    dqn_agent.policy.set_training_mode(False)
    with torch.no_grad():
        validation_observations = torch.as_tensor(
            demonstration_dataset.observations[validation_indices],
            device=dqn_agent.device,
        )
        validation_actions = torch.as_tensor(
            demonstration_dataset.agent_actions[validation_indices],
            device=dqn_agent.device,
        )
        validation_action_values = dqn_agent.q_net(validation_observations)
        if getattr(dqn_agent.policy, "uses_canonical_action_masks", False):
            validation_action_masks = canonical_agent_action_masks(
                demonstration_dataset.observations[validation_indices]
            )
            validation_action_mask_tensor = torch.as_tensor(
                validation_action_masks,
                device=dqn_agent.device,
            )
            validation_action_values.masked_fill_(
                ~validation_action_mask_tensor,
                -torch.inf,
            )
        predicted_validation_actions = validation_action_values.argmax(dim=1)
        validation_accuracy = float(
            (predicted_validation_actions == validation_actions).float().mean().cpu()
        )

    return ImitationTrainingMetrics(
        final_training_loss=final_training_loss,
        validation_accuracy=validation_accuracy,
    )

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as torch_functional

from gb_tetris_rl.agent.planner import (
    COLUMN_DTYPE,
    AgentAfterstates,
    choose_agent_actions,
    columns_to_boards,
    enumerate_agent_afterstates,
)
from gb_tetris_rl.game.contracts import (
    AGENT_OBSERVATION_SHAPE,
    BOARD_CELL_COUNT,
    BOARD_COLUMNS,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    TETROMINO_TYPE_COUNT,
)

BehaviourPolicy = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DemonstrationDataset:
    observations: NDArray[np.uint8]
    agent_actions: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.agent_actions)

    def concatenate(self, other: "DemonstrationDataset") -> "DemonstrationDataset":
        return DemonstrationDataset(
            observations=np.concatenate((self.observations, other.observations)),
            agent_actions=np.concatenate((self.agent_actions, other.agent_actions)),
        )


@dataclass(frozen=True)
class ImitationTrainingMetrics:
    final_training_loss: float
    validation_accuracy: float


class SimulatedGames:
    """Parallel hold-aware games on the bitboard simulator.

    Piece order is uniform random. A game ends when the chosen action does not
    fit or the piece limit is reached, and it restarts on an empty board.
    """

    def __init__(
        self,
        game_count: int,
        *,
        seed: int,
        maximum_episode_pieces: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if game_count < 1:
            raise ValueError("simulated game count must be at least 1")
        if maximum_episode_pieces < 1:
            raise ValueError("maximum episode pieces must be at least 1")
        self.device = torch.device(device)
        self.maximum_episode_pieces = maximum_episode_pieces
        self._random_generator = torch.Generator(device=self.device).manual_seed(seed)
        self.columns = torch.zeros(
            (game_count, BOARD_COLUMNS), dtype=COLUMN_DTYPE, device=self.device
        )
        self.current_pieces = self._draw_pieces(game_count)
        self.next_pieces = self._draw_pieces(game_count)
        self.held_pieces = torch.full_like(self.current_pieces, EMPTY_HOLD_SLOT)
        self.piece_counts = torch.zeros_like(self.current_pieces)
        self.cleared_lines = torch.zeros_like(self.current_pieces)

    def afterstates(self) -> AgentAfterstates:
        return enumerate_agent_afterstates(
            self.columns,
            self.current_pieces,
            self.next_pieces,
            self.held_pieces,
        )

    def observations(self) -> torch.Tensor:
        game_count = self.columns.shape[0]
        observations = torch.zeros(
            (game_count, *AGENT_OBSERVATION_SHAPE),
            dtype=torch.uint8,
            device=self.device,
        )
        observations[:, :BOARD_CELL_COUNT] = columns_to_boards(self.columns).reshape(game_count, -1)
        game_indices = torch.arange(game_count, device=self.device)
        next_piece_start = BOARD_CELL_COUNT + TETROMINO_TYPE_COUNT
        observations[game_indices, BOARD_CELL_COUNT + self.current_pieces] = 1
        observations[game_indices, next_piece_start + self.next_pieces] = 1
        observations[game_indices, next_piece_start + TETROMINO_TYPE_COUNT + self.held_pieces] = 1
        return observations

    def step(
        self,
        agent_actions: torch.Tensor,
        afterstates: AgentAfterstates,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one action per game.

        Returns which games just ended and every game's line total including
        this placement; ended games then restart on an empty board.
        """
        game_indices = torch.arange(self.columns.shape[0], device=self.device)
        action_was_valid = afterstates.valid[game_indices, agent_actions]
        uses_hold = agent_actions >= DIRECT_PLACEMENT_ACTION_COUNT
        holds_from_empty_slot = uses_hold & (self.held_pieces == EMPTY_HOLD_SLOT)
        first_drawn_pieces = self._draw_pieces(len(game_indices))
        second_drawn_pieces = self._draw_pieces(len(game_indices))

        self.columns = afterstates.columns[game_indices, agent_actions]
        self.cleared_lines = self.cleared_lines + torch.where(
            action_was_valid,
            afterstates.cleared_lines[game_indices, agent_actions],
            0,
        )
        self.held_pieces = torch.where(uses_hold, self.current_pieces, self.held_pieces)
        self.current_pieces = torch.where(
            holds_from_empty_slot,
            first_drawn_pieces,
            self.next_pieces,
        )
        self.next_pieces = torch.where(
            holds_from_empty_slot,
            second_drawn_pieces,
            first_drawn_pieces,
        )
        self.piece_counts += 1

        episode_line_counts = self.cleared_lines
        game_ended = ~action_was_valid | (self.piece_counts >= self.maximum_episode_pieces)
        self.columns[game_ended] = 0
        self.held_pieces[game_ended] = EMPTY_HOLD_SLOT
        self.piece_counts[game_ended] = 0
        self.cleared_lines = torch.where(game_ended, 0, self.cleared_lines)
        return game_ended, episode_line_counts

    def _draw_pieces(self, piece_count: int) -> torch.Tensor:
        return torch.randint(
            TETROMINO_TYPE_COUNT,
            (piece_count,),
            generator=self._random_generator,
            device=self.device,
        )


def generate_planner_demonstrations(
    sample_count: int,
    *,
    seed: int,
    maximum_episode_pieces: int = 200,
    use_lookahead: bool = False,
    game_count: int = 256,
    device: torch.device | str = "cpu",
    behaviour_policy: BehaviourPolicy | None = None,
) -> DemonstrationDataset:
    """Label simulated boards with the deterministic planner.

    Without a behaviour policy the games follow the planner. With one, the
    games follow the policy while every visited board is still labelled by the
    planner, so later training corrects the policy's own mistakes (DAgger).
    """
    if sample_count < 1:
        raise ValueError("demonstration sample count must be at least 1")

    games = SimulatedGames(
        min(game_count, sample_count),
        seed=seed,
        maximum_episode_pieces=maximum_episode_pieces,
        device=device,
    )
    observation_batches: list[torch.Tensor] = []
    action_batches: list[torch.Tensor] = []
    collected_sample_count = 0
    while collected_sample_count < sample_count:
        afterstates = games.afterstates()
        observations = games.observations()
        planner_actions = choose_agent_actions(
            games.columns,
            games.current_pieces,
            games.next_pieces,
            games.held_pieces,
            use_lookahead=use_lookahead,
            afterstates=afterstates,
        )
        game_indices = torch.arange(len(planner_actions), device=games.device)
        has_valid_label = afterstates.valid[game_indices, planner_actions]
        observation_batches.append(observations[has_valid_label])
        action_batches.append(planner_actions[has_valid_label])
        collected_sample_count += int(has_valid_label.sum())

        behaviour_actions = (
            planner_actions if behaviour_policy is None else behaviour_policy(observations)
        )
        games.step(behaviour_actions, afterstates)

    return DemonstrationDataset(
        observations=torch.cat(observation_batches)[:sample_count].cpu().numpy(),
        agent_actions=torch.cat(action_batches)[:sample_count].cpu().numpy().astype(np.int64),
    )


def simulate_policy_lines(
    policy: BehaviourPolicy,
    *,
    game_count: int,
    seed: int,
    maximum_episode_pieces: int,
    device: torch.device | str = "cpu",
) -> NDArray[np.int64]:
    """Cleared lines per simulated game until top-out or the piece limit."""
    games = SimulatedGames(
        game_count,
        seed=seed,
        maximum_episode_pieces=maximum_episode_pieces,
        device=device,
    )
    final_line_counts = torch.full((game_count,), -1, device=games.device)
    while bool((final_line_counts < 0).any()):
        afterstates = games.afterstates()
        game_ended, episode_line_counts = games.step(policy(games.observations()), afterstates)
        just_finished = game_ended & (final_line_counts < 0)
        final_line_counts[just_finished] = episode_line_counts[just_finished]
    return final_line_counts.cpu().numpy()


def greedy_policy(q_network: torch.nn.Module) -> BehaviourPolicy:
    def choose_actions(observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return q_network(observations).argmax(dim=1)

    return choose_actions


def pretrain_policy_from_demonstrations(
    dqn_agent,
    demonstration_dataset: DemonstrationDataset,
    *,
    epoch_count: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 1e-3,
) -> ImitationTrainingMetrics:
    """Teach the Q-network to rank the planner's action first (cross-entropy).

    The whole dataset lives on the training device, the learning rate follows
    a one-cycle schedule, and CUDA runs use bfloat16 autocast.
    """
    if epoch_count < 1:
        raise ValueError("imitation epoch count must be at least 1")
    if batch_size < 1:
        raise ValueError("imitation batch size must be at least 1")
    sample_count = len(demonstration_dataset)
    if sample_count < 2:
        raise ValueError("imitation training requires at least two samples")

    device = dqn_agent.device
    q_network = dqn_agent.q_net
    observations = torch.as_tensor(demonstration_dataset.observations, device=device)
    agent_actions = torch.as_tensor(demonstration_dataset.agent_actions, device=device)
    random_generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(sample_count, generator=random_generator).to(device)
    validation_sample_count = max(1, sample_count // 10)
    validation_indices = shuffled_indices[:validation_sample_count]
    training_indices = shuffled_indices[validation_sample_count:]
    batches_per_epoch = -(-len(training_indices) // batch_size)

    optimizer = torch.optim.AdamW(q_network.parameters(), lr=learning_rate, weight_decay=1e-4)
    learning_rate_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=epoch_count * batches_per_epoch,
        pct_start=0.1,
    )
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )
    final_training_loss = 0.0
    validation_accuracy = 0.0
    for epoch_index in range(epoch_count):
        dqn_agent.policy.set_training_mode(True)
        epoch_order = training_indices[
            torch.randperm(len(training_indices), generator=random_generator).to(device)
        ]
        epoch_loss_total = torch.zeros((), device=device)
        for batch_start in range(0, len(epoch_order), batch_size):
            batch_indices = epoch_order[batch_start : batch_start + batch_size]
            with autocast:
                action_values = q_network(observations[batch_indices])
            classification_loss = torch_functional.cross_entropy(
                action_values.float(),
                agent_actions[batch_indices],
            )
            optimizer.zero_grad(set_to_none=True)
            classification_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
            optimizer.step()
            learning_rate_schedule.step()
            epoch_loss_total += classification_loss.detach()

        final_training_loss = float(epoch_loss_total) / batches_per_epoch
        validation_accuracy = _prediction_accuracy(
            dqn_agent,
            observations[validation_indices],
            agent_actions[validation_indices],
            batch_size,
            autocast,
        )
        print(
            f"Imitation epoch {epoch_index + 1}/{epoch_count}: "
            f"loss={final_training_loss:.4f}, validation_accuracy={validation_accuracy:.1%}"
        )

    dqn_agent.q_net_target.load_state_dict(q_network.state_dict())
    dqn_agent.policy.set_training_mode(False)
    return ImitationTrainingMetrics(
        final_training_loss=final_training_loss,
        validation_accuracy=validation_accuracy,
    )


def _prediction_accuracy(dqn_agent, observations, agent_actions, batch_size, autocast) -> float:
    dqn_agent.policy.set_training_mode(False)
    correct_prediction_count = torch.zeros((), device=observations.device)
    with torch.no_grad(), autocast:
        for batch_start in range(0, len(observations), batch_size):
            batch_slice = slice(batch_start, batch_start + batch_size)
            predicted_actions = dqn_agent.q_net(observations[batch_slice]).argmax(dim=1)
            correct_prediction_count += (predicted_actions == agent_actions[batch_slice]).sum()
    return float(correct_prediction_count) / len(observations)

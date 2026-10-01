import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn
from torch.nn import functional as torch_functional

from gb_tetris_rl.agent.action_masks import (
    canonical_agent_action_mask_tensor,
    canonical_agent_action_masks,
)
from gb_tetris_rl.agent.planner import (
    columns_to_boards,
    decode_observations,
    enumerate_agent_afterstates,
)
from gb_tetris_rl.game.contracts import (
    BOARD_CELL_COUNT,
    BOARD_COLUMNS,
    BOARD_ROWS,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    HELD_PIECE_TYPE_COUNT,
    PIECE_CONTEXT_SIZE,
)

_CLEARED_LINE_CLASS_COUNT = 5
# Holding from an empty slot consumes the known next piece, so the piece that
# follows the afterstate is unknown; it gets its own class next to the seven pieces.
_UNKNOWN_UPCOMING_PIECE = EMPTY_HOLD_SLOT
AFTERSTATE_CONTEXT_SIZE = _CLEARED_LINE_CLASS_COUNT + 2 * HELD_PIECE_TYPE_COUNT


class TetrisFeatureExtractor(BaseFeaturesExtractor):
    """Encode the board spatially while keeping piece identity explicit."""

    def __init__(self, observation_space: spaces.Box) -> None:
        board_feature_count = 256
        piece_feature_count = 64
        super().__init__(observation_space, board_feature_count + piece_feature_count)
        self.board_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * BOARD_ROWS * BOARD_COLUMNS, board_feature_count),
            nn.ReLU(),
        )
        self.piece_encoder = nn.Sequential(
            nn.Linear(PIECE_CONTEXT_SIZE, piece_feature_count),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        float_observations = observations.float()
        board = float_observations[:, :BOARD_CELL_COUNT].reshape(
            -1,
            1,
            BOARD_ROWS,
            BOARD_COLUMNS,
        )
        piece_context = float_observations[:, BOARD_CELL_COUNT:]
        return torch.cat(
            (self.board_encoder(board), self.piece_encoder(piece_context)),
            dim=1,
        )


class DuelingQNetwork(QNetwork):
    """Estimate state value and per-action advantages in separate streams."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_layers = self.net_arch or [256]
        self.state_value_network = nn.Sequential(
            *create_mlp(self.features_dim, 1, hidden_layers, self.activation_fn)
        )
        self.action_advantage_network = nn.Sequential(
            *create_mlp(
                self.features_dim,
                int(self.action_space.n),
                hidden_layers,
                self.activation_fn,
            )
        )
        del self.q_net

    def forward(self, observations) -> torch.Tensor:
        extracted_features = self.extract_features(observations, self.features_extractor)
        state_values = self.state_value_network(extracted_features)
        action_advantages = self.action_advantage_network(extracted_features)
        action_values = (
            state_values + action_advantages - action_advantages.mean(dim=1, keepdim=True)
        )
        observation_batch = decode_observations(observations)
        action_masks = canonical_agent_action_mask_tensor(
            observation_batch.current_pieces,
            observation_batch.next_pieces,
            observation_batch.held_pieces,
        )
        return action_values.masked_fill(~action_masks, -torch.inf)


class AfterstateQNetwork(QNetwork):
    """Score each placement by encoding the board it leaves behind.

    A fixed, non-learned simulator applies the game rules to all 80 actions;
    the network learns how good each resulting board is, given the line clear
    and the pieces still to come. Placements that do not fit score -inf.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.q_net
        column_feature_count = 64
        self.board_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # One full-height filter bank per column summarizes its profile.
            nn.Conv2d(32, column_feature_count, kernel_size=(BOARD_ROWS, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_network = nn.Sequential(
            *create_mlp(
                column_feature_count * BOARD_COLUMNS + AFTERSTATE_CONTEXT_SIZE,
                1,
                self.net_arch or [256, 256],
                self.activation_fn,
            )
        )

    def forward(self, observations) -> torch.Tensor:
        observation_batch = decode_observations(observations)
        afterstates = enumerate_agent_afterstates(
            observation_batch.columns,
            observation_batch.current_pieces,
            observation_batch.next_pieces,
            observation_batch.held_pieces,
        )
        # A board with no fitting placement is already lost. Scoring canonical
        # actions there keeps every Q-value row finite for TD targets.
        scored_actions = afterstates.valid | (
            ~afterstates.valid.any(dim=1, keepdim=True)
            & canonical_agent_action_mask_tensor(
                observation_batch.current_pieces,
                observation_batch.next_pieces,
                observation_batch.held_pieces,
            )
        )
        state_indices, action_indices = scored_actions.nonzero(as_tuple=True)
        afterstate_boards = columns_to_boards(afterstates.columns[state_indices, action_indices])
        afterstate_context = _afterstate_context(observation_batch, afterstates.cleared_lines)
        board_features = self.board_encoder(afterstate_boards.unsqueeze(1).float())
        afterstate_values = self.value_network(
            torch.cat(
                (board_features, afterstate_context[state_indices, action_indices]),
                dim=1,
            )
        )
        return torch.full(
            scored_actions.shape,
            -torch.inf,
            device=afterstate_values.device,
        ).index_put((state_indices, action_indices), afterstate_values.squeeze(1).float())


def _afterstate_context(observation_batch, cleared_lines: torch.Tensor) -> torch.Tensor:
    """One-hot line clears, upcoming piece, and hold slot after each action."""
    direct_actions = DIRECT_PLACEMENT_ACTION_COUNT
    current_pieces = observation_batch.current_pieces[:, None]
    next_pieces = observation_batch.next_pieces[:, None]
    held_pieces = observation_batch.held_pieces[:, None]
    upcoming_after_hold = torch.where(
        held_pieces == EMPTY_HOLD_SLOT,
        _UNKNOWN_UPCOMING_PIECE,
        next_pieces,
    )
    upcoming_pieces = torch.cat(
        (next_pieces.expand(-1, direct_actions), upcoming_after_hold.expand(-1, direct_actions)),
        dim=1,
    )
    hold_slots = torch.cat(
        (held_pieces.expand(-1, direct_actions), current_pieces.expand(-1, direct_actions)),
        dim=1,
    )
    return torch.cat(
        (
            torch_functional.one_hot(cleared_lines.long(), _CLEARED_LINE_CLASS_COUNT),
            torch_functional.one_hot(upcoming_pieces, HELD_PIECE_TYPE_COUNT),
            torch_functional.one_hot(hold_slots, HELD_PIECE_TYPE_COUNT),
        ),
        dim=-1,
    ).float()


class TetrisDuelingPolicy(DQNPolicy):
    """Legacy direct-action policy, kept so older checkpoints still load."""

    uses_canonical_action_masks = True

    def make_q_net(self) -> DuelingQNetwork:
        network_arguments = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**network_arguments).to(self.device)

    def _predict(self, observations, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        return self.q_net(observations).argmax(dim=1)


class TetrisAfterstatePolicy(DQNPolicy):
    uses_canonical_action_masks = True

    def make_q_net(self) -> AfterstateQNetwork:
        network_arguments = self._update_features_extractor(self.net_args, features_extractor=None)
        return AfterstateQNetwork(**network_arguments).to(self.device)

    def _predict(self, observations, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        return self.q_net(observations).argmax(dim=1)


class TetrisDQN(DQN):
    """Keep every DQN action path inside the canonical placement contract."""

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        should_explore = not deterministic and np.random.random() < self.exploration_rate
        if should_explore:
            return self._sample_canonical_actions(observation), state
        return self.policy.predict(
            observation,
            state,
            episode_start,
            deterministic,
        )

    def _sample_action(self, learning_starts, action_noise=None, n_envs: int = 1):
        if self.num_timesteps < learning_starts:
            if self._last_obs is None:
                raise RuntimeError("cannot sample a warm-up action without an observation")
            canonical_actions = self._sample_canonical_actions(self._last_obs)
            canonical_actions = np.asarray(canonical_actions).reshape(n_envs)
            return canonical_actions, canonical_actions
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def _sample_canonical_actions(self, observation) -> np.ndarray:
        action_masks = canonical_agent_action_masks(np.asarray(observation))
        if action_masks.ndim == 1:
            valid_actions = np.flatnonzero(action_masks)
            return np.asarray(self.action_space.np_random.choice(valid_actions))
        return np.asarray(
            [
                self.action_space.np_random.choice(np.flatnonzero(action_mask))
                for action_mask in action_masks
            ]
        )

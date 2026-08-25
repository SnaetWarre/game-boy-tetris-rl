import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn

from gb_tetris_rl.agent.action_masks import canonical_agent_action_masks
from gb_tetris_rl.game.contracts import (
    BOARD_CELL_COUNT,
    BOARD_COLUMNS,
    BOARD_ROWS,
    PIECE_CONTEXT_SIZE,
)


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
        return state_values + action_advantages - action_advantages.mean(dim=1, keepdim=True)


class TetrisDuelingPolicy(DQNPolicy):
    uses_canonical_action_masks = True

    def make_q_net(self) -> DuelingQNetwork:
        network_arguments = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**network_arguments).to(self.device)

    def _predict(self, observations, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        action_values = self.q_net(observations)
        action_masks = canonical_agent_action_masks(observations.detach().cpu().numpy())
        action_mask_tensor = torch.as_tensor(action_masks, device=self.device)
        return action_values.masked_fill(~action_mask_tensor, -torch.inf).argmax(dim=1)

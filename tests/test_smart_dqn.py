import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN

from gb_tetris_rl.agent.action_masks import (
    canonical_agent_action_mask,
    canonical_agent_action_masks,
)
from gb_tetris_rl.agent.imitation import generate_planner_demonstrations
from gb_tetris_rl.agent.planner import decode_observations, enumerate_agent_afterstates
from gb_tetris_rl.agent.smart_dqn import TetrisAfterstatePolicy, TetrisDQN, TetrisDuelingPolicy
from gb_tetris_rl.game.contracts import (
    AGENT_ACTION_COUNT,
    AGENT_OBSERVATION_SHAPE,
    BOARD_SHAPE,
    EMPTY_HOLD_SLOT,
    encode_agent_observation,
)


class ContractOnlyEnvironment(gym.Env):
    def __init__(self) -> None:
        self.action_space = spaces.Discrete(AGENT_ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=AGENT_OBSERVATION_SHAPE,
            dtype=np.uint8,
        )
        self.observation = encode_agent_observation(
            np.zeros(BOARD_SHAPE, dtype=np.uint8),
            current_piece=5,
            next_piece=0,
            held_piece=EMPTY_HOLD_SLOT,
        )

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        return self.observation.copy(), {}

    def step(self, action):
        return self.observation.copy(), 0.0, False, False, {"action": action}


class TetrisDQNExplorationTests(unittest.TestCase):
    def test_exploration_never_selects_a_noncanonical_action(self) -> None:
        environment = ContractOnlyEnvironment()
        agent = TetrisDQN(
            TetrisDuelingPolicy,
            environment,
            policy_kwargs={"features_extractor_class": _feature_extractor_class()},
            device="cpu",
        )
        observation, _ = environment.reset()
        action_mask = canonical_agent_action_mask(observation)
        agent.exploration_rate = 1.0

        for _ in range(100):
            selected_action, _ = agent.predict(observation, deterministic=False)
            self.assertTrue(action_mask[int(selected_action)])

    def test_warmup_never_selects_a_noncanonical_action(self) -> None:
        environment = ContractOnlyEnvironment()
        agent = TetrisDQN(
            TetrisDuelingPolicy,
            environment,
            policy_kwargs={"features_extractor_class": _feature_extractor_class()},
            device="cpu",
        )
        observation, _ = environment.reset()
        action_mask = canonical_agent_action_mask(observation)
        agent._last_obs = observation.reshape(1, -1)

        for _ in range(100):
            selected_actions, replay_actions = agent._sample_action(
                learning_starts=1_000,
                n_envs=1,
            )
            self.assertTrue(action_mask[int(selected_actions[0])])
            np.testing.assert_array_equal(selected_actions, replay_actions)


class TetrisAfterstatePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TetrisDQN(
            TetrisAfterstatePolicy,
            ContractOnlyEnvironment(),
            buffer_size=1,
            device="cpu",
            seed=0,
        )
        self.observations = generate_planner_demonstrations(
            32,
            seed=5,
            game_count=4,
        ).observations

    def test_scores_exactly_the_placements_that_fit(self) -> None:
        observation_batch = decode_observations(torch.as_tensor(self.observations))
        afterstates = enumerate_agent_afterstates(
            observation_batch.columns,
            observation_batch.current_pieces,
            observation_batch.next_pieces,
            observation_batch.held_pieces,
        )

        with torch.no_grad():
            action_values = self.agent.q_net(torch.as_tensor(self.observations).float())

        torch.testing.assert_close(torch.isfinite(action_values), afterstates.valid)

    def test_predictions_always_fit(self) -> None:
        actions, _ = self.agent.predict(self.observations, deterministic=True)

        observation_batch = decode_observations(torch.as_tensor(self.observations))
        valid = enumerate_agent_afterstates(
            observation_batch.columns,
            observation_batch.current_pieces,
            observation_batch.next_pieces,
            observation_batch.held_pieces,
        ).valid
        self.assertTrue(bool(valid[torch.arange(len(actions)), torch.as_tensor(actions)].all()))

    def test_lost_board_keeps_finite_canonical_values(self) -> None:
        board = np.ones(BOARD_SHAPE, dtype=np.uint8)
        board[:, 4] = 0
        observation = encode_agent_observation(board, 5, 5, 5)

        with torch.no_grad():
            action_values = self.agent.q_net(torch.as_tensor(observation)[None])

        np.testing.assert_array_equal(
            torch.isfinite(action_values[0]).numpy(),
            canonical_agent_action_mask(observation),
        )

    def test_scores_kill_screen_boards_with_uncleared_rows(self) -> None:
        board = np.zeros(BOARD_SHAPE, dtype=np.uint8)
        board[-6:] = 1
        observation = encode_agent_observation(board, 0, 1, EMPTY_HOLD_SLOT)

        with torch.no_grad():
            action_values = self.agent.q_net(torch.as_tensor(observation)[None])

        self.assertTrue(bool(torch.isfinite(action_values).any()))

    def test_saved_policy_reloads_with_identical_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = f"{temporary_directory}/afterstate.zip"
            self.agent.save(model_path)
            reloaded_agent = DQN.load(model_path, device="cpu")

        with torch.no_grad():
            observations = torch.as_tensor(self.observations).float()
            torch.testing.assert_close(
                reloaded_agent.q_net(observations),
                self.agent.q_net(observations),
            )


class LegacyActionMaskTests(unittest.TestCase):
    def test_device_masks_match_the_numpy_contract(self) -> None:
        observations = generate_planner_demonstrations(64, seed=6, game_count=8).observations
        agent = TetrisDQN(
            TetrisDuelingPolicy,
            ContractOnlyEnvironment(),
            policy_kwargs={"features_extractor_class": _feature_extractor_class()},
            buffer_size=1,
            device="cpu",
        )

        with torch.no_grad():
            action_values = agent.q_net(torch.as_tensor(observations))

        np.testing.assert_array_equal(
            torch.isfinite(action_values).numpy(),
            canonical_agent_action_masks(observations),
        )


def _feature_extractor_class():
    from gb_tetris_rl.agent.smart_dqn import TetrisFeatureExtractor

    return TetrisFeatureExtractor


if __name__ == "__main__":
    unittest.main()

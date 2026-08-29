import unittest

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from gb_tetris_rl.agent.action_masks import canonical_agent_action_mask
from gb_tetris_rl.agent.smart_dqn import TetrisDQN, TetrisDuelingPolicy
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


class TetrisDuelingPolicyTests(unittest.TestCase):
    def test_dueling_network_returns_one_value_per_action(self) -> None:
        environment = ContractOnlyEnvironment()
        agent = TetrisDQN(
            TetrisDuelingPolicy,
            environment,
            policy_kwargs={"features_extractor_class": _feature_extractor_class()},
            device="cpu",
        )
        observation, _ = environment.reset()

        with torch.no_grad():
            action_values = agent.q_net(torch.as_tensor(observation).reshape(1, -1))
            target_action_values = agent.q_net_target(torch.as_tensor(observation).reshape(1, -1))

        self.assertEqual(tuple(action_values.shape), (1, AGENT_ACTION_COUNT))
        self.assertTrue(torch.isfinite(action_values[0, 0]))
        self.assertTrue(torch.isneginf(action_values[0, 9]))
        self.assertTrue(torch.isneginf(target_action_values[0, 9]))

    def test_predictions_never_select_a_noncanonical_action(self) -> None:
        environment = ContractOnlyEnvironment()
        agent = TetrisDQN(
            TetrisDuelingPolicy,
            environment,
            policy_kwargs={"features_extractor_class": _feature_extractor_class()},
            device="cpu",
        )
        observation, _ = environment.reset()
        action_mask = canonical_agent_action_mask(observation)

        for _ in range(100):
            selected_action, _ = agent.predict(observation, deterministic=True)
            self.assertTrue(action_mask[int(selected_action)])

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


def _feature_extractor_class():
    from gb_tetris_rl.agent.smart_dqn import TetrisFeatureExtractor

    return TetrisFeatureExtractor


if __name__ == "__main__":
    unittest.main()

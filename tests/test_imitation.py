import unittest

import numpy as np
import torch

from gb_tetris_rl.agent.action_masks import canonical_agent_action_masks
from gb_tetris_rl.agent.imitation import (
    SimulatedGames,
    generate_planner_demonstrations,
    simulate_policy_lines,
)
from gb_tetris_rl.agent.planner import choose_agent_actions, decode_observations
from gb_tetris_rl.game.contracts import EMPTY_HOLD_SLOT


def planner_policy(observations: torch.Tensor) -> torch.Tensor:
    observation_batch = decode_observations(observations)
    return choose_agent_actions(
        observation_batch.columns,
        observation_batch.current_pieces,
        observation_batch.next_pieces,
        observation_batch.held_pieces,
        use_lookahead=False,
    )


class DemonstrationDatasetTests(unittest.TestCase):
    def test_includes_piece_context_and_hold_actions(self) -> None:
        demonstration_dataset = generate_planner_demonstrations(
            100,
            seed=4,
            maximum_episode_pieces=20,
        )

        self.assertEqual(demonstration_dataset.observations.shape, (100, 202))
        self.assertEqual(demonstration_dataset.observations.dtype, np.uint8)
        self.assertEqual(demonstration_dataset.agent_actions.shape, (100,))
        self.assertTrue(np.all(demonstration_dataset.observations[:, 180:187].sum(axis=1) == 1))
        self.assertTrue(np.all(demonstration_dataset.observations[:, 187:194].sum(axis=1) == 1))
        self.assertTrue(np.all(demonstration_dataset.observations[:, 194:202].sum(axis=1) == 1))
        self.assertTrue(np.any(demonstration_dataset.agent_actions >= 40))
        self.assertTrue(np.all(demonstration_dataset.agent_actions < 80))
        demonstration_action_masks = canonical_agent_action_masks(
            demonstration_dataset.observations
        )
        self.assertTrue(
            np.all(
                demonstration_action_masks[
                    np.arange(len(demonstration_dataset.agent_actions)),
                    demonstration_dataset.agent_actions,
                ]
            )
        )

    def test_is_reproducible_for_a_seed(self) -> None:
        first = generate_planner_demonstrations(64, seed=9, game_count=8, use_lookahead=True)
        second = generate_planner_demonstrations(64, seed=9, game_count=8, use_lookahead=True)

        np.testing.assert_array_equal(first.observations, second.observations)
        np.testing.assert_array_equal(first.agent_actions, second.agent_actions)

    def test_labels_policy_visited_boards_with_planner_actions(self) -> None:
        def always_first_action(observations: torch.Tensor) -> torch.Tensor:
            return torch.zeros(len(observations), dtype=torch.int64)

        dataset = generate_planner_demonstrations(
            40,
            seed=2,
            game_count=4,
            behaviour_policy=always_first_action,
        )
        labels = planner_policy(torch.as_tensor(dataset.observations))

        np.testing.assert_array_equal(dataset.agent_actions, labels.numpy())
        # Stacking every piece in the left columns builds towers no planner would.
        self.assertGreater(
            int(dataset.observations[:, :180].reshape(-1, 18, 10)[:, :, 0].sum()), 40
        )


class SimulatedGameTests(unittest.TestCase):
    def test_hold_from_empty_slot_places_the_next_piece_and_draws_two(self) -> None:
        games = SimulatedGames(1, seed=0, maximum_episode_pieces=10)
        games.current_pieces[:] = 2
        games.next_pieces[:] = 5

        games.step(torch.tensor([40]), games.afterstates())

        self.assertEqual(int(games.held_pieces[0]), 2)
        self.assertEqual(int(games.columns.ne(0).sum()), 2)  # the O landed in two columns

    def test_hold_with_a_held_piece_swaps_it_in(self) -> None:
        games = SimulatedGames(1, seed=0, maximum_episode_pieces=10)
        games.current_pieces[:] = 2
        games.next_pieces[:] = 3
        games.held_pieces[:] = 5

        games.step(torch.tensor([40]), games.afterstates())

        self.assertEqual(int(games.held_pieces[0]), 2)
        self.assertEqual(int(games.current_pieces[0]), 3)

    def test_invalid_action_ends_and_restarts_the_game(self) -> None:
        games = SimulatedGames(1, seed=0, maximum_episode_pieces=10)
        games.current_pieces[:] = 5
        games.held_pieces[:] = 0

        game_ended, line_counts = games.step(torch.tensor([9]), games.afterstates())

        self.assertTrue(bool(game_ended[0]))
        self.assertEqual(int(line_counts[0]), 0)
        self.assertEqual(int(games.held_pieces[0]), EMPTY_HOLD_SLOT)
        self.assertEqual(int(games.columns.sum()), 0)

    def test_planner_clears_lines_and_reaches_the_piece_limit(self) -> None:
        line_counts = simulate_policy_lines(
            planner_policy,
            game_count=4,
            seed=1,
            maximum_episode_pieces=100,
        )

        self.assertTrue(np.all(line_counts >= 30))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from gb_tetris_rl.agent.action_masks import canonical_agent_action_masks
from gb_tetris_rl.agent.imitation import generate_planner_demonstrations


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


if __name__ == "__main__":
    unittest.main()

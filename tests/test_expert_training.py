import unittest

import numpy as np

from gb_tetris_rl.expert_training import generate_expert_dataset


class ExpertDatasetTests(unittest.TestCase):
    def test_generates_valid_placement_observations_and_actions(self) -> None:
        expert_dataset = generate_expert_dataset(20, seed=7, maximum_episode_pieces=5)

        self.assertEqual(expert_dataset.observations.shape, (20, 194))
        self.assertEqual(expert_dataset.observations.dtype, np.uint8)
        self.assertEqual(expert_dataset.placement_actions.shape, (20,))
        self.assertTrue(np.all(expert_dataset.placement_actions >= 0))
        self.assertTrue(np.all(expert_dataset.placement_actions < 40))
        self.assertTrue(np.all(expert_dataset.observations[:, 180:187].sum(axis=1) == 1))
        self.assertTrue(np.all(expert_dataset.observations[:, 187:194].sum(axis=1) == 1))

    def test_hold_dataset_includes_held_piece_context_and_hold_actions(self) -> None:
        expert_dataset = generate_expert_dataset(
            100,
            seed=4,
            maximum_episode_pieces=20,
            use_hold=True,
        )

        self.assertEqual(expert_dataset.observations.shape, (100, 202))
        self.assertTrue(np.all(expert_dataset.observations[:, 194:202].sum(axis=1) == 1))
        self.assertTrue(np.any(expert_dataset.placement_actions >= 40))
        self.assertTrue(np.all(expert_dataset.placement_actions < 80))


if __name__ == "__main__":
    unittest.main()

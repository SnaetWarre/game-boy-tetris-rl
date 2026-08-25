import unittest

import numpy as np

from gb_tetris_rl.agent.action_masks import canonical_agent_action_mask
from gb_tetris_rl.game.contracts import (
    BOARD_SHAPE,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    encode_agent_observation,
)


class CanonicalActionMaskTests(unittest.TestCase):
    def test_exposes_only_unique_in_bounds_placements_for_each_piece(self) -> None:
        expected_direct_action_counts = [17, 17, 17, 34, 34, 9, 34]

        for piece_type, expected_action_count in enumerate(expected_direct_action_counts):
            with self.subTest(piece_type=piece_type):
                observation = encode_agent_observation(
                    np.zeros(BOARD_SHAPE, dtype=np.uint8),
                    current_piece=piece_type,
                    next_piece=0,
                    held_piece=EMPTY_HOLD_SLOT,
                )
                action_mask = canonical_agent_action_mask(observation)

                self.assertEqual(
                    int(action_mask[:DIRECT_PLACEMENT_ACTION_COUNT].sum()),
                    expected_action_count,
                )

    def test_removes_duplicate_rotations_and_columns_outside_piece_width(self) -> None:
        observation = encode_agent_observation(
            np.zeros(BOARD_SHAPE, dtype=np.uint8),
            current_piece=5,
            next_piece=0,
            held_piece=EMPTY_HOLD_SLOT,
        )

        action_mask = canonical_agent_action_mask(observation)

        self.assertEqual(int(action_mask[:DIRECT_PLACEMENT_ACTION_COUNT].sum()), 9)
        self.assertEqual(int(action_mask[DIRECT_PLACEMENT_ACTION_COUNT:].sum()), 17)
        self.assertFalse(action_mask[9])
        self.assertFalse(action_mask[10])

    def test_hold_actions_use_held_piece_when_slot_is_occupied(self) -> None:
        observation = encode_agent_observation(
            np.zeros(BOARD_SHAPE, dtype=np.uint8),
            current_piece=0,
            next_piece=3,
            held_piece=5,
        )

        action_mask = canonical_agent_action_mask(observation)

        self.assertEqual(int(action_mask[:DIRECT_PLACEMENT_ACTION_COUNT].sum()), 17)
        self.assertEqual(int(action_mask[DIRECT_PLACEMENT_ACTION_COUNT:].sum()), 9)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from gb_tetris_rl.game.contracts import (
    AGENT_OBSERVATION_SHAPE,
    BOARD_CELL_COUNT,
    BOARD_SHAPE,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    board_from_observation,
    decode_agent_action,
    encode_agent_observation,
    piece_context_from_observation,
)


class AgentContractTests(unittest.TestCase):
    def test_decodes_hold_rotation_and_column(self) -> None:
        placement = decode_agent_action(DIRECT_PLACEMENT_ACTION_COUNT + 23)

        self.assertTrue(placement.uses_hold)
        self.assertEqual(placement.rotation, 2)
        self.assertEqual(placement.column_from_left_wall, 3)

    def test_encodes_board_and_piece_context_once(self) -> None:
        board = np.zeros(BOARD_SHAPE, dtype=np.uint8)
        board[-1, 2] = 1

        observation = encode_agent_observation(
            board,
            current_piece=1,
            next_piece=4,
            held_piece=EMPTY_HOLD_SLOT,
        )

        self.assertEqual(observation.shape, AGENT_OBSERVATION_SHAPE)
        self.assertEqual(int(observation[:BOARD_CELL_COUNT].sum()), 1)
        self.assertEqual(int(observation[BOARD_CELL_COUNT:].sum()), 3)
        np.testing.assert_array_equal(board_from_observation(observation), board)
        piece_context = piece_context_from_observation(observation)
        self.assertEqual(piece_context.current_piece, 1)
        self.assertEqual(piece_context.next_piece, 4)
        self.assertEqual(piece_context.held_piece, EMPTY_HOLD_SLOT)

    def test_rejects_observation_without_piece_context(self) -> None:
        invalid_observation = np.zeros(AGENT_OBSERVATION_SHAPE, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "exactly one current piece"):
            piece_context_from_observation(invalid_observation)


if __name__ == "__main__":
    unittest.main()

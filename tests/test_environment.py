import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gb_tetris_rl.game.contracts import (
    AGENT_OBSERVATION_SHAPE,
    BOARD_SHAPE,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
)
from gb_tetris_rl.game.environment import TetrisEnvironment
from gb_tetris_rl.game.rom import ValidatedRom


class FakeScreen:
    def __init__(self) -> None:
        self.ndarray = np.zeros((144, 160, 4), dtype=np.uint8)


class FakePyBoy:
    latest_instance: "FakePyBoy | None" = None

    def __init__(self, rom_path: str, *, window: str, sound_emulated: bool) -> None:
        self.rom_path = rom_path
        self.window = window
        self.sound_emulated = sound_emulated
        self.screen = FakeScreen()
        self.emulation_speed: int | None = None
        self.was_stopped = False
        FakePyBoy.latest_instance = self

    def set_emulation_speed(self, emulation_speed: int) -> None:
        self.emulation_speed = emulation_speed

    def stop(self, *, save: bool) -> None:
        self.was_stopped = True
        self.saved_state = save


class FakePandorasBlocksAdapter:
    latest_instance: "FakePandorasBlocksAdapter | None" = None

    def __init__(self, pyboy: FakePyBoy) -> None:
        del pyboy
        self.board = np.zeros(BOARD_SHAPE, dtype=np.uint8)
        self.score = 0
        self.cleared_lines = 0
        self.level = 0
        self.current_piece = 0
        self.next_piece = 1
        self.held_piece = EMPTY_HOLD_SLOT
        self.game_is_over = False
        self.last_placement: tuple[int, int, bool, bool] | None = None
        FakePandorasBlocksAdapter.latest_instance = self

    def reset(self, seed: int | None) -> None:
        self.seed = seed

    def read_board(self) -> np.ndarray:
        return self.board.copy()

    def place_piece(
        self,
        target_rotation: int,
        right_moves_from_left_wall: int,
        *,
        use_hold: bool,
        render_frames: bool,
    ) -> bool:
        self.last_placement = (
            target_rotation,
            right_moves_from_left_wall,
            use_hold,
            render_frames,
        )
        return True


def create_environment(**environment_options) -> TetrisEnvironment:
    validated_rom = ValidatedRom(Path("/tmp/PandorasBlocks.gbc"), "DMGTRIS")
    with (
        patch(
            "gb_tetris_rl.game.environment.validate_pandoras_blocks_rom",
            return_value=validated_rom,
        ),
        patch("gb_tetris_rl.game.environment.PyBoy", FakePyBoy),
        patch(
            "gb_tetris_rl.game.environment.PandorasBlocksAdapter",
            FakePandorasBlocksAdapter,
        ),
    ):
        return TetrisEnvironment(validated_rom.path, **environment_options)


class EnvironmentContractTests(unittest.TestCase):
    def test_uses_one_hold_aware_agent_contract(self) -> None:
        environment = create_environment()
        try:
            observation, initial_info = environment.reset(seed=513)
            hold_action = DIRECT_PLACEMENT_ACTION_COUNT + 13
            transition = environment.step(hold_action)
        finally:
            environment.close()

        next_observation, reward, terminated, truncated, transition_info = transition
        adapter = FakePandorasBlocksAdapter.latest_instance
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(observation.shape, AGENT_OBSERVATION_SHAPE)
        self.assertEqual(next_observation.shape, AGENT_OBSERVATION_SHAPE)
        self.assertEqual(initial_info["cleared_lines"], 0)
        self.assertEqual(transition_info["episode_steps"], 1)
        self.assertAlmostEqual(reward, 0.05)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(adapter.seed, 513)
        self.assertEqual(adapter.last_placement, (1, 3, True, False))

    def test_headless_worker_can_share_human_render_mode(self) -> None:
        environment = create_environment(
            render_mode="human",
            display_emulator_window=False,
            emulation_speed=7,
        )
        try:
            environment.step(0)
            fake_pyboy = FakePyBoy.latest_instance
            adapter = FakePandorasBlocksAdapter.latest_instance
        finally:
            environment.close()

        self.assertIsNotNone(fake_pyboy)
        self.assertIsNotNone(adapter)
        assert fake_pyboy is not None
        assert adapter is not None
        self.assertEqual(fake_pyboy.window, "null")
        self.assertEqual(fake_pyboy.emulation_speed, 7)
        assert adapter.last_placement is not None
        self.assertFalse(adapter.last_placement[3])

    def test_game_over_terminates_the_episode(self) -> None:
        environment = create_environment()
        try:
            adapter = FakePandorasBlocksAdapter.latest_instance
            assert adapter is not None
            adapter.game_is_over = True
            _, reward, terminated, _, _ = environment.step(0)
        finally:
            environment.close()

        self.assertTrue(terminated)
        self.assertLess(reward, -4.9)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gb_tetris_rl.actions import TetrisAction
from gb_tetris_rl.environment import (
    TETRIS_BOARD_SHAPE,
    TETRIS_OBSERVATION_SHAPE,
    TetrisEnvironment,
)


class FakeScreen:
    def __init__(self) -> None:
        self.ndarray = np.zeros((144, 160, 4), dtype=np.uint8)


class FakeTetrisWrapper:
    def __init__(self) -> None:
        self.mapping_minimal = np.array([0, 1, 2], dtype=np.uint8)
        self.score = 0
        self.lines = 0
        self.level = 0
        self.board = np.zeros(TETRIS_BOARD_SHAPE, dtype=np.uint8)
        self.is_game_over = False
        self.timer_divider: int | None = None

    def game_area_mapping(self, mapping: np.ndarray, sprite_offset: int) -> None:
        self.selected_mapping = mapping
        self.sprite_offset = sprite_offset

    def start_game(self, timer_div: int | None = None) -> None:
        self.timer_divider = timer_div

    def reset_game(self, timer_div: int | None = None) -> None:
        self.timer_divider = timer_div
        self.score = 0
        self.lines = 0
        self.board.fill(0)
        self.is_game_over = False

    def game_area(self) -> np.ndarray:
        return self.board

    def game_over(self) -> bool:
        return self.is_game_over


class FakePyBoy:
    latest_instance: "FakePyBoy | None" = None

    def __init__(self, rom_path: str, *, window: str, sound_emulated: bool) -> None:
        self.rom_path = rom_path
        self.window = window
        self.sound_emulated = sound_emulated
        self.game_wrapper = FakeTetrisWrapper()
        self.screen = FakeScreen()
        self.last_button: str | None = None
        self.emulation_speed: int | None = None
        self.was_stopped = False
        FakePyBoy.latest_instance = self

    def set_emulation_speed(self, emulation_speed: int) -> None:
        self.emulation_speed = emulation_speed

    def button(self, button_name: str) -> None:
        self.last_button = button_name

    def tick(self, count: int, *, render: bool, sound: bool) -> bool:
        self.tick_count = count
        self.rendered = render
        self.sampled_sound = sound
        return True

    def stop(self, *, save: bool) -> None:
        self.was_stopped = True
        self.saved_state = save


def write_fake_tetris_rom(directory: str) -> Path:
    rom_bytes = bytearray(0x8000)
    rom_bytes[0x134:0x13A] = b"TETRIS"
    rom_path = Path(directory) / "tetris.gb"
    rom_path.write_bytes(rom_bytes)
    return rom_path


class EnvironmentContractTests(unittest.TestCase):
    def test_headless_worker_can_share_human_render_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rom_path = write_fake_tetris_rom(temporary_directory)
            with patch("gb_tetris_rl.environment.PyBoy", FakePyBoy):
                environment = TetrisEnvironment(
                    rom_path,
                    render_mode="human",
                    display_emulator_window=False,
                    emulation_speed=7,
                )
                try:
                    environment.step(TetrisAction.WAIT.value)
                    fake_pyboy = FakePyBoy.latest_instance
                finally:
                    environment.close()

        self.assertIsNotNone(fake_pyboy)
        assert fake_pyboy is not None
        self.assertEqual(fake_pyboy.window, "null")
        self.assertEqual(fake_pyboy.emulation_speed, 7)
        self.assertFalse(fake_pyboy.rendered)

    def test_reset_and_step_follow_the_gymnasium_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rom_path = write_fake_tetris_rom(temporary_directory)
            with patch("gb_tetris_rl.environment.PyBoy", FakePyBoy):
                environment = TetrisEnvironment(rom_path)
                try:
                    observation, initial_info = environment.reset(seed=513)
                    transition = environment.step(TetrisAction.MOVE_LEFT.value)
                finally:
                    environment.close()

        next_observation, reward, terminated, truncated, transition_info = transition
        fake_pyboy = FakePyBoy.latest_instance
        self.assertIsNotNone(fake_pyboy)
        assert fake_pyboy is not None
        self.assertEqual(observation.shape, TETRIS_OBSERVATION_SHAPE)
        self.assertEqual(next_observation.shape, TETRIS_OBSERVATION_SHAPE)
        self.assertEqual(initial_info["cleared_lines"], 0)
        self.assertEqual(transition_info["episode_steps"], 1)
        self.assertAlmostEqual(reward, 0.001)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(fake_pyboy.game_wrapper.timer_divider, 1)
        self.assertEqual(fake_pyboy.last_button, "left")
        self.assertTrue(fake_pyboy.was_stopped)
        self.assertFalse(fake_pyboy.saved_state)

    def test_game_over_terminates_the_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rom_path = write_fake_tetris_rom(temporary_directory)
            with patch("gb_tetris_rl.environment.PyBoy", FakePyBoy):
                environment = TetrisEnvironment(rom_path)
                try:
                    fake_pyboy = FakePyBoy.latest_instance
                    assert fake_pyboy is not None
                    fake_pyboy.game_wrapper.is_game_over = True
                    _, reward, terminated, _, _ = environment.step(TetrisAction.WAIT.value)
                finally:
                    environment.close()

        self.assertTrue(terminated)
        self.assertLess(reward, -4.9)


if __name__ == "__main__":
    unittest.main()

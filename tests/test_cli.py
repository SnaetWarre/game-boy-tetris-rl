import unittest

from gb_tetris_rl.cli import build_argument_parser


class TrainCommandDefaultsTests(unittest.TestCase):
    def test_defaults_to_the_accepted_imitation_only_training_path(self) -> None:
        command_arguments = build_argument_parser().parse_args(["train"])

        self.assertEqual(command_arguments.timesteps, 0)
        self.assertEqual(command_arguments.demonstrations, 50_000)
        self.assertEqual(command_arguments.imitation_epochs, 80)


class DemoCommandSpeedTests(unittest.TestCase):
    def test_defaults_to_normal_emulator_speed(self) -> None:
        command_arguments = build_argument_parser().parse_args(["demo"])

        self.assertFalse(command_arguments.fast)

    def test_accepts_uncapped_fast_mode(self) -> None:
        command_arguments = build_argument_parser().parse_args(["demo", "--fast"])

        self.assertTrue(command_arguments.fast)


if __name__ == "__main__":
    unittest.main()

import unittest

from gb_tetris_rl.cli import build_argument_parser


class TrainCommandDefaultsTests(unittest.TestCase):
    def test_defaults_to_the_accepted_imitation_only_training_path(self) -> None:
        command_arguments = build_argument_parser().parse_args(["train"])

        self.assertEqual(command_arguments.timesteps, 0)
        self.assertEqual(command_arguments.demonstrations, 50_000)
        self.assertEqual(command_arguments.imitation_epochs, 80)
        self.assertEqual(command_arguments.demonstration_episode_pieces, 200)
        self.assertFalse(command_arguments.planner_lookahead)

    def test_accepts_long_horizon_lookahead_demonstrations(self) -> None:
        command_arguments = build_argument_parser().parse_args(
            [
                "train",
                "--demonstration-episode-pieces",
                "4000",
                "--planner-lookahead",
            ]
        )

        self.assertEqual(command_arguments.demonstration_episode_pieces, 4_000)
        self.assertTrue(command_arguments.planner_lookahead)


class DemoCommandSpeedTests(unittest.TestCase):
    def test_defaults_to_normal_emulator_speed(self) -> None:
        command_arguments = build_argument_parser().parse_args(["demo"])

        self.assertFalse(command_arguments.fast)

    def test_accepts_uncapped_fast_mode(self) -> None:
        command_arguments = build_argument_parser().parse_args(["demo", "--fast"])

        self.assertTrue(command_arguments.fast)


class NextPhaseCommandDefaultsTests(unittest.TestCase):
    def test_defaults_to_long_horizon_lookahead_imitation_and_fixed_seed_gate(self) -> None:
        command_arguments = build_argument_parser().parse_args(["next-phase"])

        self.assertEqual(command_arguments.demonstrations, 100_000)
        self.assertEqual(command_arguments.imitation_epochs, 120)
        self.assertEqual(command_arguments.demonstration_episode_pieces, 4_000)
        self.assertEqual(command_arguments.timesteps, 0)
        self.assertEqual(command_arguments.evaluation_seed, 10_000)
        self.assertEqual(command_arguments.evaluation_episodes, 50)
        self.assertFalse(command_arguments.promote)


if __name__ == "__main__":
    unittest.main()

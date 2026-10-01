import unittest

from gb_tetris_rl.cli import build_argument_parser


class TrainCommandDefaultsTests(unittest.TestCase):
    def test_defaults_to_afterstate_imitation_with_one_dagger_round(self) -> None:
        command_arguments = build_argument_parser().parse_args(["train"])

        self.assertEqual(command_arguments.policy, "afterstate")
        self.assertEqual(command_arguments.timesteps, 0)
        self.assertEqual(command_arguments.demonstrations, 100_000)
        self.assertEqual(command_arguments.imitation_epochs, 10)
        self.assertEqual(command_arguments.demonstration_episode_pieces, 4_000)
        self.assertTrue(command_arguments.planner_lookahead)
        self.assertEqual(command_arguments.dagger_rounds, 1)

    def test_can_reproduce_the_v030_training_path(self) -> None:
        command_arguments = build_argument_parser().parse_args(
            [
                "train",
                "--policy",
                "dueling",
                "--demonstrations",
                "50000",
                "--imitation-epochs",
                "80",
                "--demonstration-episode-pieces",
                "200",
                "--no-planner-lookahead",
                "--dagger-rounds",
                "0",
            ]
        )

        self.assertEqual(command_arguments.policy, "dueling")
        self.assertFalse(command_arguments.planner_lookahead)
        self.assertEqual(command_arguments.dagger_rounds, 0)


class DemoCommandSpeedTests(unittest.TestCase):
    def test_defaults_to_normal_emulator_speed(self) -> None:
        command_arguments = build_argument_parser().parse_args(["demo"])

        self.assertFalse(command_arguments.fast)

    def test_accepts_uncapped_fast_mode(self) -> None:
        command_arguments = build_argument_parser().parse_args(["demo", "--fast"])

        self.assertTrue(command_arguments.fast)


class NextPhaseCommandDefaultsTests(unittest.TestCase):
    def test_shares_training_defaults_and_uses_a_capped_fixed_seed_gate(self) -> None:
        command_arguments = build_argument_parser().parse_args(["next-phase"])

        self.assertEqual(command_arguments.policy, "afterstate")
        self.assertEqual(command_arguments.demonstrations, 100_000)
        self.assertEqual(command_arguments.imitation_epochs, 10)
        self.assertEqual(command_arguments.demonstration_episode_pieces, 4_000)
        self.assertTrue(command_arguments.planner_lookahead)
        self.assertEqual(command_arguments.timesteps, 0)
        self.assertEqual(command_arguments.evaluation_seed, 10_000)
        self.assertEqual(command_arguments.evaluation_episodes, 50)
        self.assertEqual(command_arguments.evaluation_target_lines, 500)
        self.assertFalse(command_arguments.promote)


if __name__ == "__main__":
    unittest.main()

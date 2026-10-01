import unittest

from gb_tetris_rl.cli import build_argument_parser
from gb_tetris_rl.commands import _training_config


class TrainingArgumentTests(unittest.TestCase):
    def test_train_and_next_phase_build_the_same_training_config(self) -> None:
        parser = build_argument_parser()

        self.assertEqual(
            _training_config(parser.parse_args(["train"])),
            _training_config(parser.parse_args(["next-phase"])),
        )


if __name__ == "__main__":
    unittest.main()

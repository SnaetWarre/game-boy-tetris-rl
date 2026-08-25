import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gb_tetris_rl.agent.evaluation import (
    EpisodeSummary,
    _action_requires_planner_rescue,
    _write_gif,
    write_evaluation_report,
)
from gb_tetris_rl.game.contracts import BOARD_SHAPE, EMPTY_HOLD_SLOT, encode_agent_observation


class GifRecordingTests(unittest.TestCase):
    def test_writes_all_supplied_frames(self) -> None:
        frames = [np.full((12, 10, 3), fill_value=shade, dtype=np.uint8) for shade in (0, 127, 255)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            recording_path = Path(temporary_directory) / "agent.gif"

            _write_gif(frames, recording_path, frame_duration_ms=50)

            with Image.open(recording_path) as recorded_gif:
                self.assertEqual(recorded_gif.n_frames, 3)
                self.assertEqual(recorded_gif.info["duration"], 50)


class EvaluationReportTests(unittest.TestCase):
    def test_writes_reproducible_aggregate_and_episode_metrics(self) -> None:
        episode_summaries = [
            EpisodeSummary(255, 1, 30, 4, 3, 1),
            EpisodeSummary(1021, 4, 42, 6, 5, 2),
            EpisodeSummary(1533, 6, 49, 8, 7, 0),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "reports" / "evaluation.json"
            saved_report_path = write_evaluation_report(
                episode_summaries,
                report_path,
                model_path="models/candidate.zip",
                starting_seed=10_000,
                evaluation_mode="learned policy only",
            )
            report = json.loads(saved_report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["starting_seed"], 10_000)
        self.assertEqual(report["episode_count"], 3)
        self.assertEqual(report["metrics"]["mean_cleared_lines"], 11 / 3)
        self.assertEqual(report["metrics"]["median_cleared_lines"], 4.0)
        self.assertEqual(report["metrics"]["minimum_cleared_lines"], 1)
        self.assertEqual(report["metrics"]["maximum_cleared_lines"], 6)
        self.assertEqual(report["metrics"]["total_planner_disagreements"], 15)
        self.assertEqual(report["metrics"]["total_planner_rescues"], 3)
        self.assertEqual(report["episodes"][1]["episode_steps"], 42)

    def test_rejects_empty_episode_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty evaluation"):
            write_evaluation_report(
                [],
                "unused.json",
                model_path="unused.zip",
                starting_seed=0,
                evaluation_mode="learned policy only",
            )


class PlannerSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = encode_agent_observation(
            np.zeros(BOARD_SHAPE, dtype=np.uint8),
            current_piece=5,
            next_piece=0,
            held_piece=EMPTY_HOLD_SLOT,
        )
        self.episode_info = {
            "current_piece": 5,
            "next_piece": 0,
            "held_piece": EMPTY_HOLD_SLOT,
        }

    def test_rescues_noncanonical_rotation(self) -> None:
        self.assertTrue(
            _action_requires_planner_rescue(
                self.observation,
                self.episode_info,
                proposed_action=10,
            )
        )

    def test_keeps_canonical_placement_when_next_piece_still_fits(self) -> None:
        self.assertFalse(
            _action_requires_planner_rescue(
                self.observation,
                self.episode_info,
                proposed_action=0,
            )
        )


if __name__ == "__main__":
    unittest.main()

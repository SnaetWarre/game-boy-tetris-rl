import unittest

from gb_tetris_rl.agent.evaluation import EpisodeSummary
from gb_tetris_rl.agent.experiment import compare_neural_evaluations


def _episode_summary(cleared_lines: int) -> EpisodeSummary:
    return EpisodeSummary(
        score=0,
        cleared_lines=cleared_lines,
        episode_steps=0,
        hold_actions=0,
        planner_disagreements=0,
        planner_rescues=0,
    )


class NeuralPromotionGateTests(unittest.TestCase):
    def test_accepts_higher_mean_without_a_median_regression(self) -> None:
        incumbent_summaries = [_episode_summary(lines) for lines in (4, 6, 6, 7)]
        candidate_summaries = [_episode_summary(lines) for lines in (5, 6, 7, 9)]

        decision = compare_neural_evaluations(incumbent_summaries, candidate_summaries)

        self.assertTrue(decision.candidate_qualifies_for_promotion)
        self.assertGreater(
            decision.candidate_metrics.mean_cleared_lines,
            decision.incumbent_metrics.mean_cleared_lines,
        )

    def test_rejects_higher_mean_with_a_lower_median(self) -> None:
        incumbent_summaries = [_episode_summary(lines) for lines in (5, 6, 6, 7, 7)]
        candidate_summaries = [_episode_summary(lines) for lines in (1, 2, 5, 20, 20)]

        decision = compare_neural_evaluations(incumbent_summaries, candidate_summaries)

        self.assertFalse(decision.candidate_qualifies_for_promotion)
        self.assertIn("median regressed", decision.reason)

    def test_requires_matching_episode_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching episode counts"):
            compare_neural_evaluations(
                [_episode_summary(4)],
                [_episode_summary(4), _episode_summary(5)],
            )


if __name__ == "__main__":
    unittest.main()

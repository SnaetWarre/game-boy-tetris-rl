import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean, median

from gb_tetris_rl.agent.evaluation import (
    EpisodeSummary,
    evaluate_agent,
    write_evaluation_report,
)
from gb_tetris_rl.agent.training import TrainingConfig, train_agent


@dataclass(frozen=True)
class NeuralEvaluationMetrics:
    mean_cleared_lines: float
    median_cleared_lines: float
    minimum_cleared_lines: int
    maximum_cleared_lines: int


@dataclass(frozen=True)
class NeuralPromotionDecision:
    incumbent_metrics: NeuralEvaluationMetrics
    candidate_metrics: NeuralEvaluationMetrics
    candidate_qualifies_for_promotion: bool
    reason: str


@dataclass(frozen=True)
class NeuralExperimentConfig:
    training: TrainingConfig
    incumbent_model_path: Path
    evaluation_episode_count: int = 50
    evaluation_seed: int = 10_000
    promote_qualifying_candidate: bool = False


@dataclass(frozen=True)
class NeuralExperimentArtifacts:
    candidate_model_path: Path
    incumbent_report_path: Path
    candidate_report_path: Path
    comparison_path: Path
    promotion_decision: NeuralPromotionDecision
    candidate_was_promoted: bool


def run_neural_experiment(
    rom_path: str | Path,
    run_directory: str | Path,
    config: NeuralExperimentConfig,
) -> NeuralExperimentArtifacts:
    if config.evaluation_episode_count < 1:
        raise ValueError("evaluation_episode_count must be at least 1")

    resolved_run_directory = Path(run_directory).expanduser().resolve()
    resolved_run_directory.mkdir(parents=True, exist_ok=True)
    resolved_incumbent_model_path = config.incumbent_model_path.expanduser().resolve()
    if not resolved_incumbent_model_path.is_file():
        raise ValueError(f"incumbent model file does not exist: {resolved_incumbent_model_path}")
    incumbent_sha256_before_experiment = _file_sha256(resolved_incumbent_model_path)

    training_artifacts = train_agent(rom_path, resolved_run_directory, config.training)
    candidate_model_path = (
        training_artifacts.dqn_model_path
        if training_artifacts.dqn_model_path is not None
        else training_artifacts.imitation_model_path
    )
    if candidate_model_path is None:
        raise RuntimeError("training completed without producing a candidate model")
    if candidate_model_path.resolve() == resolved_incumbent_model_path:
        raise ValueError("candidate and incumbent model paths must be different")

    print("Evaluating incumbent neural policy...")
    incumbent_episode_summaries = evaluate_agent(
        rom_path,
        resolved_incumbent_model_path,
        episode_count=config.evaluation_episode_count,
        show_window=False,
        seed=config.evaluation_seed,
    )
    incumbent_report_path = write_evaluation_report(
        incumbent_episode_summaries,
        resolved_run_directory / "incumbent-neural-eval.json",
        model_path=resolved_incumbent_model_path,
        starting_seed=config.evaluation_seed,
        evaluation_mode="learned policy only",
    )

    print("Evaluating candidate neural policy...")
    candidate_episode_summaries = evaluate_agent(
        rom_path,
        candidate_model_path,
        episode_count=config.evaluation_episode_count,
        show_window=False,
        seed=config.evaluation_seed,
    )
    candidate_report_path = write_evaluation_report(
        candidate_episode_summaries,
        resolved_run_directory / "candidate-neural-eval.json",
        model_path=candidate_model_path,
        starting_seed=config.evaluation_seed,
        evaluation_mode="learned policy only",
    )

    promotion_decision = compare_neural_evaluations(
        incumbent_episode_summaries,
        candidate_episode_summaries,
    )
    candidate_was_promoted = (
        config.promote_qualifying_candidate and promotion_decision.candidate_qualifies_for_promotion
    )
    if candidate_was_promoted:
        _replace_model_atomically(candidate_model_path, resolved_incumbent_model_path)

    comparison_path = resolved_run_directory / "comparison.json"
    comparison_document = {
        "mode": "learned policy only",
        "training": asdict(config.training),
        "evaluation_episode_count": config.evaluation_episode_count,
        "evaluation_seed": config.evaluation_seed,
        "incumbent": {
            "model": str(resolved_incumbent_model_path),
            "sha256_before_experiment": incumbent_sha256_before_experiment,
            "report": str(incumbent_report_path),
            "metrics": asdict(promotion_decision.incumbent_metrics),
        },
        "candidate": {
            "model": str(candidate_model_path),
            "sha256": _file_sha256(candidate_model_path),
            "report": str(candidate_report_path),
            "metrics": asdict(promotion_decision.candidate_metrics),
        },
        "promotion": {
            "requested": config.promote_qualifying_candidate,
            "qualified": promotion_decision.candidate_qualifies_for_promotion,
            "promoted": candidate_was_promoted,
            "reason": promotion_decision.reason,
        },
    }
    comparison_path.write_text(
        json.dumps(comparison_document, indent=2) + "\n",
        encoding="utf-8",
    )

    return NeuralExperimentArtifacts(
        candidate_model_path=candidate_model_path,
        incumbent_report_path=incumbent_report_path,
        candidate_report_path=candidate_report_path,
        comparison_path=comparison_path,
        promotion_decision=promotion_decision,
        candidate_was_promoted=candidate_was_promoted,
    )


def compare_neural_evaluations(
    incumbent_episode_summaries: list[EpisodeSummary],
    candidate_episode_summaries: list[EpisodeSummary],
) -> NeuralPromotionDecision:
    if not incumbent_episode_summaries or not candidate_episode_summaries:
        raise ValueError("neural model comparison requires non-empty evaluations")
    if len(incumbent_episode_summaries) != len(candidate_episode_summaries):
        raise ValueError("neural model comparison requires matching episode counts")

    incumbent_metrics = _summarize_cleared_lines(incumbent_episode_summaries)
    candidate_metrics = _summarize_cleared_lines(candidate_episode_summaries)
    candidate_qualifies_for_promotion = (
        candidate_metrics.mean_cleared_lines > incumbent_metrics.mean_cleared_lines
        and candidate_metrics.median_cleared_lines >= incumbent_metrics.median_cleared_lines
    )
    if candidate_qualifies_for_promotion:
        reason = "candidate mean improved without reducing the median"
    elif candidate_metrics.mean_cleared_lines <= incumbent_metrics.mean_cleared_lines:
        reason = "candidate mean did not beat the incumbent"
    else:
        reason = "candidate median regressed despite a higher mean"

    return NeuralPromotionDecision(
        incumbent_metrics=incumbent_metrics,
        candidate_metrics=candidate_metrics,
        candidate_qualifies_for_promotion=candidate_qualifies_for_promotion,
        reason=reason,
    )


def _summarize_cleared_lines(
    episode_summaries: list[EpisodeSummary],
) -> NeuralEvaluationMetrics:
    cleared_line_counts = [summary.cleared_lines for summary in episode_summaries]
    return NeuralEvaluationMetrics(
        mean_cleared_lines=float(mean(cleared_line_counts)),
        median_cleared_lines=float(median(cleared_line_counts)),
        minimum_cleared_lines=min(cleared_line_counts),
        maximum_cleared_lines=max(cleared_line_counts),
    )


def _replace_model_atomically(source_model_path: Path, destination_model_path: Path) -> None:
    destination_model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file_descriptor, temporary_file_name = tempfile.mkstemp(
        dir=destination_model_path.parent,
        prefix=f".{destination_model_path.name}.",
        suffix=".tmp",
    )
    os.close(temporary_file_descriptor)
    temporary_model_path = Path(temporary_file_name)
    try:
        shutil.copy2(source_model_path, temporary_model_path)
        temporary_model_path.replace(destination_model_path)
    finally:
        temporary_model_path.unlink(missing_ok=True)


def _file_sha256(file_path: Path) -> str:
    digest = sha256()
    with file_path.open("rb") as model_file:
        for file_chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(file_chunk)
    return digest.hexdigest()

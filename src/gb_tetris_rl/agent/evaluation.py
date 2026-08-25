import json
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gb_tetris_rl.agent.planner import choose_agent_action, enumerate_placements
from gb_tetris_rl.game.contracts import (
    AGENT_ACTION_COUNT,
    AGENT_OBSERVATION_SHAPE,
    DIRECT_PLACEMENT_ACTION_COUNT,
    EMPTY_HOLD_SLOT,
    EpisodeInfo,
    board_from_observation,
    decode_agent_action,
)
from gb_tetris_rl.game.environment import TetrisEnvironment


@dataclass(frozen=True)
class EpisodeSummary:
    score: int
    cleared_lines: int
    episode_steps: int
    hold_actions: int
    planner_disagreements: int
    planner_rescues: int


def evaluate_agent(
    rom_path: str | Path,
    model_path: str | Path,
    *,
    episode_count: int,
    show_window: bool,
    seed: int,
    use_planner_safety: bool = False,
    use_planner_override: bool = False,
    target_lines: int | None = None,
    play_forever: bool = False,
    recording_path: str | Path | None = None,
    capture_every_n_steps: int = 2,
) -> list[EpisodeSummary]:
    """Evaluate a learned policy, optionally guarded by the deterministic planner."""
    if episode_count < 1:
        raise ValueError("episode_count must be at least 1")
    if capture_every_n_steps < 1:
        raise ValueError("capture_every_n_steps must be at least 1")
    if target_lines is not None and target_lines < 1:
        raise ValueError("target_lines must be at least 1")
    if use_planner_safety and use_planner_override:
        raise ValueError("planner safety and planner override are mutually exclusive")

    dqn_agent = _load_compatible_agent(model_path)
    render_mode = "human" if show_window else "rgb_array" if recording_path else None
    environment = TetrisEnvironment(rom_path, render_mode=render_mode)
    recorded_frames: list[NDArray[np.uint8]] = []
    episode_summaries: list[EpisodeSummary] = []
    maximum_recorded_frames = 3_000

    try:
        episode_indices = count() if play_forever else range(episode_count)
        for episode_index in episode_indices:
            observation, episode_info = environment.reset(seed=seed + episode_index)
            episode_step_count = 0
            planner_disagreement_count = 0
            planner_rescue_count = 0
            hold_action_count = 0

            while True:
                predicted_action, _ = dqn_agent.predict(observation, deterministic=True)
                selected_action = int(np.asarray(predicted_action).item())
                if use_planner_safety or use_planner_override:
                    planner_action = _choose_safe_action(observation, episode_info)
                    if planner_action != selected_action:
                        planner_disagreement_count += 1
                        action_requires_rescue = _action_requires_planner_rescue(
                            observation,
                            episode_info,
                            selected_action,
                        )
                        if use_planner_override or action_requires_rescue:
                            planner_rescue_count += int(action_requires_rescue)
                            selected_action = planner_action

                if selected_action >= DIRECT_PLACEMENT_ACTION_COUNT:
                    hold_action_count += 1
                observation, _, terminated, truncated, episode_info = environment.step(
                    selected_action
                )
                episode_step_count += 1

                if _should_capture_frame(
                    recording_path,
                    episode_step_count,
                    capture_every_n_steps,
                    len(recorded_frames),
                    maximum_recorded_frames,
                ):
                    rendered_frame = environment.render()
                    if rendered_frame is not None:
                        recorded_frames.append(rendered_frame)

                target_was_reached = (
                    target_lines is not None and episode_info["cleared_lines"] >= target_lines
                )
                if terminated or truncated or target_was_reached:
                    break

            episode_summary = _summarize_episode(
                episode_info,
                hold_action_count,
                planner_disagreement_count,
                planner_rescue_count,
            )
            if play_forever:
                episode_summaries[:] = [episode_summary]
            else:
                episode_summaries.append(episode_summary)
    finally:
        environment.close()

    if recording_path is not None:
        _write_gif(
            recorded_frames,
            recording_path,
            frame_duration_ms=round((1_000 / 30) * capture_every_n_steps),
        )
    return episode_summaries


def _load_compatible_agent(model_path: str | Path):
    try:
        from stable_baselines3 import DQN
    except ImportError as import_error:
        raise RuntimeError("evaluation dependencies are missing; run: uv sync") from import_error

    resolved_model_path = Path(model_path).expanduser().resolve()
    if not resolved_model_path.is_file():
        raise ValueError(f"model file does not exist: {resolved_model_path}")
    dqn_agent = DQN.load(str(resolved_model_path))

    observation_shape = dqn_agent.observation_space.shape
    action_count = getattr(dqn_agent.action_space, "n", None)
    if observation_shape != AGENT_OBSERVATION_SHAPE or action_count != AGENT_ACTION_COUNT:
        raise ValueError(
            f"model uses observation/action contract {observation_shape}/{action_count}; "
            f"this project expects {AGENT_OBSERVATION_SHAPE}/{AGENT_ACTION_COUNT}"
        )
    return dqn_agent


def _choose_safe_action(observation, episode_info: EpisodeInfo) -> int:
    return choose_agent_action(
        board_from_observation(observation),
        episode_info["current_piece"],
        episode_info["next_piece"],
        episode_info["held_piece"],
    )


def _action_requires_planner_rescue(
    observation,
    episode_info: EpisodeInfo,
    proposed_action: int,
) -> bool:
    placement_decision = decode_agent_action(proposed_action)
    direct_placement_action = proposed_action % DIRECT_PLACEMENT_ACTION_COUNT
    piece_to_place = episode_info["current_piece"]
    if placement_decision.uses_hold:
        piece_to_place = (
            episode_info["next_piece"]
            if episode_info["held_piece"] == EMPTY_HOLD_SLOT
            else episode_info["held_piece"]
        )
    matching_placements = [
        placement
        for placement in enumerate_placements(board_from_observation(observation), piece_to_place)
        if placement.placement_action == direct_placement_action
    ]
    if not matching_placements:
        return True
    if placement_decision.uses_hold and episode_info["held_piece"] == EMPTY_HOLD_SLOT:
        return False
    return not enumerate_placements(matching_placements[0].board, episode_info["next_piece"])


def _summarize_episode(
    episode_info: EpisodeInfo,
    hold_action_count: int,
    planner_disagreement_count: int,
    planner_rescue_count: int,
) -> EpisodeSummary:
    return EpisodeSummary(
        score=episode_info["score"],
        cleared_lines=episode_info["cleared_lines"],
        episode_steps=episode_info["episode_steps"],
        hold_actions=hold_action_count,
        planner_disagreements=planner_disagreement_count,
        planner_rescues=planner_rescue_count,
    )


def write_evaluation_report(
    episode_summaries: list[EpisodeSummary],
    report_path: str | Path,
    *,
    model_path: str | Path,
    starting_seed: int,
    evaluation_mode: str,
) -> Path:
    if not episode_summaries:
        raise ValueError("cannot report an empty evaluation")
    cleared_line_counts = np.asarray(
        [episode.cleared_lines for episode in episode_summaries],
        dtype=np.float64,
    )
    resolved_report_path = Path(report_path).expanduser().resolve()
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(Path(model_path).expanduser().resolve()),
        "mode": evaluation_mode,
        "starting_seed": starting_seed,
        "episode_count": len(episode_summaries),
        "metrics": {
            "mean_cleared_lines": float(cleared_line_counts.mean()),
            "median_cleared_lines": float(np.median(cleared_line_counts)),
            "minimum_cleared_lines": int(cleared_line_counts.min()),
            "maximum_cleared_lines": int(cleared_line_counts.max()),
            "total_planner_disagreements": sum(
                episode.planner_disagreements for episode in episode_summaries
            ),
            "total_planner_rescues": sum(episode.planner_rescues for episode in episode_summaries),
        },
        "episodes": [episode.__dict__ for episode in episode_summaries],
    }
    resolved_report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return resolved_report_path


def _should_capture_frame(
    recording_path: str | Path | None,
    episode_step_count: int,
    capture_every_n_steps: int,
    recorded_frame_count: int,
    maximum_recorded_frames: int,
) -> bool:
    return (
        recording_path is not None
        and episode_step_count % capture_every_n_steps == 0
        and recorded_frame_count < maximum_recorded_frames
    )


def _write_gif(
    frames: list[NDArray[np.uint8]],
    recording_path: str | Path,
    *,
    frame_duration_ms: int,
) -> None:
    if not frames:
        raise RuntimeError("no rendered frames were available for recording")
    try:
        from PIL import Image
    except ImportError as import_error:
        raise RuntimeError("GIF recording dependencies are missing; run: uv sync") from import_error

    resolved_recording_path = Path(recording_path).expanduser().resolve()
    resolved_recording_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        resolved_recording_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )

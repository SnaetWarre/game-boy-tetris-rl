import argparse
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version

from gb_tetris_rl.agent.evaluation import (
    EpisodeSummary,
    evaluate_agent,
    write_evaluation_report,
)
from gb_tetris_rl.agent.experiment import NeuralExperimentConfig, run_neural_experiment
from gb_tetris_rl.agent.planner import choose_agent_action
from gb_tetris_rl.agent.training import TrainingConfig, train_agent
from gb_tetris_rl.game.contracts import DIRECT_PLACEMENT_ACTION_COUNT, board_from_observation
from gb_tetris_rl.game.environment import TetrisEnvironment
from gb_tetris_rl.game.rom import validate_pandoras_blocks_rom
from gb_tetris_rl.homebrew import download_pandoras_blocks


def run_bootstrap_command(command_arguments: argparse.Namespace) -> None:
    homebrew_files = download_pandoras_blocks(command_arguments.output_dir)
    print(f"ROM: {homebrew_files.rom_path}")
    print(f"Symbols: {homebrew_files.symbols_path}")


def run_doctor_command(command_arguments: argparse.Namespace) -> None:
    validated_rom = validate_pandoras_blocks_rom(command_arguments.rom)
    print(f"ROM: {validated_rom.path}")
    print(f"Cartridge title: {validated_rom.cartridge_title}")
    for package_name in ("pyboy", "gymnasium", "numpy", "stable-baselines3", "torch"):
        try:
            installed_version = version(package_name)
        except PackageNotFoundError:
            installed_version = "not installed"
        print(f"{package_name}: {installed_version}")

    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    environment = TetrisEnvironment(validated_rom.path)
    try:
        observation, initial_episode_info = environment.reset(seed=0)
        _, reward, terminated, truncated, transition_info = environment.step(0)
    finally:
        environment.close()
    print(f"Observation: shape={observation.shape}, dtype={observation.dtype}")
    print(f"Initial state: {initial_episode_info}")
    print(
        "Transition: "
        f"reward={reward:.4f}, terminated={terminated}, truncated={truncated}, "
        f"state={transition_info}"
    )


def run_planner_command(command_arguments: argparse.Namespace) -> None:
    environment = TetrisEnvironment(
        command_arguments.rom,
        render_mode="human" if command_arguments.window else None,
        emulation_speed=command_arguments.speed if command_arguments.window else 0,
    )
    completed_episode_count = 0
    total_hold_action_count = 0
    try:
        while True:
            observation, episode_info = environment.reset(
                seed=command_arguments.seed + completed_episode_count
            )
            for _ in range(command_arguments.maximum_pieces):
                planner_action = choose_agent_action(
                    board_from_observation(observation),
                    episode_info["current_piece"],
                    episode_info["next_piece"],
                    episode_info["held_piece"],
                )
                total_hold_action_count += int(planner_action >= DIRECT_PLACEMENT_ACTION_COUNT)
                observation, _, terminated, truncated, episode_info = environment.step(
                    planner_action
                )
                target_was_reached = (
                    not command_arguments.forever
                    and episode_info["cleared_lines"] >= command_arguments.target_lines
                )
                if target_was_reached or terminated or truncated:
                    break
            completed_episode_count += 1
            if not command_arguments.forever:
                break
    finally:
        environment.close()

    print(
        "Planner: "
        f"target_reached={episode_info['cleared_lines'] >= command_arguments.target_lines}, "
        f"pieces={episode_info['episode_steps']}, score={episode_info['score']}, "
        f"lines={episode_info['cleared_lines']}, holes={episode_info['holes']}, "
        f"holds={total_hold_action_count}"
    )


def run_train_command(command_arguments: argparse.Namespace) -> None:
    training_config = replace(
        _training_config(command_arguments),
        show_window=command_arguments.window,
        emulation_speed=command_arguments.speed,
    )
    training_artifacts = train_agent(
        command_arguments.rom,
        command_arguments.run_dir,
        training_config,
    )
    if training_artifacts.imitation_model_path is not None:
        print(f"Imitation model: {training_artifacts.imitation_model_path}")
    if training_artifacts.dqn_model_path is not None:
        print(f"DQN model: {training_artifacts.dqn_model_path}")


def run_next_phase_command(command_arguments: argparse.Namespace) -> None:
    training_config = _training_config(command_arguments)
    experiment_config = NeuralExperimentConfig(
        training=training_config,
        incumbent_model_path=command_arguments.incumbent_model,
        evaluation_episode_count=command_arguments.evaluation_episodes,
        evaluation_seed=command_arguments.evaluation_seed,
        evaluation_target_lines=command_arguments.evaluation_target_lines,
        promote_qualifying_candidate=command_arguments.promote,
    )
    experiment_artifacts = run_neural_experiment(
        command_arguments.rom,
        command_arguments.run_dir,
        experiment_config,
    )
    promotion_decision = experiment_artifacts.promotion_decision
    incumbent_metrics = promotion_decision.incumbent_metrics
    candidate_metrics = promotion_decision.candidate_metrics
    print(
        "Incumbent neural result: "
        f"mean={incumbent_metrics.mean_cleared_lines:.2f}, "
        f"median={incumbent_metrics.median_cleared_lines:.2f}, "
        f"range={incumbent_metrics.minimum_cleared_lines}-"
        f"{incumbent_metrics.maximum_cleared_lines}"
    )
    print(
        "Candidate neural result: "
        f"mean={candidate_metrics.mean_cleared_lines:.2f}, "
        f"median={candidate_metrics.median_cleared_lines:.2f}, "
        f"range={candidate_metrics.minimum_cleared_lines}-"
        f"{candidate_metrics.maximum_cleared_lines}"
    )
    print(f"Promotion gate: {promotion_decision.reason}")
    print(f"Candidate model: {experiment_artifacts.candidate_model_path}")
    print(f"Comparison report: {experiment_artifacts.comparison_path}")
    if experiment_artifacts.candidate_was_promoted:
        print(f"Promoted candidate to: {command_arguments.incumbent_model.expanduser().resolve()}")
    elif command_arguments.promote:
        print("Incumbent unchanged because the candidate did not pass the promotion gate.")


def run_evaluate_command(command_arguments: argparse.Namespace) -> None:
    if command_arguments.planner_safety and command_arguments.planner_override:
        raise ValueError("--planner-safety and --planner-override are mutually exclusive")
    if command_arguments.planner_override:
        evaluation_mode = "learned policy + planner override"
    elif command_arguments.planner_safety:
        evaluation_mode = "learned policy + planner safety"
    else:
        evaluation_mode = "learned policy only"
    print(f"Evaluation mode: {evaluation_mode}")
    episode_summaries = evaluate_agent(
        command_arguments.rom,
        command_arguments.model,
        episode_count=command_arguments.episodes,
        show_window=command_arguments.window,
        seed=command_arguments.seed,
        use_planner_safety=command_arguments.planner_safety,
        use_planner_override=command_arguments.planner_override,
        target_lines=command_arguments.target_lines,
        recording_path=command_arguments.record,
        capture_every_n_steps=command_arguments.record_every,
        device=command_arguments.device,
    )
    _print_episode_summaries(episode_summaries)
    if command_arguments.report is not None:
        saved_report_path = write_evaluation_report(
            episode_summaries,
            command_arguments.report,
            model_path=command_arguments.model,
            starting_seed=command_arguments.seed,
            evaluation_mode=evaluation_mode,
        )
        print(f"Saved evaluation report: {saved_report_path}")
    if command_arguments.record is not None:
        print(f"Saved recording: {command_arguments.record.expanduser().resolve()}")


def run_demo_command(command_arguments: argparse.Namespace) -> None:
    uses_planner_override = not command_arguments.neural_only
    demo_mode = (
        "learned policy + explicit planner override"
        if uses_planner_override
        else "learned policy only"
    )
    print(f"Demo mode: {demo_mode}")
    print("Every planner disagreement is reported; override mode follows the planner.")
    if command_arguments.fast:
        print("Emulator speed: uncapped")
    episode_summaries = evaluate_agent(
        command_arguments.rom,
        command_arguments.model,
        episode_count=1,
        show_window=not command_arguments.headless,
        seed=command_arguments.seed,
        emulation_speed=0 if command_arguments.fast else None,
        use_planner_override=uses_planner_override,
        target_lines=None if command_arguments.forever else command_arguments.target_lines,
        play_forever=command_arguments.forever,
        device=command_arguments.device,
    )
    _print_episode_summaries(episode_summaries)


def _training_config(command_arguments: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        total_timesteps=command_arguments.timesteps,
        seed=command_arguments.seed,
        device=command_arguments.device,
        environment_count=command_arguments.envs,
        policy_architecture=command_arguments.policy,
        demonstration_count=command_arguments.demonstrations,
        imitation_epoch_count=command_arguments.imitation_epochs,
        demonstration_episode_piece_limit=command_arguments.demonstration_episode_pieces,
        planner_lookahead=command_arguments.planner_lookahead,
        dagger_round_count=command_arguments.dagger_rounds,
    )


def _print_episode_summaries(episode_summaries: list[EpisodeSummary]) -> None:
    for episode_number, episode_summary in enumerate(episode_summaries, start=1):
        print(
            f"Episode {episode_number}: score={episode_summary.score}, "
            f"lines={episode_summary.cleared_lines}, "
            f"pieces={episode_summary.episode_steps}, "
            f"holds={episode_summary.hold_actions}, "
            f"planner_disagreements={episode_summary.planner_disagreements}, "
            f"planner_rescues={episode_summary.planner_rescues}"
        )

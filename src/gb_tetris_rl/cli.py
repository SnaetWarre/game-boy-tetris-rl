import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from gb_tetris_rl.roms import RomValidationError, validate_tetris_rom


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gb-tetris-rl",
        description="Train and inspect a reinforcement-learning agent for Game Boy Tetris.",
    )
    command_parsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = command_parsers.add_parser(
        "doctor", help="validate the ROM, dependencies, and one environment transition"
    )
    _add_rom_argument(doctor_parser)

    random_parser = command_parsers.add_parser(
        "random", help="run a random policy as an environment smoke test"
    )
    _add_rom_argument(random_parser)
    random_parser.add_argument("--steps", type=_positive_integer, default=2_000)
    random_parser.add_argument("--seed", type=int, default=0)
    random_parser.add_argument("--window", action="store_true")

    train_parser = command_parsers.add_parser("train", help="train a DQN agent")
    _add_rom_argument(train_parser)
    train_parser.add_argument("--timesteps", type=_positive_integer, default=250_000)
    train_parser.add_argument("--output", type=Path, default=Path("models/tetris-dqn"))
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")

    watch_parser = command_parsers.add_parser(
        "watch", help="evaluate a trained model and optionally record a GIF"
    )
    _add_rom_argument(watch_parser)
    watch_parser.add_argument("--model", type=Path, required=True)
    watch_parser.add_argument("--episodes", type=_positive_integer, default=3)
    watch_parser.add_argument("--record", type=Path)
    watch_parser.add_argument(
        "--record-every",
        type=_positive_integer,
        default=2,
        help="capture one GIF frame after this many agent actions",
    )
    watch_parser.add_argument("--seed", type=int, default=10_000)
    watch_parser.add_argument("--window", action="store_true")

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        if arguments.command == "doctor":
            _run_doctor(arguments.rom)
        elif arguments.command == "random":
            _run_random_policy(
                arguments.rom,
                maximum_steps=arguments.steps,
                seed=arguments.seed,
                show_window=arguments.window,
            )
        elif arguments.command == "train":
            _run_training(arguments)
        elif arguments.command == "watch":
            _run_playback(arguments)
    except (RomValidationError, RuntimeError, ValueError) as command_error:
        parser.exit(status=2, message=f"error: {command_error}\n")


def _add_rom_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rom",
        type=Path,
        required=True,
        help="path to a legally supplied Game Boy Tetris ROM",
    )


def _positive_integer(raw_value: str) -> int:
    parsed_value = int(raw_value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def _run_doctor(rom_path: Path) -> None:
    validated_rom = validate_tetris_rom(rom_path)
    print(f"ROM: {validated_rom.path}")
    print(f"Cartridge title: {validated_rom.cartridge_title}")
    for package_name in ("pyboy", "gymnasium", "numpy", "stable-baselines3"):
        try:
            installed_version = version(package_name)
        except PackageNotFoundError:
            installed_version = "not installed (optional for stable-baselines3)"
        print(f"{package_name}: {installed_version}")

    from gb_tetris_rl.environment import TetrisEnvironment

    environment = TetrisEnvironment(validated_rom.path)
    try:
        observation, initial_info = environment.reset(seed=0)
        _, reward, terminated, truncated, transition_info = environment.step(0)
    finally:
        environment.close()
    print(f"Observation: shape={observation.shape}, dtype={observation.dtype}")
    print(f"Initial state: {initial_info}")
    print(
        "Transition: "
        f"reward={reward:.4f}, terminated={terminated}, truncated={truncated}, "
        f"state={transition_info}"
    )


def _run_random_policy(
    rom_path: Path,
    *,
    maximum_steps: int,
    seed: int,
    show_window: bool,
) -> None:
    from gb_tetris_rl.environment import TetrisEnvironment

    render_mode = "human" if show_window else None
    environment = TetrisEnvironment(rom_path, render_mode=render_mode)
    total_reward = 0.0
    completed_steps = 0
    try:
        _, episode_info = environment.reset(seed=seed)
        for _ in range(maximum_steps):
            action = environment.action_space.sample()
            _, reward, terminated, truncated, episode_info = environment.step(action)
            total_reward += reward
            completed_steps += 1
            if terminated or truncated:
                break
    finally:
        environment.close()

    print(
        f"Random policy: steps={completed_steps}, reward={total_reward:.3f}, "
        f"score={episode_info['score']}, lines={episode_info['cleared_lines']}, "
        f"holes={episode_info['holes']}"
    )


def _run_training(arguments: argparse.Namespace) -> None:
    from gb_tetris_rl.training import train_dqn_agent

    saved_model_path = train_dqn_agent(
        arguments.rom,
        arguments.output,
        total_timesteps=arguments.timesteps,
        seed=arguments.seed,
        device=arguments.device,
    )
    print(f"Saved model: {saved_model_path}")


def _run_playback(arguments: argparse.Namespace) -> None:
    from gb_tetris_rl.playback import watch_trained_agent

    episode_summaries = watch_trained_agent(
        arguments.rom,
        arguments.model,
        episode_count=arguments.episodes,
        show_window=arguments.window,
        recording_path=arguments.record,
        capture_every_n_steps=arguments.record_every,
        seed=arguments.seed,
    )
    for episode_number, episode_info in enumerate(episode_summaries, start=1):
        print(
            f"Episode {episode_number}: score={episode_info['score']}, "
            f"lines={episode_info['cleared_lines']}, steps={episode_info['episode_steps']}"
        )
    if arguments.record is not None:
        print(f"Saved recording: {arguments.record.expanduser().resolve()}")


if __name__ == "__main__":
    main()

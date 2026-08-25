import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

from gb_tetris_rl.commands import (
    run_bootstrap_command,
    run_demo_command,
    run_doctor_command,
    run_evaluate_command,
    run_planner_command,
    run_train_command,
)
from gb_tetris_rl.game.rom import RomValidationError

DEFAULT_ROM_PATH = Path("roms/PandorasBlocks.gbc")
DEFAULT_DEMO_MODEL_PATH = Path("models/demo-agent.zip")

CommandRunner = Callable[[argparse.Namespace], None]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gb-tetris-rl",
        description=(
            "Train and evaluate one hold-aware DQN agent in the open-source Pandora's Blocks ROM."
        ),
    )
    command_parsers = parser.add_subparsers(dest="command", required=True)
    _add_bootstrap_command(command_parsers)
    _add_doctor_command(command_parsers)
    _add_demo_command(command_parsers)
    _add_planner_command(command_parsers)
    _add_train_command(command_parsers)
    _add_evaluate_command(command_parsers)
    return parser


def main() -> None:
    parser = build_argument_parser()
    command_arguments = parser.parse_args()
    command_runner = cast(CommandRunner, command_arguments.command_runner)
    try:
        command_runner(command_arguments)
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RomValidationError, RuntimeError, ValueError) as command_error:
        parser.exit(status=2, message=f"error: {command_error}\n")


def _add_bootstrap_command(command_parsers) -> None:
    bootstrap_parser = command_parsers.add_parser(
        "bootstrap",
        help="download the checksum-pinned GPL ROM and symbols",
    )
    bootstrap_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("roms"),
        help="download directory (default: roms)",
    )
    bootstrap_parser.set_defaults(command_runner=run_bootstrap_command)


def _add_doctor_command(command_parsers) -> None:
    doctor_parser = command_parsers.add_parser(
        "doctor",
        help="validate dependencies, ROM state, and one environment transition",
    )
    _add_rom_argument(doctor_parser)
    doctor_parser.set_defaults(command_runner=run_doctor_command)


def _add_demo_command(command_parsers) -> None:
    demo_parser = command_parsers.add_parser(
        "demo",
        help="run the canonical hold-aware model with an explicit planner override",
    )
    _add_rom_argument(demo_parser)
    demo_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_DEMO_MODEL_PATH,
        help=f"model checkpoint (default: {DEFAULT_DEMO_MODEL_PATH})",
    )
    demo_parser.add_argument("--target-lines", type=_positive_integer, default=40)
    demo_parser.add_argument("--seed", type=int, default=10_000)
    demo_parser.add_argument(
        "--neural-only",
        action="store_true",
        help="disable the planner override and show only learned policy performance",
    )
    demo_parser.add_argument(
        "--headless",
        action="store_true",
        help="do not open the emulator window",
    )
    demo_parser.add_argument(
        "--forever",
        action="store_true",
        help="start a new seeded game after every top-out until interrupted",
    )
    demo_parser.set_defaults(command_runner=run_demo_command)


def _add_planner_command(command_parsers) -> None:
    planner_parser = command_parsers.add_parser(
        "planner",
        help="run the deterministic planner without a neural model",
    )
    _add_rom_argument(planner_parser)
    planner_parser.add_argument("--target-lines", type=_positive_integer, default=40)
    planner_parser.add_argument("--maximum-pieces", type=_positive_integer, default=2_000)
    planner_parser.add_argument("--seed", type=int, default=0)
    planner_parser.add_argument("--window", action="store_true")
    planner_parser.add_argument("--forever", action="store_true")
    planner_parser.add_argument(
        "--speed",
        type=_nonnegative_integer,
        default=1,
        help="visible emulator speed multiplier; 0 removes the frame limiter",
    )
    planner_parser.set_defaults(command_runner=run_planner_command)


def _add_train_command(command_parsers) -> None:
    train_parser = command_parsers.add_parser(
        "train",
        help="run planner imitation with optional DQN fine-tuning",
    )
    _add_rom_argument(train_parser)
    train_parser.add_argument("--run-dir", type=Path, default=Path("models/runs/latest"))
    train_parser.add_argument(
        "--timesteps",
        type=_nonnegative_integer,
        default=0,
        help="optional emulator DQN steps; disabled by default because validation regressed",
    )
    train_parser.add_argument("--demonstrations", type=_nonnegative_integer, default=50_000)
    train_parser.add_argument("--imitation-epochs", type=_nonnegative_integer, default=80)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument(
        "--envs",
        type=_positive_integer,
        default=4,
        help="parallel PyBoy workers feeding the shared DQN",
    )
    train_parser.add_argument("--window", action="store_true")
    train_parser.add_argument(
        "--speed",
        type=_nonnegative_integer,
        default=0,
        help="visible worker speed; 0 removes the frame limiter",
    )
    train_parser.set_defaults(command_runner=run_train_command)


def _add_evaluate_command(command_parsers) -> None:
    evaluate_parser = command_parsers.add_parser(
        "evaluate",
        help="measure a model with optional planner safety or override",
    )
    _add_rom_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_DEMO_MODEL_PATH,
        help=f"model checkpoint (default: {DEFAULT_DEMO_MODEL_PATH})",
    )
    evaluate_parser.add_argument("--episodes", type=_positive_integer, default=5)
    evaluate_parser.add_argument("--seed", type=int, default=10_000)
    evaluate_parser.add_argument("--target-lines", type=_positive_integer)
    evaluate_parser.add_argument("--window", action="store_true")
    evaluate_parser.add_argument(
        "--planner-safety",
        action="store_true",
        help="replace only noncanonical actions or placements that force a known top-out",
    )
    evaluate_parser.add_argument(
        "--planner-override",
        action="store_true",
        help="replace every neural action that differs from the planner",
    )
    evaluate_parser.add_argument("--report", type=Path, help="write aggregate metrics as JSON")
    evaluate_parser.add_argument("--record", type=Path)
    evaluate_parser.add_argument(
        "--record-every",
        type=_positive_integer,
        default=2,
        help="capture one GIF frame after this many placements",
    )
    evaluate_parser.set_defaults(command_runner=run_evaluate_command)


def _add_rom_argument(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--rom",
        type=Path,
        default=DEFAULT_ROM_PATH,
        help=f"Pandora's Blocks ROM (default: {DEFAULT_ROM_PATH})",
    )


def _positive_integer(raw_value: str) -> int:
    parsed_value = int(raw_value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def _nonnegative_integer(raw_value: str) -> int:
    parsed_value = int(raw_value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed_value


if __name__ == "__main__":
    main()

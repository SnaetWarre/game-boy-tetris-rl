import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gymnasium as gym

from gb_tetris_rl.game.environment import (
    TetrisEnvironment,
    agent_action_space,
    agent_observation_space,
)

POLICY_ARCHITECTURES = ("afterstate", "dueling")
_IMITATION_BATCH_SIZE = 256
_IMITATION_LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 0
    seed: int = 0
    device: str = "auto"
    environment_count: int = 4
    show_window: bool = False
    emulation_speed: int = 0
    policy_architecture: str = "afterstate"
    demonstration_count: int = 100_000
    imitation_epoch_count: int = 10
    demonstration_episode_piece_limit: int = 4_000
    planner_lookahead: bool = True
    dagger_round_count: int = 1


@dataclass(frozen=True)
class TrainingArtifacts:
    imitation_model_path: Path | None
    dqn_model_path: Path | None


class _ContractOnlyEnvironment(gym.Env):
    """Agent spaces without an emulator, for runs that never step a game."""

    def __init__(self) -> None:
        self.observation_space = agent_observation_space()
        self.action_space = agent_action_space()

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("the contract-only environment cannot be reset")

    def step(self, action):
        raise RuntimeError("the contract-only environment cannot be stepped")


def _create_monitored_environment(
    rom_path: str | Path,
    *,
    render_mode: str | None,
    display_emulator_window: bool | None = None,
    emulation_speed: int = 0,
):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(
        TetrisEnvironment(
            rom_path,
            render_mode=render_mode,
            display_emulator_window=display_emulator_window,
            emulation_speed=emulation_speed,
        )
    )


def train_agent(
    rom_path: str | Path,
    run_directory: str | Path,
    config: TrainingConfig,
) -> TrainingArtifacts:
    """Run planner imitation with DAgger, then optional DQN fine-tuning in the emulator."""
    _validate_training_config(config)

    try:
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as import_error:
        raise RuntimeError("training dependencies are missing; run: uv sync") from import_error

    resolved_run_directory = Path(run_directory).expanduser().resolve()
    resolved_run_directory.mkdir(parents=True, exist_ok=True)

    if config.total_timesteps:
        contract_check_environment = TetrisEnvironment(rom_path)
        try:
            check_env(contract_check_environment, warn=True)
        finally:
            contract_check_environment.close()
        training_environment = _create_training_environment(rom_path, config, SubprocVecEnv)
    else:
        # Imitation runs entirely in the bitboard simulator.
        training_environment = _ContractOnlyEnvironment()

    from gb_tetris_rl.agent.smart_dqn import (
        TetrisAfterstatePolicy,
        TetrisDQN,
        TetrisDuelingPolicy,
        TetrisFeatureExtractor,
    )

    if config.policy_architecture == "afterstate":
        policy_class = TetrisAfterstatePolicy
        policy_kwargs = {"net_arch": [256, 256]}
    else:
        policy_class = TetrisDuelingPolicy
        policy_kwargs = {"features_extractor_class": TetrisFeatureExtractor, "net_arch": [256]}

    dqn_agent = TetrisDQN(
        policy_class,
        training_environment,
        learning_rate=1e-4,
        buffer_size=500_000 if config.total_timesteps else 1,
        learning_starts=min(25_000, max(1_000, config.total_timesteps // 20)),
        batch_size=512,
        gamma=0.99,
        train_freq=max(1, 4 // config.environment_count),
        gradient_steps=max(1, config.environment_count // 4),
        target_update_interval=5_000,
        exploration_fraction=0.25,
        exploration_initial_eps=0.15 if config.demonstration_count else 1.0,
        exploration_final_eps=0.02 if config.demonstration_count else 0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config.seed,
        device=config.device,
    )
    imitation_model_path: Path | None = None
    dqn_model_path: Path | None = None
    try:
        if config.demonstration_count:
            imitation_model_path = _run_imitation_stage(
                dqn_agent,
                resolved_run_directory,
                config,
            )

        if config.total_timesteps:
            checkpoint_directory = resolved_run_directory / "dqn-checkpoints"
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=max(50_000 // config.environment_count, 1),
                save_path=str(checkpoint_directory),
                name_prefix="dqn",
            )
            print(f"Starting DQN fine-tuning for {config.total_timesteps:,} emulator steps...")
            dqn_agent.learn(
                total_timesteps=config.total_timesteps,
                callback=checkpoint_callback,
                log_interval=100,
            )
            dqn_model_stem = resolved_run_directory / "dqn-final"
            dqn_agent.save(str(dqn_model_stem))
            dqn_model_path = dqn_model_stem.with_suffix(".zip")
    finally:
        training_environment.close()

    return TrainingArtifacts(
        imitation_model_path=imitation_model_path,
        dqn_model_path=dqn_model_path,
    )


def _create_training_environment(rom_path, config: TrainingConfig, subprocess_vector_class):
    if config.environment_count == 1:
        return _create_monitored_environment(
            rom_path,
            render_mode="human" if config.show_window else None,
            emulation_speed=config.emulation_speed,
        )

    shared_render_mode = "human" if config.show_window else None
    environment_factories = [
        partial(
            _create_monitored_environment,
            rom_path,
            render_mode=shared_render_mode,
            display_emulator_window=config.show_window and worker_index == 0,
            emulation_speed=config.emulation_speed if worker_index == 0 else 0,
        )
        for worker_index in range(config.environment_count)
    ]
    return subprocess_vector_class(environment_factories)


def _run_imitation_stage(
    dqn_agent,
    run_directory: Path,
    config: TrainingConfig,
):
    from gb_tetris_rl.agent.imitation import (
        generate_planner_demonstrations,
        greedy_policy,
        pretrain_policy_from_demonstrations,
    )

    def collect_demonstrations(seed: int, behaviour_policy=None):
        started_at = time.perf_counter()
        demonstration_dataset = generate_planner_demonstrations(
            config.demonstration_count,
            seed=seed,
            maximum_episode_pieces=config.demonstration_episode_piece_limit,
            use_lookahead=config.planner_lookahead,
            device=dqn_agent.device,
            behaviour_policy=behaviour_policy,
        )
        print(f"Collected in {time.perf_counter() - started_at:.1f}s")
        return demonstration_dataset

    print(f"Generating {config.demonstration_count:,} planner demonstrations...")
    demonstration_dataset = collect_demonstrations(config.seed)
    imitation_metrics = pretrain_policy_from_demonstrations(
        dqn_agent,
        demonstration_dataset,
        epoch_count=config.imitation_epoch_count,
        batch_size=_IMITATION_BATCH_SIZE,
        seed=config.seed,
        learning_rate=_IMITATION_LEARNING_RATE,
    )
    for dagger_round in range(1, config.dagger_round_count + 1):
        print(
            f"DAgger round {dagger_round}/{config.dagger_round_count}: planner labels "
            f"{config.demonstration_count:,} boards the policy reaches by itself..."
        )
        demonstration_dataset = demonstration_dataset.concatenate(
            collect_demonstrations(
                config.seed + dagger_round,
                behaviour_policy=greedy_policy(dqn_agent.q_net),
            )
        )
        imitation_metrics = pretrain_policy_from_demonstrations(
            dqn_agent,
            demonstration_dataset,
            epoch_count=max(1, config.imitation_epoch_count // 2),
            batch_size=_IMITATION_BATCH_SIZE,
            seed=config.seed + dagger_round,
            learning_rate=_IMITATION_LEARNING_RATE / 2,
        )
    print(
        "Imitation pretraining complete: "
        f"validation_accuracy={imitation_metrics.validation_accuracy:.1%}"
    )
    imitation_model_stem = run_directory / "imitation"
    dqn_agent.save(str(imitation_model_stem))
    imitation_model_path = imitation_model_stem.with_suffix(".zip")
    print(f"Saved imitation checkpoint: {imitation_model_path}")
    return imitation_model_path


def _validate_training_config(config: TrainingConfig) -> None:
    if config.total_timesteps < 0:
        raise ValueError("total_timesteps cannot be negative")
    if config.environment_count < 1:
        raise ValueError("environment_count must be at least 1")
    if config.policy_architecture not in POLICY_ARCHITECTURES:
        raise ValueError(f"policy_architecture must be one of {', '.join(POLICY_ARCHITECTURES)}")
    if config.demonstration_count < 0:
        raise ValueError("demonstration_count cannot be negative")
    if config.imitation_epoch_count < 0:
        raise ValueError("imitation_epoch_count cannot be negative")
    if config.demonstration_episode_piece_limit < 1:
        raise ValueError("demonstration_episode_piece_limit must be at least 1")
    if config.dagger_round_count < 0:
        raise ValueError("dagger_round_count cannot be negative")
    if (config.demonstration_count == 0) != (config.imitation_epoch_count == 0):
        raise ValueError(
            "demonstration_count and imitation_epoch_count must both be zero or positive"
        )
    if config.total_timesteps == 0 and config.demonstration_count == 0:
        raise ValueError("at least one training stage must be enabled")

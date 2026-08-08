from functools import partial
from pathlib import Path

from gb_tetris_rl.environment import TetrisEnvironment


def _create_monitored_environment(
    rom_path: str | Path,
    *,
    render_mode: str | None,
    display_emulator_window: bool | None = None,
    emulation_speed: int = 0,
    control_mode: str = "buttons",
):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(
        TetrisEnvironment(
            rom_path,
            render_mode=render_mode,
            display_emulator_window=display_emulator_window,
            emulation_speed=emulation_speed,
            control_mode=control_mode,
        )
    )


def train_dqn_agent(
    rom_path: str | Path,
    output_path: str | Path,
    *,
    total_timesteps: int,
    seed: int,
    device: str,
    show_window: bool = False,
    environment_count: int = 1,
    emulation_speed: int = 0,
    control_mode: str = "buttons",
    expert_sample_count: int = 0,
    expert_epoch_count: int = 0,
) -> Path:
    if environment_count < 1:
        raise ValueError("environment_count must be at least 1")
    if expert_sample_count < 0:
        raise ValueError("expert sample count cannot be negative")
    if expert_epoch_count < 0:
        raise ValueError("expert epoch count cannot be negative")
    if expert_sample_count > 0 and control_mode not in {"placements", "placements-hold"}:
        raise ValueError("expert pretraining requires placement controls")
    if (expert_sample_count == 0) != (expert_epoch_count == 0):
        raise ValueError("expert samples and epochs must either both be zero or both be positive")

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as import_error:
        raise RuntimeError(
            "training dependencies are missing; install with: python -m pip install -e '.[train]'"
        ) from import_error

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = resolved_output_path.parent / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    environment = TetrisEnvironment(rom_path, control_mode=control_mode)
    try:
        check_env(environment, warn=True)
    finally:
        environment.close()

    if environment_count == 1:
        training_render_mode = "human" if show_window else None
        training_environment = _create_monitored_environment(
            rom_path,
            render_mode=training_render_mode,
            emulation_speed=emulation_speed,
            control_mode=control_mode,
        )
    else:
        shared_render_mode = "human" if show_window else None
        environment_factories = [
            partial(
                _create_monitored_environment,
                rom_path,
                render_mode=shared_render_mode,
                display_emulator_window=show_window and worker_index == 0,
                emulation_speed=emulation_speed if worker_index == 0 else 0,
                control_mode=control_mode,
            )
            for worker_index in range(environment_count)
        ]
        training_environment = SubprocVecEnv(environment_factories)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // environment_count, 1),
        save_path=str(checkpoint_directory),
        name_prefix="tetris-dqn",
    )
    replay_warmup_steps = min(25_000, max(1_000, total_timesteps // 20))
    vector_step_training_frequency = max(1, 4 // environment_count)
    gradient_steps_per_update = max(1, environment_count // 4)
    uses_placement_controls = control_mode in {"placements", "placements-hold"}
    network_layers = [512, 512, 256] if uses_placement_controls else [256, 256]
    training_batch_size = 512 if uses_placement_controls else 128
    replay_buffer_size = 500_000 if uses_placement_controls else 100_000
    model = DQN(
        "MlpPolicy",
        training_environment,
        learning_rate=1e-4,
        buffer_size=replay_buffer_size,
        learning_starts=replay_warmup_steps,
        batch_size=training_batch_size,
        gamma=0.99,
        train_freq=vector_step_training_frequency,
        gradient_steps=gradient_steps_per_update,
        target_update_interval=5_000,
        exploration_fraction=0.25,
        exploration_initial_eps=0.15 if expert_sample_count > 0 else 1.0,
        exploration_final_eps=0.02 if expert_sample_count > 0 else 0.05,
        policy_kwargs={"net_arch": network_layers},
        verbose=1,
        seed=seed,
        device=device,
    )
    try:
        if expert_sample_count > 0:
            from gb_tetris_rl.expert_training import (
                generate_expert_dataset,
                pretrain_dqn_policy,
            )

            print(f"Generating {expert_sample_count:,} heuristic expert placements...")
            expert_dataset = generate_expert_dataset(
                expert_sample_count,
                seed=seed,
                use_hold=control_mode == "placements-hold",
            )
            expert_metrics = pretrain_dqn_policy(
                model,
                expert_dataset,
                epoch_count=expert_epoch_count,
                batch_size=2_048,
                seed=seed,
            )
            print(
                "Expert pretraining complete: "
                f"validation_accuracy={expert_metrics.validation_accuracy:.1%}"
            )
            expert_model_path = (
                resolved_output_path.parent / f"{resolved_output_path.stem}-expert"
            )
            model.save(str(expert_model_path))
            print(f"Saved expert checkpoint: {expert_model_path.with_suffix('.zip')}")
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        model.save(str(resolved_output_path))
    finally:
        training_environment.close()

    return resolved_output_path.with_suffix(".zip")

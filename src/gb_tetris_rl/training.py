from functools import partial
from pathlib import Path

from gb_tetris_rl.environment import TetrisEnvironment


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
) -> Path:
    if environment_count < 1:
        raise ValueError("environment_count must be at least 1")

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

    environment = TetrisEnvironment(rom_path)
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
            )
            for worker_index in range(environment_count)
        ]
        training_environment = SubprocVecEnv(environment_factories)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // environment_count, 1),
        save_path=str(checkpoint_directory),
        name_prefix="tetris-dqn",
    )
    replay_warmup_steps = min(10_000, max(100, total_timesteps // 10))
    vector_step_training_frequency = max(1, 4 // environment_count)
    model = DQN(
        "MlpPolicy",
        training_environment,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=replay_warmup_steps,
        batch_size=128,
        gamma=0.99,
        train_freq=vector_step_training_frequency,
        gradient_steps=1,
        target_update_interval=5_000,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=seed,
        device=device,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        model.save(str(resolved_output_path))
    finally:
        training_environment.close()

    return resolved_output_path.with_suffix(".zip")

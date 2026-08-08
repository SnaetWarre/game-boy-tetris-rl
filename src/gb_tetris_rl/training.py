from pathlib import Path

from gb_tetris_rl.environment import TetrisEnvironment


def train_dqn_agent(
    rom_path: str | Path,
    output_path: str | Path,
    *,
    total_timesteps: int,
    seed: int,
    device: str,
) -> Path:
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
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

    monitored_environment = Monitor(TetrisEnvironment(rom_path))
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_directory),
        name_prefix="tetris-dqn",
    )
    model = DQN(
        "MlpPolicy",
        monitored_environment,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
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
        monitored_environment.close()

    return resolved_output_path.with_suffix(".zip")

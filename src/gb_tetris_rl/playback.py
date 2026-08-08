from pathlib import Path

from gb_tetris_rl.environment import TetrisEnvironment


def watch_trained_agent(
    rom_path: str | Path,
    model_path: str | Path,
    *,
    episode_count: int,
    show_window: bool,
    recording_path: str | Path | None,
    capture_every_n_steps: int,
    seed: int,
) -> list[dict[str, int]]:
    if episode_count < 1:
        raise ValueError("episode_count must be at least 1")
    if capture_every_n_steps < 1:
        raise ValueError("capture_every_n_steps must be at least 1")

    try:
        from stable_baselines3 import DQN
    except ImportError as import_error:
        raise RuntimeError(
            "playback dependencies are missing; install with: python -m pip install -e '.[train]'"
        ) from import_error

    render_mode = "human" if show_window else "rgb_array"
    environment = TetrisEnvironment(rom_path, render_mode=render_mode)
    model = DQN.load(str(Path(model_path).expanduser().resolve()))
    recorded_frames = []
    episode_summaries: list[dict[str, int]] = []
    maximum_recorded_frames = 3_000

    try:
        for episode_index in range(episode_count):
            observation, episode_info = environment.reset(seed=seed + episode_index)
            episode_finished = False
            episode_step = 0
            while not episode_finished:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, episode_info = environment.step(int(action))
                episode_finished = terminated or truncated
                episode_step += 1
                should_capture_frame = (
                    recording_path is not None
                    and episode_step % capture_every_n_steps == 0
                    and len(recorded_frames) < maximum_recorded_frames
                )
                if should_capture_frame:
                    frame = environment.render()
                    if frame is not None:
                        recorded_frames.append(frame)
            episode_summaries.append(episode_info)
    finally:
        environment.close()

    if recording_path is not None:
        _write_gif(
            recorded_frames,
            recording_path,
            frame_duration_ms=round((1_000 / 30) * capture_every_n_steps),
        )
    return episode_summaries


def _write_gif(
    frames: list,
    recording_path: str | Path,
    *,
    frame_duration_ms: int,
) -> None:
    if not frames:
        raise RuntimeError("no rendered frames were available for recording")
    try:
        from PIL import Image
    except ImportError as import_error:
        raise RuntimeError(
            "GIF recording requires Pillow; install with: python -m pip install -e '.[record]'"
        ) from import_error

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

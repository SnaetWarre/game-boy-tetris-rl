# Game Boy Tetris RL

A deliberately small reinforcement-learning project that trains a DQN agent to
play the original Game Boy version of Tetris through
[PyBoy](https://github.com/Baekalfen/PyBoy).

The agent sees PyBoy's simplified 18 by 10 board instead of scraping pixels. It
can wait, move left or right, rotate in either direction, and soft-drop. Rewards
combine actual score and cleared lines with small board-quality signals for
holes, height, and bumpiness.

## Legal boundary

This repository does not contain, download, patch, or redistribute a Nintendo
ROM. Supply a Game Boy Tetris ROM dumped from a cartridge you own. ROMs and
emulator save files are ignored by Git.

No decompilation is required. PyBoy already provides the interoperability layer:
controller input, board state, score, lines, level, deterministic reset, and
headless emulation.

## Setup

The project is tested with Python 3.14. Install PyTorch's CPU wheel first so pip
does not pull a separate multi-gigabyte CUDA runtime for this small network:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.13.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train,record,dev]'
```

Keep the ROM outside the repository or place it in the ignored `roms/` folder.
Every command validates the cartridge header before starting PyBoy.

```sh
gb-tetris-rl doctor --rom /path/to/tetris.gb
```

## Try the environment

Run a short random-policy episode first. This verifies input, observations,
rewards, and resets without beginning a long training run.

```sh
gb-tetris-rl random --rom /path/to/tetris.gb --steps 2000
```

Add `--window` to watch the emulator at normal speed.

## Train

```sh
gb-tetris-rl train \
  --rom /path/to/tetris.gb \
  --timesteps 250000 \
  --output models/tetris-dqn
```

Training uses Stable-Baselines3's DQN with a compact two-layer MLP. PyBoy runs
headlessly and without a speed limit. Checkpoints are written beside the final
model.

The first useful milestone is not "perfect Tetris." It is beating the random
policy on mean lines cleared over the same deterministic episode seeds.

## Watch or record the agent

```sh
gb-tetris-rl watch \
  --rom /path/to/tetris.gb \
  --model models/tetris-dqn.zip \
  --episodes 3 \
  --record recordings/tetris-agent.gif
```

Use `--window` for an SDL window. GIF recording is intentionally optional so
headless training does not pay rendering costs.

## Project structure

```text
src/gb_tetris_rl/
  environment.py   Gymnasium/PyBoy boundary
  rewards.py       Pure board measurements and reward shaping
  roms.py          ROM header validation
  training.py      DQN configuration and checkpoints
  playback.py      Evaluation and GIF recording
  cli.py           doctor, random, train, and watch commands
tests/
  test_environment.py
  test_playback.py
  test_rewards.py
  test_roms.py
  test_training.py
```

## Why Tetris

Tetris has short repeatable episodes, a discrete action space, measurable
progress, and a compact observation. It is a much more manageable first emulator
RL target than a long exploration game with sparse rewards.

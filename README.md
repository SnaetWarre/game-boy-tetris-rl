# Game Boy Tetris RL

A deliberately small reinforcement-learning project that trains a CUDA-backed
DQN agent to play a real Game Boy falling-block game through
[PyBoy](https://github.com/Baekalfen/PyBoy).

The agent sees PyBoy's simplified 18 by 10 board instead of scraping pixels. It
can wait, move left or right, rotate in either direction, soft-drop, and
hard-drop. Rewards
combine actual score and cleared lines with small board-quality signals for
holes, height, and bumpiness.

## Legal boundary

The zero-friction path uses
[Pandora's Blocks](https://github.com/Villadelfia/dmgtris), an original Game Boy
homebrew released under GPL-3.0. Its author distributes the ROM and matching
debug symbols, so `bootstrap` can legally download a checksum-pinned build.

Nintendo's Tetris is also supported when you supply a ROM dumped from a
cartridge you own. This repository never downloads or redistributes that ROM.
All ROMs and emulator save files remain ignored by Git.

No screen scraping is required. The Pandora's Blocks adapter reads the named
playfield, score, line-clear, level, and game-state locations published by its
assembly source and `.sym` file.

## Setup

The project is tested with Python 3.14 and PyTorch's CUDA 13 runtime:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.13.0'
python -m pip install -e '.[train,record,dev]'
```

Fetch the open-source homebrew and verify the complete setup:

```sh
gb-tetris-rl bootstrap
gb-tetris-rl doctor --rom roms/PandorasBlocks.gbc
```

## Try the environment

Run a short random-policy episode first. This verifies input, observations,
rewards, and resets without beginning a long training run.

```sh
gb-tetris-rl random --rom roms/PandorasBlocks.gbc --steps 2000
```

Add `--window` to watch the emulator at normal speed.

## Train

```sh
gb-tetris-rl train \
  --rom roms/PandorasBlocks.gbc \
  --timesteps 250000 \
  --output models/tetris-dqn \
  --device cuda \
  --envs 4 \
  --speed 0 \
  --window
```

Training uses Stable-Baselines3's DQN with a compact two-layer MLP. PyBoy runs
headlessly and without a speed limit by default. `--window` instead renders the
actual emulator continuously at normal Game Boy speed while the CUDA-backed
network trains. `--envs 4` runs four independent PyBoy processes, displays the
first one, and batches their experience into the shared CUDA policy. This is
more useful than putting the instruction-dependent Game Boy CPU loop on a GPU.
`--speed 0` keeps drawing the visible worker without applying the normal 60 FPS
frame limiter; use `--speed 1` when you want human-speed playback while training.
Checkpoints are written beside the final model.

The first useful milestone is not "perfect Tetris." It is beating the random
policy on mean lines cleared over the same deterministic episode seeds.

## Watch or record the agent

```sh
gb-tetris-rl watch \
  --rom roms/PandorasBlocks.gbc \
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
  game_adapters.py source-backed state readers for both supported games
  homebrew.py      pinned GPL ROM and symbol downloader
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

Falling-block games have short repeatable episodes, a discrete action space,
measurable progress, and a compact observation. It is a much more manageable
first emulator RL target than a long exploration game with sparse rewards.

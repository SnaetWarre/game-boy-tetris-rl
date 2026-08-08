# Game Boy Tetris RL

A deliberately small reinforcement-learning project that trains a CUDA-backed
DQN agent to play a real Game Boy falling-block game through
[PyBoy](https://github.com/Baekalfen/PyBoy).

The agent sees the settled 18 by 10 board instead of scraping pixels. Pandora's
Blocks supports both frame-level buttons and a faster placement action: choose
one of four rotations and one of ten columns, then hard-drop. Rewards prioritize
actual cleared lines and score, with smaller signals for holes, height, and
bumpiness.

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
  --controls placements \
  --expert-samples 50000 \
  --expert-epochs 80 \
  --timesteps 50000 \
  --output models/tetris-dqn \
  --device cuda \
  --envs 8
```

Training first distills good placements from a deterministic board planner into
the CUDA network, then fine-tunes with Stable-Baselines3 DQN. The pre-RL model is
saved as `models/tetris-dqn-expert.zip`; this matters because RL is only an
improvement if real-ROM evaluation beats that checkpoint. PyBoy runs headlessly
and without a frame limit by default. `--envs 8` feeds the shared CUDA policy
from eight emulator processes. Add `--window --speed 0` to display the first
worker at unlimited speed, or `--speed 1` for human-speed rendering.

The first useful milestone is not "perfect Tetris." It is beating the random
policy on mean lines cleared over the same deterministic episode seeds.

## Watch or record the agent

```sh
gb-tetris-rl watch \
  --rom roms/PandorasBlocks.gbc \
  --model models/tetris-dqn-expert.zip \
  --controls placements \
  --expert-safety \
  --target-lines 100 \
  --episodes 1 \
  --window
```

`--expert-safety` keeps the network's placement when it agrees with the
two-piece afterstate planner and corrects it otherwise. Omit the flag to measure
the neural policy alone. This separation prevents a visually impressive hybrid
run from being reported as pure-network performance. Use `--record` to write a
GIF instead of opening an SDL window.

Hold-aware placement uses a separate observation/action contract so older
models remain loadable: 202 inputs include the held-piece one-hot state, and 80
actions represent 40 direct placements plus 40 hold-then-place decisions.

```sh
gb-tetris-rl heuristic \
  --rom roms/PandorasBlocks.gbc \
  --hold \
  --forever \
  --window \
  --speed 1
```

`--forever` resets to another deterministic seed after a top-out or the 20,000
piece safety limit. It intentionally has no line target. `--speed 1` preserves
the Game Boy clock, so the ROM's level-dependent gravity acceleration remains
visible; use `--speed 0` for unlimited emulation speed.

To validate the emulator controls and counters without any neural model:

```sh
gb-tetris-rl heuristic \
  --rom roms/PandorasBlocks.gbc \
  --target-lines 40 \
  --seed 0
```

## Project structure

```text
src/gb_tetris_rl/
  environment.py   Gymnasium/PyBoy boundary
  game_adapters.py source-backed state readers for both supported games
  homebrew.py      pinned GPL ROM and symbol downloader
  heuristic.py     fast afterstate simulation and two-piece planner
  expert_training.py CUDA policy distillation dataset and optimizer
  rewards.py       Pure board measurements and reward shaping
  roms.py          ROM header validation
  training.py      DQN configuration and checkpoints
  playback.py      Evaluation and GIF recording
  cli.py           doctor, random, heuristic, train, and watch commands
tests/
  test_expert_training.py
  test_environment.py
  test_game_adapters.py
  test_heuristic.py
  test_playback.py
  test_rewards.py
  test_roms.py
  test_training.py
```

## Why Tetris

Falling-block games have short repeatable episodes, a discrete action space,
measurable progress, and a compact observation. It is a much more manageable
first emulator RL target than a long exploration game with sparse rewards.

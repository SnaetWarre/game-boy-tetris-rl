# Game Boy Tetris RL

[![CI](https://github.com/SnaetWarre/game-boy-tetris-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/SnaetWarre/game-boy-tetris-rl/actions/workflows/ci.yml)

A small reinforcement-learning project that trains one hold-aware DQN policy
to play [Pandora's Blocks](https://github.com/Villadelfia/dmgtris), an
open-source Game Boy falling-block game, through
[PyBoy](https://github.com/Baekalfen/PyBoy).

![Hold-aware agent playing Pandora's Blocks through PyBoy](docs/demo.gif)

This preview is a real 10-line PyBoy run of the learned policy with planner
safety: 27 pieces, 10 hold actions, and 12 reported planner interventions.

The policy does not scrape pixels. It receives a 202-value observation:

- 180 settled board cells
- 7 values for the current piece
- 7 values for the next piece
- 8 values for the hold slot, including empty

It chooses one of 80 actions: four rotations by ten columns, either directly or
after using hold.

## Run the demo

Install the locked environment and fetch the checksum-pinned GPL ROM:

```sh
uv sync --locked
uv run gb-tetris-rl bootstrap
uv run gb-tetris-rl doctor
```

Then run the canonical local checkpoint:

```sh
uv run gb-tetris-rl demo
```

The demo opens PyBoy, aims for 40 lines, and uses the learned policy with a
deterministic planner as a safety layer. It prints the intervention count so a
stable hybrid run cannot be mistaken for pure neural performance.

To see what the neural policy can do by itself:

```sh
uv run gb-tetris-rl demo --neural-only
```

`models/demo-agent.zip` is a generated local artifact and is intentionally not
committed. Train a replacement if it is missing.

## Where the agent learns

There is one neural architecture and two consecutive training stages:

```text
deterministic planner
        |
        | labels simulated board states
        v
imitation pretraining                 src/gb_tetris_rl/agent/imitation.py
        |
        | initializes the DQN policy
        v
PyBoy environment -> rewards -> DQN   src/gb_tetris_rl/agent/training.py
                              |
                              v
                         model artifacts
```

1. `generate_planner_demonstrations` simulates legal placements and records the
   planner's chosen action for each board.
2. `pretrain_policy_from_demonstrations` updates the policy network with
   cross-entropy loss so it learns to copy those choices.
3. `train_agent` calls `dqn_agent.learn(...)`. This is the reinforcement-learning
   stage: Stable-Baselines3 collects transitions from real PyBoy workers and
   optimizes the policy from the shaped reward in `game/rewards.py`.

The planner in `agent/planner.py` never learns. It is a deterministic baseline,
a source of imitation labels, and an optional demo safety layer.

## Train

```sh
uv run gb-tetris-rl train \
  --run-dir models/runs/hold-aware-v1 \
  --demonstrations 50000 \
  --imitation-epochs 80 \
  --timesteps 50000 \
  --device cuda \
  --envs 8
```

Every run has an explicit artifact layout:

```text
models/runs/hold-aware-v1/
  imitation.zip       policy after planner imitation
  dqn-final.zip       policy after emulator reinforcement learning
  dqn-checkpoints/    intermediate recovery checkpoints
```

These are checkpoints of the same policy architecture, not different model
implementations. The project root keeps only `models/demo-agent.zip` as the
canonical demo checkpoint. Historical local experiments live in
`models/archive/` and are not part of the runtime path.

Evaluate checkpoints on fixed seeds before promoting one to the demo:

```sh
# Honest neural-only result
uv run gb-tetris-rl evaluate \
  --model models/runs/hold-aware-v1/dqn-final.zip \
  --episodes 5

# Hybrid result, reported separately
uv run gb-tetris-rl evaluate \
  --model models/runs/hold-aware-v1/dqn-final.zip \
  --episodes 5 \
  --planner-safety \
  --target-lines 40
```

Line counts on deterministic, unseen seeds are the acceptance metric. Training
reward alone is not enough evidence that the policy learned useful play.

## Project map

```text
src/gb_tetris_rl/
  agent/
    planner.py       pure board simulator and deterministic action planner
    imitation.py     demonstration generation and imitation optimization
    training.py      DQN construction, PyBoy workers, learning, checkpoints
    evaluation.py    neural-only and planner-guarded evaluation, GIF output
  game/
    contracts.py     the single 202-input/80-action agent contract
    environment.py   Gymnasium environment and reward transitions
    pandoras_blocks.py  source-backed memory reader and emulator controls
    rewards.py       pure board measurements and reward shaping
    rom.py           cartridge title and checksum validation
  commands.py        command workflows
  cli.py             argument parsing only
  homebrew.py        pinned ROM and symbol download
tests/                mirrors the modules above
models/               generated artifacts, documented separately
```

The runtime loop is deliberately short:

```text
Pandora's Blocks ROM
        v
PandorasBlocksAdapter -> TetrisEnvironment -> 202-value observation
        ^                                         |
        |                                         v
        +---------- one of 80 actions <- DQN policy
```

## Other useful commands

Run the deterministic planner without loading a neural model:

```sh
uv run gb-tetris-rl planner --target-lines 40 --window
```

Record an evaluated policy as a GIF:

```sh
uv run gb-tetris-rl evaluate \
  --model models/demo-agent.zip \
  --episodes 1 \
  --record recordings/demo.gif
```

## Legal boundary

`bootstrap` downloads a source-matched Pandora's Blocks ROM and symbol file from
the upstream GPL-3.0 project and verifies both checksums. ROMs, symbols, emulator
state, recordings, and model weights stay ignored by Git.

This project no longer carries a second Nintendo Tetris integration. Keeping one
legally reproducible game and one agent contract makes the training and demo path
easy to audit.

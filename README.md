# Game Boy Tetris RL

[![CI](https://github.com/SnaetWarre/game-boy-tetris-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/SnaetWarre/game-boy-tetris-rl/actions/workflows/ci.yml)

A small reinforcement-learning project that trains one hold-aware spatial policy
to play [Pandora's Blocks](https://github.com/Villadelfia/dmgtris), an
open-source Game Boy falling-block game, through
[PyBoy](https://github.com/Baekalfen/PyBoy).

![Hold-aware agent playing Pandora's Blocks through PyBoy](docs/demo.gif)

This preview is a real 10-line PyBoy hybrid run with an explicit planner
override: 27 pieces, 10 hold actions, and 12 reported planner disagreements.

The policy does not scrape pixels. It receives a 202-value observation:

- 180 settled board cells
- 7 values for the current piece
- 7 values for the next piece
- 8 values for the hold slot, including empty

It chooses one of 80 actions: four rotations by ten columns, either directly or
after using hold.

These are atomic placement-control actions, not human-speed button sequences.
The adapter returns the active piece to its legal spawn row and pauses gravity
only while applying the requested rotation and horizontal movement. It restores
the ROM's current gravity before the hard drop. This keeps the Gymnasium action
contract exact as Pandora's Blocks accelerates, but placement-control results
are not human-equivalent records.

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

The demo opens PyBoy, aims for 40 lines, and explicitly replaces every neural
action that differs from the deterministic planner. It prints disagreement and
rescue counts so a stable hybrid run cannot be mistaken for pure neural
performance.

Add `--fast` to keep the emulator window visible but remove its frame limiter:

```sh
uv run gb-tetris-rl demo --fast
```

To see what the neural policy can do by itself:

```sh
uv run gb-tetris-rl demo --neural-only
```

`models/demo-agent.zip` is a generated local artifact and is intentionally not
committed. Train a replacement if it is missing.

## Where the agent learns

The v0.4 policy scores afterstates, the boards each action leaves behind:

- a fixed bitboard simulator applies the game rules to all 80 actions and drops
  any placement that does not fit from above the field
- a convolutional encoder with full-height column filters reads each resulting
  18 by 10 board
- a value head combines those features with the cleared line count, the next
  piece still to come, and the hold slot after the action

The simulator never learns and never chooses: it only says where a piece would
land. The network learns how good each board is, and its highest-scoring board
is the action. Placements that do not fit score negative infinity, so the
policy cannot pick one.

Training is planner imitation followed by DAgger:

```text
deterministic planner (next-piece lookahead)
        |
        | labels boards from parallel simulated games
        v
imitation pretraining                 src/gb_tetris_rl/agent/imitation.py
        |
        | the policy plays simulated games by itself;
        | the planner labels every board it reaches (DAgger)
        v
  imitation.zip                       accepted checkpoint
```

1. `generate_planner_demonstrations` plays hundreds of simulated games in
   parallel on the training device and records the planner's action for every
   board.
2. `pretrain_policy_from_demonstrations` trains the network with cross-entropy
   loss so the planner's action scores highest.
3. Each DAgger round lets the trained policy play, labels the boards it reaches
   (including its own mistakes) with the planner, and retrains on the combined
   data.

The v0.3 direct-action dueling network is still available with
`--policy dueling`, and older checkpoints still load.

Emulator DQN fine-tuning remains available through `--timesteps`, but it is off
by default. In the v0.3 acceptance run, 50,000 extra DQN steps reduced the
20-seed mean from 7.60 to 2.90 lines, so that checkpoint was rejected rather
than promoted.

The planner in `agent/planner.py` never learns. It is a deterministic baseline,
a source of imitation labels, and an optional demo guard or override. A
placement that leaves the known next piece nowhere to go now counts as a
top-out in its lookahead instead of scoring as neutral.

## Train

```sh
uv run gb-tetris-rl train --run-dir models/runs/afterstate-v040 --device cuda
```

The defaults generate 100,000 lookahead demonstrations from trajectories of up
to 4,000 pieces, train for 10 epochs, then run one DAgger round of another
100,000 boards. Imitation-only runs never start PyBoy. On an RTX 3060 laptop
GPU, generating demonstrations takes a few seconds and the whole run takes a few
minutes.

Every run has an explicit artifact layout:

```text
models/runs/afterstate-v040/
  imitation.zip       policy after planner imitation and DAgger
  dqn-final.zip       only created when --timesteps is greater than zero
  dqn-checkpoints/    optional reinforcement-learning recovery checkpoints
```

To reproduce the v0.3 network instead:

```sh
uv run gb-tetris-rl train --policy dueling --demonstrations 50000 \
  --imitation-epochs 80 --demonstration-episode-pieces 200 \
  --no-planner-lookahead --dagger-rounds 0
```

The project root keeps only `models/demo-agent.zip` as the canonical local demo
checkpoint. Historical local experiments live in `models/archive/` and are not
part of the runtime path.

### Next neural phase

The experiment command trains a candidate with the same options as `train`,
then evaluates it and the current neural checkpoint on the same 50 seeds:

```sh
uv run gb-tetris-rl next-phase --device cuda
```

Evaluation episodes stop at 500 lines (`--evaluation-target-lines`), because a
strong policy would otherwise play until the 20,000-piece episode limit. The
run directory contains the candidate model, separate incumbent and candidate
evaluation reports, and `comparison.json` with model checksums and the
promotion decision.

The command never changes `models/demo-agent.zip` by default. Add `--promote`
when you want it to replace the incumbent, and it will still do so only when the
candidate has a higher mean without a lower median on the fixed-seed neural-only
evaluation:

```sh
uv run gb-tetris-rl next-phase --device cuda --promote
```

DQN remains opt-in with `--timesteps` because the last validated fine-tuning
run regressed.

## Verified result

### Long-running placement control

The former controller started missing requested columns as gravity increased.
On seeds 10000 through 10004 it topped out at 239, 253, 255, 257, and 240
cleared lines. After making placement actions atomic, all five seeds reached the
500-line evaluation cap with zero final holes. Seed 10000 also reached 1,000
lines with zero holes in 3,014 pieces.

```sh
uv run gb-tetris-rl planner \
  --target-lines 1000 \
  --maximum-pieces 10000 \
  --seed 10000
```

Both the planner and the v0.4 neural policy currently stop near level 3000
(about 7,500 pieces and 2,400 to 2,540 lines on seeds 10000 through 10002). On
seed 10001 the planner's board is clean at level 3001 and tops out within ten
pieces, so the ROM changes something there that the placement adapter does not
handle yet. Treat roughly 2,400 lines as the current ceiling for any agent.

Those are deterministic planner results used to validate the emulator action
contract. They are a stronger source of imitation labels, not neural-only
performance. The raw comparison is committed in
`docs/benchmarks/placement-control-v0.4.0.json`.

### Neural-only policy

Neural-only evaluation on the same 50 deterministic seeds, 10000 through 10049:

| Checkpoint | Mean lines | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Previous local agent | 2.18 | 2 | 1 | 5 |
| v0.3 spatial imitation | 6.70 | 6 | 1 | 17 |
| v0.4 afterstate imitation + DAgger | 500.06 | 500 | 500 | 501 |

The v0.4 run stopped every episode at a 500-line cap, so 500 is a floor, not
the policy's limit: all 50 seeds reached it without planner help. The v0.3
checkpoint scored 6.62 in the same run, because evaluation now loads models on
the CPU and a few near-tied action values resolve differently than on CUDA.
Raw per-episode counts and model checksums are committed in
`docs/benchmarks/v0.3.0.json` and `docs/benchmarks/afterstate-v0.4.0.json`.
This is the strongest neural checkpoint tested in this repository, not a
general Tetris record.

Evaluate checkpoints on fixed seeds before promoting one to the demo:

```sh
# Honest neural-only result
uv run gb-tetris-rl evaluate \
  --model models/runs/afterstate-v040/imitation.zip \
  --episodes 50 \
  --seed 10000 \
  --target-lines 500 \
  --report models/runs/afterstate-v040/neural-50-eval.json

# Hybrid result, reported separately
uv run gb-tetris-rl evaluate \
  --model models/runs/afterstate-v040/imitation.zip \
  --episodes 5 \
  --planner-safety \
  --target-lines 40
```

`--planner-safety` only replaces structurally invalid actions or a placement
that makes the known next piece impossible. Use `--planner-override` when every
planner disagreement should be replaced. Reports identify the mode and keep
disagreements separate from actual safety rescues.

Line counts on deterministic, unseen seeds are the acceptance metric. Training
reward alone is not enough evidence that the policy learned useful play.

## Project map

```text
src/gb_tetris_rl/
  agent/
    planner.py       batched bitboard simulator and deterministic action planner
    imitation.py     parallel simulated games, demonstrations, DAgger, imitation
    action_masks.py  unique rotations and width-aware valid action masks
    smart_dqn.py     afterstate policy and the legacy dueling policy
    training.py      policy construction, training stages, checkpoints
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

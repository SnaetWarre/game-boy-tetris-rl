# Model artifacts

`demo-agent.zip` is the one local checkpoint used by `gb-tetris-rl demo`. It is
an 80-action hold-aware policy trained from planner demonstrations. Since v0.4
it scores the afterstate each action leaves behind; the v0.3 dueling
checkpoint is kept as `archive/demo-agent-v0.3.0.zip`.

New training runs go in `runs/<run-name>/` and contain clearly named stages:

```text
imitation.zip       policy after planner imitation and DAgger
dqn-final.zip       optional policy after --timesteps DQN fine-tuning
dqn-checkpoints/    optional reinforcement-learning recovery checkpoints
```

The accepted checkpoint is `imitation.zip`. The tested v0.3 DQN continuation
scored lower, so fine-tuning is disabled by default and its model was not
promoted.

Model binaries are ignored by Git because they are generated artifacts. Old
local experiments belong in `archive/`, outside the demo path.

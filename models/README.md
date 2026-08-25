# Model artifacts

`demo-agent.zip` is the one local checkpoint used by `gb-tetris-rl demo`. It is
an 80-action hold-aware spatial dueling policy trained from planner
demonstrations.

New training runs go in `runs/<run-name>/` and contain clearly named stages:

```text
imitation.zip       policy after copying planner demonstrations
dqn-final.zip       optional policy after --timesteps DQN fine-tuning
dqn-checkpoints/    optional reinforcement-learning recovery checkpoints
```

The accepted v0.3 checkpoint is `imitation.zip`. The tested DQN continuation
scored lower, so fine-tuning is disabled by default and its model was not
promoted.

Model binaries are ignored by Git because they are generated artifacts. Old
local experiments belong in `archive/`, outside the demo path.

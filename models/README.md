# Model artifacts

`demo-agent.zip` is the one local checkpoint used by `gb-tetris-rl demo`. It is
an 80-action hold-aware DQN policy pretrained from planner demonstrations.

New training runs go in `runs/<run-name>/` and contain clearly named stages:

```text
imitation.zip       policy after copying planner demonstrations
dqn-final.zip       same policy after reinforcement-learning fine-tuning
dqn-checkpoints/    intermediate recovery checkpoints
```

Model binaries are ignored by Git because they are generated artifacts. Old
local experiments belong in `archive/`, outside the demo path.

from enum import IntEnum


class TetrisAction(IntEnum):
    """Discrete actions exposed to the reinforcement-learning agent."""

    WAIT = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    ROTATE_CLOCKWISE = 3
    ROTATE_COUNTERCLOCKWISE = 4
    SOFT_DROP = 5


PYBOY_BUTTON_BY_ACTION: dict[TetrisAction, str | None] = {
    TetrisAction.WAIT: None,
    TetrisAction.MOVE_LEFT: "left",
    TetrisAction.MOVE_RIGHT: "right",
    TetrisAction.ROTATE_CLOCKWISE: "a",
    TetrisAction.ROTATE_COUNTERCLOCKWISE: "b",
    TetrisAction.SOFT_DROP: "down",
}

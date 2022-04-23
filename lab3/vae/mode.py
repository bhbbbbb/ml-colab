from enum import Enum, auto

class Mode(Enum):

    TRAIN = auto()
    EVAL = auto()
    INFERENCE = auto()
    
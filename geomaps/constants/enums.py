from enum import Enum, unique


@unique
class TaskType(Enum):
    TRAIN = "train"
    TEST = "test"
    PRED = "pred"

    def __str__(self) -> str:
        return self.value


@unique
class WriteIntervals(Enum):
    """
        Enumeration based on the class WriteInterval(LightningEnum) for the built-in callback prediction writer.
    """
    BATCH = "batch"
    EPOCH = "epoch"
    BATCH_AND_EPOCH = "batch_and_epoch"

    def __str__(self) -> str:
        return self.value

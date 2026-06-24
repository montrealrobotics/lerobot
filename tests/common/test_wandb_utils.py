from lerobot.common.wandb_utils import WandBLogger


class FakeWandB:
    def __init__(self):
        self.logged = []
        self.defined_metrics = []

    def define_metric(self, key, hidden=False):
        self.defined_metrics.append((key, hidden))

    def log(self, data, step=None):
        self.logged.append((data, step))


def make_logger():
    logger = WandBLogger.__new__(WandBLogger)
    logger._wandb = FakeWandB()
    logger._wandb_custom_step_key = None
    return logger


def test_log_dict_expands_numeric_lists():
    logger = make_logger()

    logger.log_dict({"loss": 1.0, "loss_per_dim": [0.1, 0.2, 0.3]}, step=42)

    assert logger._wandb.logged == [
        ({"train/loss": 1.0}, 42),
        ({"train/loss_per_dim/0": 0.1, "train/loss_per_dim/1": 0.2, "train/loss_per_dim/2": 0.3}, 42),
    ]


def test_log_dict_expands_numeric_lists_with_custom_step():
    logger = make_logger()

    logger.log_dict({"frames": 12, "loss_per_dim": [0.1, 0.2]}, mode="eval", custom_step_key="frames")

    assert logger._wandb.logged == [
        ({"eval/frames": 12}, None),
        ({"eval/loss_per_dim/0": 0.1, "eval/loss_per_dim/1": 0.2, "eval/frames": 12}, None),
    ]

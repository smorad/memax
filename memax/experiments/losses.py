"""Loss function resolution for experiment scripts."""

from typing import Callable, Optional

from memax.equinox.train_utils import loss_classify_terminal_output as eqx_loss_classify
from memax.equinox.train_utils import loss_regress_terminal_output as eqx_loss_regress
from memax.experiments.datasets import get_default_loss_name


def resolve_loss_name(dataset_name: str, loss_function: Optional[str]) -> str:
    if loss_function is not None:
        return loss_function
    return get_default_loss_name(dataset_name)


def get_equinox_loss(loss_name: str) -> Callable:
    if loss_name == "loss_classify_terminal_output":
        return eqx_loss_classify
    if loss_name == "loss_regress_terminal_output":
        return eqx_loss_regress
    raise ValueError(f"Unknown equinox loss: {loss_name}")

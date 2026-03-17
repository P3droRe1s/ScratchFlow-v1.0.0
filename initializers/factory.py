from typing import Callable

from exceptions.kernel_not_found import KernelNotFound
from exceptions.loss_function_not_found import (
    LossFunctionNotFound
)
from exceptions.optimizer_not_found import OptimizerNotFound


from initializers.registry import INITIALIZERS_KERNEL
from initializers.registry import LOSSES
from initializers.registry import OPTIMIZERS


def get_kernel(kernel: str, /) -> Callable:
    if kernel not in INITIALIZERS_KERNEL:
        raise KernelNotFound(f'unknown initializer: {kernel}')

    return INITIALIZERS_KERNEL[kernel]


def get_loss(loss: str, /) -> Callable:
    if loss not in LOSSES:
        raise LossFunctionNotFound(f'unknown loss function: {loss}')

    return LOSSES[loss]


def get_optimizer(optimizer: str, /) -> Callable:
    if optimizer not in OPTIMIZERS:
        raise OptimizerNotFound(f'unknown optimizer: {optimizer}')

    return OPTIMIZERS[optimizer]

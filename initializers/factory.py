from typing import Callable

from exceptions.kernel_not_found import KernelNotFound

from initializers.registry import INITIALIZERS_KERNEL
from initializers.registry import LOSSES
from initializers.registry import OPTIMIZERS


def get_kernel(kernel: str, /) -> Callable:
    if kernel not in INITIALIZERS_KERNEL.keys():
        raise KernelNotFound(f'unknown initializer: {kernel}')

    return INITIALIZERS_KERNEL[kernel]


def get_loss(loss: str, /) -> Callable:
    if loss not in LOSSES.keys():
        raise KernelNotFound(f'unknown loss function: {loss}')

    return LOSSES[loss]


def get_optimizer(optimizer: str, /) -> Callable:
    if optimizer not in OPTIMIZERS.keys():
        raise KernelNotFound(f'unknown optimizer: {optimizer}')

    return OPTIMIZERS[optimizer]

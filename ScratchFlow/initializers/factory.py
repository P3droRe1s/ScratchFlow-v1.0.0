from typing import Callable

from exceptions import (
    KernelNotFound,
    ActivationNotFound,
    DerivativeNotFound
)

from initializers.registry import (
    KERNEL_INITIALIZERS,
    ACTIVATIONS,
    DERIVATIVES
)


def get_kernel(kernel: str) -> Callable:
    if kernel not in KERNEL_INITIALIZERS.keys():
        raise KernelNotFound(f'unknown initializer: {kernel}')

    return KERNEL_INITIALIZERS[kernel]


def get_activation(activation: str) -> Callable:
    if activation not in ACTIVATIONS.keys():
        ERROR = f'unknown activation function: {activation}'

        raise ActivationNotFound(ERROR)

    return ACTIVATIONS[activation]


def get_derivative(derivative: str) -> Callable:
    if derivative not in DERIVATIVES.keys():
        ERROR = f'unknown derivative: {derivative}'

        raise DerivativeNotFound(ERROR)

    return DERIVATIVES[derivative]

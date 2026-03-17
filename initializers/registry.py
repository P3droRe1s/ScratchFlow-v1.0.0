from typing import Callable, Any

from kernel.initializers.uniform_random_distribution import (
    uniform_random_distribution
)

from kernel.losses.absolute_error import absolute_error
from kernel.losses.square_error import square_error

from kernel.optimizers.linear_trick import linear_trick
from kernel.optimizers.square_trick import square_trick

INITIALIZERS_KERNEL: dict[str, Callable[..., Any]] = {
    'uniform_random_distribution': uniform_random_distribution
}

LOSSES: dict[str, Callable[..., Any]] = {
    'absolute_error': absolute_error,
    'square_error': square_error
}

OPTIMIZERS: dict[str, Callable[..., Any]] = {
    'linear_trick': linear_trick,
    'square_trick': square_trick
}

from kernel.initializers.uniform_random_distribution import (
    uniform_random_distribution
)

from kernel.losses.absolute_error import absolute_error

from kernel.optimizers.linear_trick import linear_trick

INITIALIZERS_KERNEL = {
    'uniform_random_distribution': uniform_random_distribution
}

LOSSES = {
    'absolute_error': absolute_error
}

OPTIMIZERS = {
    'linear_trick': linear_trick
}

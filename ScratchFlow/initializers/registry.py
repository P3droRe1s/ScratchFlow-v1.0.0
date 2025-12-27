from kernel.initializers.uniform_random_distribution import (
    uniform_random_distribution
)

from kernel.activations_functions.activations.linear import tensor_linear
from kernel.activations_functions.activations.sigmoid import tensor_sigmoid
from kernel.activations_functions.activations.silu import tensor_silu

from kernel.activations_functions.derivatives.derivative_linear import derivative_linear
from kernel.activations_functions.derivatives.derivative_sigmoid import derivative_sigmoid
from kernel.activations_functions.derivatives.derivative_silu import derivative_silu

from kernel.losses.loss.cross_entropy import cross_entropy

from kernel.losses.derivatives.derivative_cross_entropy import derivative_cross_entropy

KERNEL_INITIALIZERS = {
    'uniform_random_distribution': uniform_random_distribution
}

ACTIVATIONS = {
    'linear': tensor_linear,
    'sigmoid': tensor_sigmoid,
    'silu': tensor_silu
}

LOSSES = {
    'cross_entropy': cross_entropy
}

DERIVATIVES_FUNCTIONS = {
    'linear': derivative_linear,
    'sigmoid': derivative_sigmoid,
    'silu': derivative_silu
}

DERIVATIVES_LOSSES = {
    'cross_entropy': derivative_cross_entropy
}

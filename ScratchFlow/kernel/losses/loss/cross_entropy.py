from math import log

from type.tensor import Tensor


def cross_entropy(y: Tensor, y_hat: Tensor) -> float:
    epsilon = 1e-15

    return sum(
        -value_y * log(min(max(value_y_hat, epsilon), 1 - epsilon))
        for value_y, value_y_hat in zip(y[0], y_hat[0])
    )

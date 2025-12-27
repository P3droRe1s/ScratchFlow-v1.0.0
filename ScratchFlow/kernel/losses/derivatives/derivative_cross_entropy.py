from type.tensor import Tensor


def derivative_cross_entropy(y: Tensor, y_hat: Tensor) -> Tensor:
    derivatives = [
        value_y / value_y_hat
        for value_y, value_y_hat in zip(y[0], y_hat[0])
    ]

    return Tensor(derivatives)

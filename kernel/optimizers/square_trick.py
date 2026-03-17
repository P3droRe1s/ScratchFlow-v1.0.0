def square_trick(
    weights: list[float],
    biases: float,
    y_hat: float,
    y: float,
    x: list[float],
    lr: float = 0.01
) -> tuple[list[float], float]:
    error = y - y_hat

    weights = [w + lr*error*x_ for w, x_ in zip(weights, x)]

    if biases is not None:
        biases += lr*error

    return weights, biases

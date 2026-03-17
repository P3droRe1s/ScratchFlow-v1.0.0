def linear_trick(
    weights: list[float],
    biases: float,
    y_hat: float,
    y: float,
    lr: float = 0.001
) -> tuple[list[float], float]:
    if y > y_hat and y > 0:
        weights = [w + lr for w in weights]

        if biases is not None:
            biases += lr
    elif y > y_hat and y < 0:
        weights = [w - lr for w in weights]

        if biases is not None:
            biases += lr
    elif y < y_hat and y > 0:
        weights = [w - lr for w in weights]

        if biases is not None:
            biases -= lr
    elif y < y_hat and y < 0:
        weights = [w + lr for w in weights]

        if biases is not None:
            biases -= lr

    return weights, biases

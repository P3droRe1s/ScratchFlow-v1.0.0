def linear_trick(
    weights: list[float],
    biases: float,
    y_hat: float,
    y: float,
    eta: float = 0.001
) -> tuple[list[float], float]:
    if y > y_hat and y > 0:
        weights = [w + eta for w in weights]
        biases += eta if bool(biases) else 0
    elif y > y_hat and y < 0:
        weights = [w - eta for w in weights]
        biases += eta if bool(biases) else 0
    elif y < y_hat and y > 0:
        weights = [w - eta for w in weights]
        biases -= eta if bool(biases) else 0
    elif y < y_hat and y < 0:
        weights = [w + eta for w in weights]
        biases -= eta if bool(biases) else 0

    return weights, biases

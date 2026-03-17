from random import uniform


def uniform_random_distribution(a: int) -> list[float]:
    weights = [
        uniform(-1.0, 1.0)
        for _ in range(a)
    ]

    return weights

from random import uniform

from type.tensor import Tensor


def uniform_random_distribution(a: int, b: int) -> Tensor:
    weight_matrix = [[
            uniform(-1.0, 1.0) for _ in range(b)
        ]
        for _ in range(a)
    ]

    return Tensor(weight_matrix)

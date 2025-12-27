from math import exp

from type.tensor import Tensor


def scalar_sigmoid(x: float, /) -> float:
    return 1 / (1 + exp(-x))


def derivative_sigmoid(x: Tensor, /) -> Tensor:
    output = [[
            scalar_sigmoid(value_x)*(1 - scalar_sigmoid(value_x))
            for value_x in vector_x
        ]
        for vector_x in x
    ]

    return Tensor(output)

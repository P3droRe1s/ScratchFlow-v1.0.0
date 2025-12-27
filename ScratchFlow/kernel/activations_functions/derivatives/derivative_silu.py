from type.tensor import Tensor
from kernel.activations.sigmoid import scalar_sigmoid


def scalar_silu(x: float) -> float:
    return x*scalar_sigmoid(x)


def derivative_silu(x: Tensor, /) -> Tensor:
    output = [[
            value_x*scalar_sigmoid(value_x)
            for value_x in vector_x
        ]
        for vector_x in x
    ]

    return Tensor(output)

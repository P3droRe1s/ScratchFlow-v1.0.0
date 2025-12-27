from type.tensor import Tensor


def derivative_linear(x: Tensor) -> Tensor:
    return Tensor([[1.0]*len(x)])

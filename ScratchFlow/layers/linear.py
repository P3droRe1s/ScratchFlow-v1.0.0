from initializers.factory import (
    get_kernel,
    get_activation,
    get_derivative
)
from type.tensor import Tensor


class Linear:
    def __init__(
        self,
        units: int,
        use_bias: bool = True,
        activation: str = 'linear',
        kernel_initializer: str = 'uniform_random_distribution'
    ) -> None:
        self._units = units
        self._use_bias = use_bias
        self._activation = activation
        self._kernel_initializer = kernel_initializer

        self._weights = None
        self._biases = None
        self._params = None

        self._built = False

    def get_weights(self) -> None | Tensor:
        return self._weights
    
    def get_biases(self) -> None | Tensor:
        return self._biases

    def count_params(self) -> None | int:
        return self._params

    def build(self, input_columns: int) -> None:
        kernel = get_kernel(self._kernel_initializer)

        self._weights = kernel(input_columns, self._units)
        if self._use_bias:
            self._biases = Tensor([[0.0]*self._units])

        self._params = input_columns*self._units + self._units

        self._built = True

    def __forward(self, x: Tensor, /) -> Tensor:
        activation = get_activation(self._activation)

        output = x @ self._weights
        output = [[
            value + bias
            for value, bias in zip(row_a, row_b)
        ]
            for row_a, row_b in zip(output, self._biases)
        ] # Brodcasting manual, estou com preguiça de adaptar a classe Tensor :p
          # Futuramente faço isto...
        
        output = activation(output)

        return output

    def __backward(
        self,
    ):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        if not self._built:
            self.build(x.columns)
        
        output = self.__forward(x)

        return output


linear = Linear(
    units=2,
    use_bias=True,
    activation='sigmoid',
    kernel_initializer='uniform_random_distribution'
)

x = Tensor([[2], [2]])
y = linear(x)

print(x)
print(y)

# output shape

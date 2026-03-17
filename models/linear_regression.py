from initializers.factory import get_kernel
from initializers.factory import get_loss
from initializers.factory import get_optimizer


class LinearRegression:
    def __init__(
        self,
        use_bias: bool = True,
        initializer_kernel: str = 'uniform_random_distribution',
        loss_function: str = 'absolute_error',
        optimizer: str = 'square_trick',
        lr: float = 0.01
    ) -> None:
        self._use_bias = use_bias
        self._identifier_optimizer = optimizer
        self._lr = lr

        self._initializer = get_kernel(initializer_kernel)
        self._loss = get_loss(loss_function)
        self._optimizer = get_optimizer(optimizer)

        self.weights: list[float] = []
        self.biases: float = 0.0
        self.params: int = 0

        self._built = False

    def build(self, input_columns: int) -> None:
        self.weights = self._initializer(input_columns)
        self.params = len(self.weights) + (1 if self._use_bias else 0)

        self._built = True

    def __forward(self, x: list[float], /) -> float:
        y_hat = sum(
            (w*x_ for x_, w in zip(self.weights, x))
        )

        if self.biases:
            y_hat += self.biases

        return y_hat

    def backward(self, x: list[float], y: float, y_hat: float) -> float:
        args = [self.weights, self.biases if self._use_bias else None]

        if self._identifier_optimizer == 'linear_trick':
            args.extend([y_hat, y, self._lr])
        elif self._identifier_optimizer == 'square_trick':
            args.extend([y_hat, y, x, self._lr])

        loss = self._loss(y_hat=y_hat, y=y)

        weights, biases = self._optimizer(*args)

        self.weights = weights
        self.biases = biases

        return loss

    def __call__(self, x: list[float]) -> float:
        if not self._built:
            self.build(len(x))

        output = self.__forward(x)

        return output

from collections.abc import Iterator

Vector = list[int | float]
Matrix = list[list[int | float]]


class Tensor:
    def __init__(
        self,
        data: Matrix,
        /
    ) -> None:
        self.__verify_data(data=data)

        self._data = data
        self._tensor = self.__format_tensor()

        self.rows = len(self._data)
        self.columns = len(self._data[0])
        self.shape = self.rows, self.columns

    def __verify_data(
        self,
        data: Matrix
    ) -> None:
        if len(data) == 0:
            raise ValueError('tensor cannot be empty')

        for index, row in enumerate(data):
            if not isinstance(row, list):
                raise TypeError('each row must be a list')

            if len(row) == 0:
                raise ValueError('row cannot be empty')

            if len(data[0]) != len(data[index]):
                raise TypeError('all lines must be the same size')

            for value in data[index]:
                if not isinstance(value, (int, float)):
                    raise TypeError('tensor values must be int or float')

        self.ndim = self.__get_dimensions(data)

        if self.ndim != 2:
            raise ValueError('the tensor must have a dimension equal to two')

    def __get_dimensions(self, data: Matrix | Vector) -> int:
        if isinstance(data[0], list):
            return 1 + self.__get_dimensions(data[0])

        return 1

    def __format_tensor(self) -> str:
        separator = f',\n{" "*8}'

        tensor = separator.join(
            [str(row) for row in self._data]
        )

        return tensor

    def __addsub(
        self,
        other: 'Tensor',
        add: bool = True
    ) -> 'Tensor':
        if self.shape != other.shape:
            ERROR = (
                'matrix A has a different format than matrix B '
                f'{self.shape} != {other.shape}'
            )

            raise ValueError(ERROR)

        matrix_add = [[
            float(value_a + value_b) if add else float(value_a - value_b)
            for value_a, value_b in zip(row_a, row_b)
        ]
            for row_a, row_b in zip(self._data, other._data)
        ]

        return Tensor(matrix_add)

    def __mul(self, other: int | float) -> 'Tensor':
        if not isinstance(other, (int, float)):
            ERROR = (
                'matrices can only be multiplied by '
                'integer or float scalars'
            )

            raise ValueError(ERROR)

        matrix_scalar = [[
                float(value_a*other)
                for value_a in row_a
            ]
            for row_a in self._data
        ]

        return Tensor(matrix_scalar)

    def __str__(self) -> str:
        return f'Tensor([{self._tensor}])'

    def __repr__(self) -> str:
        return f'Tensor({self._tensor}, shape={self.shape})'

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if self.columns != other.rows:
            ERROR = (
                f'matrix A has {self.columns} columns but matrix B '
                f'has {other.rows} rows ({self.columns} != {other.rows})'
            )

            raise ValueError(ERROR)

        matrix_product = [[
            sum([
                value_a*value_b
                for value_a, value_b in zip(row_a, row_b)
            ])
            for row_b in other.T._data
        ]
            for row_a in self._data
        ]

        return Tensor(matrix_product)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return self.__addsub(other=other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return self.__addsub(other=other, add=False)

    def __mul__(self, other: int | float) -> 'Tensor':
        return self.__mul(other=other)

    def __rmul__(self, other: int | float) -> 'Tensor':
        return self.__mul(other=other)

    def __truediv__(self, other: int | float) -> 'Tensor':
        return self.__mul(other=1/other)

    def __rtruediv__(self, other: int | float) -> 'Tensor':
        return self.__mul(other=1/other)

    def __getitem__(self, index: int) -> list[int | float]:
        return self._data[index]

    def __iter__(self) -> Iterator[Vector]:
        return iter(self._data)

    @property
    def T(self) -> 'Tensor':
        transposed_matrix = [
            list(row) for row in zip(*self._data)
        ]

        return Tensor(transposed_matrix)

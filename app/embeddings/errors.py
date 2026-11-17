class EmbeddingError(RuntimeError):
    pass


class EmbeddingDimensionMismatch(EmbeddingError, ValueError):
    def __init__(self, expected: int, actual: int | None) -> None:
        super().__init__(
            f"embedding model produces {actual} dimensions, expected {expected}"
        )

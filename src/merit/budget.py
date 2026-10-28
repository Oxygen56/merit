"""Hard budget enforcement shared by every optimizer adapter."""


class BudgetExceeded(RuntimeError):
    """Raised when an optimizer attempts to exceed its candidate budget."""


class Budget:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Budget limit must be positive")
        self.limit = limit
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self) -> None:
        if self.remaining <= 0:
            raise BudgetExceeded(f"Candidate evaluation budget exhausted: {self.limit}")
        self.used += 1

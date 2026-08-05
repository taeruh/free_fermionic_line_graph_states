from abc import ABC, abstractmethod
import numpy as np


class Coupling(ABC):
    @abstractmethod
    def sample(self) -> float:
        pass

    @abstractmethod
    def is_zero(self) -> bool:
        pass


class ConstantCoupling(Coupling):
    def __init__(self, value: float):
        self.value = value

    def sample(self) -> float:
        return self.value

    def is_zero(self) -> bool:
        return self.value == 0

    def __str__(self):
        return f"ConstantCoupling(value={self.value})"


class RandomCoupling(Coupling):
    def __init__(self, low: float, high: float, seed: int | None = None):
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def sample(self) -> float:
        return self.rng.uniform(self.low, self.high)

    def is_zero(self) -> bool:
        return self.low == 0 and self.high == 0

    def __str__(self):
        return f"RandomCoupling(low={self.low}, high={self.high})"

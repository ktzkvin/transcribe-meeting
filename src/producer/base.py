from typing import Any
from abc import ABC, abstractmethod


class BaseProducer(ABC):

    @abstractmethod
    def producer(self, value: Any) -> Any: ...

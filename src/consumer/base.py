from typing import Any, Optional
from abc import ABC, abstractmethod


class BaseConsumer(ABC):

    @abstractmethod
    def consume(self, value: Optional[Any] = None) -> Any: ...

from abc import ABC, abstractmethod

from bot.models import UserMessage


class BaseGPT(ABC):
    @abstractmethod
    async def ask(self, question: str, history: list[UserMessage]) -> str:
        ...

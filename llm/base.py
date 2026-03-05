from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        pass

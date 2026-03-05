from llm.base import LLMClient
from llm.ollama import OllamaClient
from llm.groq import GroqClient


class LLMFactory:
    @staticmethod
    def create(provider: str, **kwargs) -> LLMClient:
        if provider == "ollama":
            return OllamaClient(**kwargs)
        elif provider == "groq":
            return GroqClient(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

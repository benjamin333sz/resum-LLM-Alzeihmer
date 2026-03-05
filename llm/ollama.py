from llm.base import LLMClient
from langfuse import observe
import ollama


class OllamaClient(LLMClient):
    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.3):
        self.model = model
        self.default_temperature = temperature

    @observe(name="ollama_completion", as_type="generation")
    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> str:

        temp = temperature if temperature is not None else self.default_temperature

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temp},
            )

            return response["message"]["content"].strip()

        except Exception as e:
            raise RuntimeError(f"Ollama failed: {str(e)}")

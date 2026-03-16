from groq import Groq, RateLimitError
from llm.base import LLMClient
from langfuse import observe
import os
import time


class GroqClient(LLMClient):
    supports_parallelism = True

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.3,
        nb_retry: int = 5,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.nb_retry = nb_retry

    @observe(name="groq_completion", as_type="generation")
    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        nb_retry: int = 5,
    ) -> str:

        temp = temperature if temperature is not None else self.temperature
        response = None
        try:
            for attempt in range(nb_retry):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp,
                        max_tokens=max_tokens,
                    )
                    break

                except RateLimitError as e:
                    wait = 2**attempt
                    time.sleep(wait)

            if response is None:
                raise RuntimeError(f"Groq failed after {nb_retry} retries")

            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("Groq returned empty response")
            return content.strip()

        except Exception as e:
            raise RuntimeError(f"Groq completion failed: {str(e)}")

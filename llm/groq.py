from groq import Groq
from llm.base import LLMClient
from langfuse import observe
from dotenv import load_dotenv
import os


class GroqClient(LLMClient):
    def __init__(self, model: str = "openai/gpt-oss-120b", temperature: float = 0.3):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature

    @observe(name="groq_completion", as_type="generation")
    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> str:

        temp = temperature if temperature is not None else self.temperature
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"Groq completion failed: {str(e)}")

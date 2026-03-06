from abc import ABC, abstractmethod
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed


class LLMClient(ABC):
    supports_parallelism: bool = True

    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        pass

    def complete_parallel(self, prompts, max_workers=5, **kwargs):
        if not self.supports_parallelism:
            return [self.complete(p, **kwargs) for p in prompts]
        print("Process call in parallel")
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.complete, prompt, **kwargs): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results

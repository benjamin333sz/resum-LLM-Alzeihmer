from pathlib import Path
from langfuse import Langfuse, observe
from .langfuse_prompts import load_prompt_file, prompt_needs_update

langfuse = Langfuse()


def create_or_update_prompt(name: str, prompt_text: str, label: str, tags: list[str]):
    """Crée ou met à jour un prompt sur Langfuse."""
    langfuse.create_prompt(
        name=name,
        prompt=prompt_text,
        labels=[label],
        tags=tags,
    )


@observe(name="sync_and_load_prompts", as_type="tool")
def sync_and_load_prompts(prompt_config: dict) -> dict[str, str]:
    """
    Synchronise les prompts locaux avec Langfuse.

    - Met à jour seulement si le contenu diffère
    - Préserve les tags
    - Compatible avec les versions sans metadata
    """
    compiled = {}

    for name, cfg in prompt_config.items():
        label = cfg.get("label", "production")
        tags = cfg.get("tags", [])

        prompt_text = load_prompt_file(cfg["file"])

        try:
            remote_prompt = langfuse.get_prompt(name, label=label)
            remote_text = getattr(remote_prompt, "prompt", "")
            remote_tags = getattr(remote_prompt, "tags", []) or []

            if not prompt_needs_update(remote_text, prompt_text):
                compiled[name] = remote_prompt.compile()
                print(f"[Langfuse] Prompt '{name}' unchanged")
                continue

            print(f"[Langfuse] Updating prompt '{name}'")
            create_or_update_prompt(name, prompt_text, label, remote_tags)
            compiled[name] = langfuse.get_prompt(name, label=label).compile()

        except Exception:
            print(f"[Langfuse] Creating prompt '{name}'")
            create_or_update_prompt(name, prompt_text, label, tags)
            compiled[name] = langfuse.get_prompt(name, label=label).compile()

    return compiled

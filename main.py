# Load key of the project
from dotenv import load_dotenv

load_dotenv()

from utils.io import log_crash
from langfuse import get_client, observe
from tracking.prompt_registry import sync_and_load_prompts
from config.load_settings import load_settings
from pipeline.step1_collect.run import run_step1
from pipeline.step2_clustering.run import run_step2
from pipeline.step3_generation.run import run_step3
from llm.base import LLMClient
from llm.LLMFactory import LLMFactory
from datetime import datetime
from config.modalities.user_modalities import USER_MODALITIES
from config.modalities.alzheimer_modalities import MODALITIES

nb_paper = 10
provider = "ollama"
model = "gemma3:4b"
subject = "Alzheimer"
user_modalities = {}


@observe(name=f"{subject}_pipeline", as_type="chain")
def main(subject: str):
    print("Begin")
    subject = subject.lower()
    print(f"Subject : {subject} \n provider : {provider}\n model : {model}")
    llm = LLMFactory.create(provider=provider, model=model)
    # Load and synchronise prompts langfuse <-> local
    settings = load_settings()
    prompts = sync_and_load_prompts(settings["langfuse"]["prompts"])

    papers = run_step1(prompts=prompts, nb_paper=nb_paper, llm=llm, subject=subject)

    clusters = run_step2(
        papers=papers,
        prompts=prompts,
        llm=llm,
        subject=subject,
        user_modalities=user_modalities,
    )
    latex = run_step3(
        clusters=clusters,
        papers=papers,
        prompts=prompts,
        llm=llm,
        reviewer_iterations=1,
        subject=subject,
        provider=provider,
        model=model,
    )
    day = datetime.now().strftime("%d_%m_%Y")
    with open(
        f"results/State_of_the_art_{subject}_{day}.tex", "w", encoding="utf-8"
    ) as f:
        f.write(latex)
    print("End")


if __name__ == "__main__":
    try:
        main(subject)
    except Exception as e:
        log_crash(str(e))
        raise
    finally:
        get_client().flush()

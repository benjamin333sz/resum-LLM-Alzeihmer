# End-of-Study Project

This Python project is an Automated State-of-the-Art Pipeline to generate a **State-of-the-Art (SoTA)** review from scientific papers (e.g., arXiv).  
It combines metadata retrieval, LLM-based structured reasoning, clustering, and axis-based organization into a fully automated research synthesis workflow.

---

The user is prompted to make the changes they wish to generate the SoTA as they want.

---

# Pipeline Architecture

The pipeline runs in **three main steps**:

---

## Step 1 — Paper Collection & Enrichment

**Goal:** Retrieve relevant scientific papers and enrich them with metadata.

### What happens:

1. Papers are retrieved via arXiv API and are stored as a class `Paper`.
2. arXiv IDs are normalized (`XXXX.XXXXX`, version removed).
3. Papers are enriched using Semantic Scholar:
   - Citation counts
   - Additional metadata
4. The `.bib` file is generated or updated.
5. URLs are extracted from `.bib` when needed.

### Output:
- Structured `Paper` objects
- A clean `.bib` file
- Metadata-ready dataset

---

## Step 2 — Clustering & Modalities Construction

**Goal:** Identify research modalities (clusters) dynamically.

### Key idea:
The system iterates article-by-article and progressively builds a list of modalities.

For each article:
- The LLM assigns it to an existing modality giving by user **or**
- Creates a new modality if necessary

This ensures:
- Consistency across articles
- No modality duplication
- Dynamic update of the modality list

### Output:
```json
{
  "step": "step2_clustering",
  "method": ...,
  "model": ...,
  "n_clusters": X,
  "n_papers": X,
  "modalities": [

  "NEURO_IMAGE": [...],
  "ARCHITECTURE_GENETIQUE": [...],
  ...
  ]
}
```

## Step 3 — State-of-the-Art Generation

Generating a high-quality State-of-the-Art (SoTA) requires structured reasoning.  
We **do not** provide all articles and clusters at once and ask a model to “write a review”.

Instead, we decompose the process into controlled sub-steps to ensure coherence, traceability, and structural consistency.

---

### 3.1 Extract Research Axes

Before generating text, we identify the **research axes** that structure the cluster.

An axis represents a conceptual research direction, such as:

- Clinical Decision Support
- Biomarker Discovery
- Multimodal Learning
- Model Interpretability

Axes are extracted from the cluster content using a constrained prompt.  
They must:

- Be conceptually distinct
- Cover the major research directions
- Avoid redundancy
- Use precise scientific wording

This step ensures that the SoTA is not a flat summary, but a structured synthesis.

---

### 3.2 Assign Articles to Axes

Each article is processed with:

- Its arXiv ID (`XXXX.XXXXX` format)
- Its structured summary
- The predefined list of axes

A validation layer guarantees:

- All input IDs are present
- No unknown IDs are returned
- No hallucinated axes appear
- Maximum 2 axes per article
- Missing assignments default to []
- This ensures deterministic downstream grouping:

---

### 3.3 Generate Paragraphs for Each Axis

For each research axis:
1. Retrieve the list of assigned articles
2. Provide their summaries to the model
3. Generate a structured synthesis paragraph

Each paragraph must:
- Compare approaches
- Identify methodological trends
- Highlight contributions
- Mention limitations when relevant
- Avoid article-by-article listing
- Maintain scientific tone
- This produces coherent sub-sections of the SoTA.

---

### 3.4 Generate the Global Synthesis
Once all axis-level sections are generated, the model produces:
- A cross-axis synthesis
- Emerging research trends
- Methodological patterns
- Open challenges
- Future research directions

This ensures the final document is not a collection of independent paragraphs, but a unified scientific narrative.

---

### 3.5 Generate the LaTeX Version
The final SoTA is exported as a LaTeX-ready document.
Key features:
- Citation protection during generation
- Clean formatting
- Automatic bibliography integration:

# Environment Setup

For using this repo, please ensure first:
1. Created a Virtual Environnement
```bash
python -m venv venv
```
Then activate it:
- For macOS / Linux:
```bash
source venv/bin/activate
```
- For Windows:
```bash
venv\Scripts\activate
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Configure Environment Variables
Create a .env file at the root of the project and indicates the API_KEY you use.

**Please note that you have to create account following the provider you want to use.**

4. Choose your parameters in the main.py and run it.

5. Find your results in the folder ```results```

# LLM Backend
Currently, you can use Groq and Ollama provider.

Feel free to add your own LLM Backend.
For that, respect the following base :

```Python
class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        pass
```

and add in the class LLMFactory.py your new LLM Backend
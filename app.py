import streamlit as st
import subprocess
import sys
import threading
import queue
import time
from io import StringIO
from pathlib import Path
from datetime import datetime

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research SotA Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  .stApp {
    background: #0d0f14;
    color: #e8e6e0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #12151c;
    border-right: 1px solid #1e2330;
  }

  /* Title block */
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f0ede6;
    line-height: 1.15;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #5a6480;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  /* Step badges */
  .step-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    background: #1a1f2e;
    border: 1px solid #2a3048;
    color: #7b8ab8;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
  }

  /* Cards */
  .param-card {
    background: #12151c;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .param-card h4 {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4a5570;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.7rem;
  }

  /* Log terminal */
  .log-terminal {
    background: #080a0f;
    border: 1px solid #1a2040;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #5ee87a;
    height: 340px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
  }

  /* Run button override */
  .stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }
  .stButton > button:active { opacity: 0.75; }

  /* Status chips */
  .chip-idle    { color: #4a5570; }
  .chip-running { color: #f59e0b; }
  .chip-done    { color: #22c55e; }
  .chip-error   { color: #ef4444; }

  /* Metric boxes */
  [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem !important;
    color: #a5b4fc !important;
  }
  [data-testid="stMetricLabel"] { color: #4a5570 !important; }

  /* Inputs */
  .stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background: #0d0f14 !important;
    border-color: #1e2330 !important;
    color: #e8e6e0 !important;
  }
  .stSlider [data-testid="stThumbValue"] { color: #a5b4fc !important; }

  hr { border-color: #1e2330; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
PROVIDERS = ["ollama", "groq"]  # "openai", "anthropic", "mistral",

PROVIDER_MODELS = {
    "ollama": [
        "gemma3:4b",
        "phi3:mini",
        "qwen2.5:7b-instruct",
        "mistral:7b-instruct",
        "llama3.1:latest",
    ],
    # "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    # "anthropic": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
    # "mistral":   ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b"],
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
}


def build_command(params: dict) -> list[str]:
    """Turn UI params into a python -c ... call that imports and runs main()."""
    modalities_repr = repr(params["user_modalities"])
    code = f"""
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from main import main
main(
    subject={params['subject']!r},
    provider={params['provider']!r},
    model={params['model']!r},
    nb_paper={params['nb_paper']},
    max_articles_per_generation={params['max_articles_per_generation']!r},
    scholar_citation={params['scholar_citation']},
    user_modalities={modalities_repr},
    pdf_compilation={params['pdf_compilation']},
    batch_size={params['batch_size']},
)
"""
    return [sys.executable, "-c", code]


def stream_process(cmd: list[str], log_queue: queue.Queue):
    """Run cmd in a thread and push stdout/stderr lines to log_queue."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log_queue.put(("log", line.rstrip()))
        proc.wait()
        status = "done" if proc.returncode == 0 else "error"
        log_queue.put(("status", status))
    except Exception as e:
        log_queue.put(("log", f"[ERROR] {e}"))
        log_queue.put(("status", "error"))


# ── Session state init ───────────────────────────────────────────────────────
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "run_status" not in st.session_state:
    st.session_state.run_status = "idle"
if "log_queue" not in st.session_state:
    st.session_state.log_queue = None

# ── Layout ───────────────────────────────────────────────────────────────────
sidebar, main_col = st.columns([1, 2], gap="large")

# ═══════════════════════════  SIDEBAR  ═══════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="hero-title">SotA<br>Generator</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-sub">Research Pipeline v1.0</div>', unsafe_allow_html=True
    )

    # ── Subject ──────────────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">SUBJECT</div>', unsafe_allow_html=True)
    subject = st.text_input(
        "Research subject",
        value="Alzheimer",
        label_visibility="collapsed",
        placeholder="e.g. Alzheimer, RLHF, diffusion models…",
    )

    st.markdown("---")

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">MODEL</div>', unsafe_allow_html=True)
    provider = st.selectbox("Provider", PROVIDERS, index=0)
    model_options = PROVIDER_MODELS.get(provider, [])
    model = st.selectbox("Model", model_options, index=0)

    st.markdown("---")

    # ── Fetch ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">STEP 1 — FETCH</div>', unsafe_allow_html=True)
    nb_paper = st.number_input(
        "Number of papers", min_value=5, max_value=200, value=10, step=5
    )
    scholar_citation = st.toggle("Semantic Scholar citations", value=False)

    st.markdown("---")

    # ── Clustering ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="step-badge">STEP 2 — CLUSTERING</div>', unsafe_allow_html=True
    )
    custom_modalities_raw = st.text_area(
        "Custom modalities (JSON, optional)",
        placeholder='{"imaging": "MRI and PET studies", "genetics": "GWAS and omics"}',
        height=200,
    )
    user_modalities = {}
    if custom_modalities_raw.strip():
        import json

        try:
            user_modalities = json.loads(custom_modalities_raw)
            st.success(f"{len(user_modalities)} modalities parsed ✓")
        except json.JSONDecodeError:
            st.error("Invalid JSON — modalities ignored")

    st.markdown("---")

    # ── Generation ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="step-badge">STEP 3 — GENERATION</div>', unsafe_allow_html=True
    )
    max_articles_raw = st.number_input(
        "Max articles per section (0 = all)", min_value=0, value=0, step=1
    )
    max_articles_per_generation = (
        None if max_articles_raw == 0 else int(max_articles_raw)
    )

    batch_size = st.number_input(
        "Batch size (0 = sequential)", min_value=0, value=0, step=1
    )
    pdf_compilation = st.toggle("Compile to PDF", value=True)

    st.markdown("---")

    # ── Launch ────────────────────────────────────────────────────────────────
    run_disabled = st.session_state.run_status == "running"
    run_clicked = st.button("🚀  Run Pipeline", disabled=run_disabled)

# ═══════════════════════════  MAIN PANEL  ════════════════════════════════════
with main_col:

    # Status row
    status_map = {
        "idle": ("⬤", "chip-idle", "Idle — configure and launch"),
        "running": ("⬤", "chip-running", "Running…"),
        "done": ("⬤", "chip-done", "Completed successfully"),
        "error": ("⬤", "chip-error", "Finished with errors"),
    }
    dot, chip_cls, label = status_map[st.session_state.run_status]
    st.markdown(
        f'<span class="{chip_cls}" style="font-size:1.5rem">{dot}</span>'
        f' <span style="color:#a0aec0;font-size:0.9rem">{label}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Metrics preview
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Subject", subject or "—")
    m2.metric("Papers", nb_paper)
    m3.metric("Provider", provider)
    m4.metric("PDF", "Yes" if pdf_compilation else "No")

    st.markdown("---")

    # Log terminal
    st.markdown("**Live output**")
    log_placeholder = st.empty()

    def render_logs():
        content = (
            "\n".join(st.session_state.log_lines[-300:])
            or "Waiting for pipeline to start…"
        )
        log_placeholder.markdown(
            f'<div class="log-terminal">{content}</div>', unsafe_allow_html=True
        )

    render_logs()
    # ── Results browser ───────────────────────────────────────────────────────
    st.markdown("**Generated documents**")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    pdf_files = sorted(
        results_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    tex_files = sorted(
        results_dir.glob("*.tex"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    all_files = sorted(
        list(pdf_files) + list(tex_files),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not all_files:
        st.markdown(
            "<div style=\"color:#4a5570;font-family:'DM Mono',monospace;font-size:0.8rem;"
            'padding:1rem;background:#0d0f14;border:1px dashed #1e2330;border-radius:8px;">'
            "No files yet — run the pipeline to generate documents."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Refresh button
        if st.button("↻  Refresh list"):
            st.rerun()

        for fpath in all_files:
            is_pdf = fpath.suffix == ".pdf"
            icon = "📄" if is_pdf else "🔣"
            size_kb = fpath.stat().st_size / 1024
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime).strftime(
                "%d/%m/%Y %H:%M"
            )

            col_info, col_dl = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f"<div style=\"font-family:'DM Mono',monospace;font-size:0.78rem;\">"
                    f'{icon} <span style="color:#c4c9d8">{fpath.name}</span><br>'
                    f'<span style="color:#3a4460">{size_kb:.1f} KB · {mtime}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_dl:
                with open(fpath, "rb") as f:
                    mime = "application/pdf" if is_pdf else "text/x-tex"
                    st.download_button(
                        label="⬇ Download",
                        data=f.read(),
                        file_name=fpath.name,
                        mime=mime,
                        key=str(fpath),  # unique key per file
                    )
            st.markdown(
                '<hr style="border-color:#1a1f2e;margin:0.4rem 0;">',
                unsafe_allow_html=True,
            )

# ── Run trigger ───────────────────────────────────────────────────────────────
if run_clicked and st.session_state.run_status != "running":
    params = dict(
        subject=subject,
        provider=provider,
        model=model,
        nb_paper=nb_paper,
        max_articles_per_generation=max_articles_per_generation,
        scholar_citation=scholar_citation,
        user_modalities=user_modalities,
        pdf_compilation=pdf_compilation,
        batch_size=batch_size,
    )
    cmd = build_command(params)
    q: queue.Queue = queue.Queue()
    st.session_state.log_lines = [f"$ Running pipeline for: {subject!r}", ""]
    st.session_state.run_status = "running"
    st.session_state.log_queue = q
    t = threading.Thread(target=stream_process, args=(cmd, q), daemon=True)
    t.start()
    st.rerun()

# ── Poll queue while running ──────────────────────────────────────────────────
if st.session_state.run_status == "running" and st.session_state.log_queue:
    q = st.session_state.log_queue
    updated = False
    while True:
        try:
            kind, value = q.get_nowait()
            if kind == "log":
                st.session_state.log_lines.append(value)
                updated = True
            elif kind == "status":
                st.session_state.run_status = value
                updated = True
                break
        except queue.Empty:
            break
    if updated:
        render_logs()
    time.sleep(0.5)
    st.rerun()

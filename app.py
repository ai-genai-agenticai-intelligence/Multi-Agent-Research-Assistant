import base64
import html
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from groq import RateLimitError
from langchain_google_genai.chat_models import GoogleRateLimitError
from PIL import Image

from agents import critic_chain, writer_chain
from tools import scrape_url, web_search

ASSETS = Path(__file__).parent / "assets"
ICON_SVG = ASSETS / "icon.svg"
FAVICON_PNG = ASSETS / "favicon.png"
SAMPLE_TOPIC = "Where does fusion energy stand today?"
PIPELINE_STEPS = (
    ("search", "Search Agent", "Finds recent, relevant sources"),
    ("reader", "Reader Agent", "Reads the strongest source"),
    ("writer", "Writer Chain", "Builds a structured report"),
    ("critic", "Critic Chain", "Checks clarity and evidence"),
)
STEP_ICON_PATHS = {
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    "reader": (
        '<path d="M4 5a3 3 0 0 1 3-3h13v17H7a3 3 0 0 0-3 3z"/>' '<path d="M4 5v17"/>'
    ),
    "writer": (
        '<path d="M12 20h9"/>' '<path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>'
    ),
    "critic": (
        '<path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
}

SAMPLE_RESULTS = {
    "search": """Title: ITER - The Way to New Energy
URL: https://www.iter.org/
Snippet: An international collaboration building an experimental fusion reactor.

Title: Fusion Energy Sciences
URL: https://www.energy.gov/science/fes/fusion-energy-sciences
Snippet: U.S. Department of Energy research programs advancing fusion science.

Title: National Ignition Facility
URL: https://lasers.llnl.gov/
Snippet: Research and updates from the world's largest and highest-energy laser.

Title: Nature - Fusion Energy
URL: https://www.nature.com/subjects/fusion-energy
Snippet: Research and reporting on the science and technology of fusion energy.""",
    "reader": """SAMPLE READER OUTPUT

Fusion research has achieved important experimental milestones, while sustained,
commercially viable electricity production remains a future engineering goal.
Current programs address plasma control, materials that tolerate neutron exposure,
and reliable fuel-cycle systems.""",
    "writer": """# Where does fusion energy stand today?

Fusion has passed important scientific milestones, but it is not yet a source of
commercial electricity. Public programs and private companies are working to turn
promising experiments into reliable power systems.

## What has changed

- Experiments have demonstrated that fusion reactions can release substantial energy
  under carefully controlled conditions.
- International projects are developing larger facilities to study long-duration
  plasma behavior and reactor-scale engineering.
- Materials, heat handling, fuel supply, and maintenance remain major challenges.

## Bottom line

Fusion is an active research and engineering field, not a near-term replacement for
today's power sources. The next milestones are repeatable operation, integrated
reactor systems, and a credible path to economical electricity.

*Demo report: this saved example is for interface preview only and is not live research.*""",
    "critic": """Score: 8/10

### Strengths
- Clearly distinguishes experimental progress from commercial power generation.
- Identifies key engineering challenges.
- Uses cautious language about timelines and readiness.

### Areas to improve
- A live report should cite specific experiments, dates, and primary sources.
- Compare the approaches and timelines of major public and private programs.""",
}

_SOURCE_PATTERN = re.compile(
    r"Title:\s*(?P<title>[^\n]+)\s*\nURL:\s*(?P<url>https?://[^\s]+)"
    r"(?:\s*\nSnippet:\s*(?P<snippet>[^\n]*))?",
    re.IGNORECASE,
)


def load_favicon():
    if FAVICON_PNG.is_file():
        return Image.open(FAVICON_PNG)
    return "🔬"


def logo_data_uri() -> str:
    if not ICON_SVG.is_file():
        return ""
    encoded = base64.b64encode(ICON_SVG.read_bytes()).decode()
    return f"data:image/svg+xml;base64,{encoded}"


def extract_sources(text: str) -> list[dict[str, str]]:
    sources = []
    seen_urls = set()
    for match in _SOURCE_PATTERN.finditer(text):
        url = match.group("url").rstrip(".,);]")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append(
            {
                "title": match.group("title").strip(),
                "url": url,
                "snippet": (match.group("snippet") or "").strip(),
                "domain": urlparse(url).netloc.removeprefix("www."),
            }
        )
    return sources


def critic_score(text: str) -> str:
    match = re.search(r"\bScore:\s*(\d+(?:\.\d+)?)\s*/\s*10\b", text, re.IGNORECASE)
    return f"{match.group(1)} / 10" if match else "—"


class SearchResultsUnavailableError(RuntimeError):
    pass


def load_demo_results() -> None:
    st.session_state.results = dict(SAMPLE_RESULTS)
    st.session_state.running = False
    st.session_state.done = True
    st.session_state.active_step = None
    st.session_state.started_at = None
    st.session_state.elapsed_seconds = 4.1
    st.session_state.step_times = {
        "search": 1.0,
        "reader": 1.1,
        "writer": 1.2,
        "critic": 0.8,
    }
    st.session_state.active_topic = SAMPLE_TOPIC
    st.session_state.results_is_demo = True
    st.session_state.run_error = None


def on_demo_mode_change() -> None:
    if st.session_state.demo_mode:
        st.session_state.research_topic_input = SAMPLE_TOPIC
        load_demo_results()
    elif st.session_state.results_is_demo:
        st.session_state.results = {}
        st.session_state.running = False
        st.session_state.done = False
        st.session_state.active_step = None
        st.session_state.elapsed_seconds = 0.0
        st.session_state.step_times = {}
        st.session_state.active_topic = None
        st.session_state.results_is_demo = False
        st.session_state.run_error = None


def select_example(topic: str) -> None:
    st.session_state.research_topic_input = topic


def render_pipeline(slot, results: dict[str, str], active_step: str | None) -> None:
    elapsed = st.session_state.elapsed_seconds
    if st.session_state.running and st.session_state.started_at:
        elapsed = time.time() - st.session_state.started_at
        status = f"RUNNING · {elapsed:.1f}s"
    elif st.session_state.done:
        status = f"COMPLETE · {elapsed:.1f}s total"
    elif st.session_state.run_error:
        status = "PAUSED"
    else:
        status = "READY"

    completed = sum(1 for key, _, _ in PIPELINE_STEPS if key in results)
    progress = 1.0 if st.session_state.done else completed / len(PIPELINE_STEPS)
    if active_step:
        progress = max(progress, (completed + 0.35) / len(PIPELINE_STEPS))

    with slot.container():
        st.markdown(
            f'<div class="pipeline-heading"><span>RESEARCH PIPELINE</span>'
            f'<span class="pipeline-status">{html.escape(status)}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(progress)
        columns = st.columns(4, gap="small")
        for column, (key, title, description) in zip(columns, PIPELINE_STEPS):
            if key in results:
                state = "done"
                duration = st.session_state.step_times.get(key)
                detail = (
                    f"COMPLETE · {duration:.1f}s"
                    if duration is not None
                    else "COMPLETE"
                )
                marker = "✓"
            elif key == active_step:
                state = "active"
                detail = "IN PROGRESS"
                marker = "…"
            else:
                state = "waiting"
                detail = "UP NEXT"
                marker = "·"
            column.markdown(
                f"""
<div class="agent-card {state}">
  <div class="agent-card-top"><span class="agent-marker">{marker}</span>
    <span class="agent-state">{detail}</span></div>
  <div class="agent-title">
    <svg class="agent-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
      aria-hidden="true">{STEP_ICON_PATHS[key]}</svg>
    <span>{title}</span>
  </div>
  <div class="agent-description">{description}</div>
</div>
""",
                unsafe_allow_html=True,
            )


st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon=load_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ink: #e8edf5;
    --muted: #91a0b4;
    --faint: #718198;
    --line: #253247;
    --panel: #111a29;
    --accent: #73e0c2;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp {
    background:
        radial-gradient(ellipse 52% 38% at 50% 0%, rgba(39, 98, 102, 0.15), transparent 72%),
        #080d17;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 820px !important; padding: 1.5rem 1rem 2.5rem !important; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #263449 !important;
    border-radius: 15px !important;
    background: rgba(12, 19, 31, 0.97) !important;
    padding: 1.15rem 1.3rem !important;
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.24);
}
.topbar {
    display: grid;
    grid-template-columns: 2.7rem 1fr auto;
    align-items: center;
    gap: 0.75rem;
    padding: 0.15rem 0 1.15rem;
    border-bottom: 1px solid var(--line);
}
.brand-icon {
    width: 2.55rem; height: 2.55rem; border-radius: 10px;
    display: grid; place-items: center; overflow: hidden;
    background: #102a2a; border: 1px solid #24514a;
}
.brand-icon img { width: 100%; height: 100%; object-fit: cover; }
.brand-fallback { color: var(--accent); font: 800 1rem 'Manrope', sans-serif; }
.brand-copy { text-align: center; }
.brand-name { color: #eaf0f8; font: 700 0.76rem 'Manrope', sans-serif; letter-spacing: 0.15em; }
.brand-subtitle { margin-top: 0.18rem; color: var(--faint); font: 0.62rem 'DM Mono', monospace; letter-spacing: 0.1em; }
.mode-pill {
    padding: 0.38rem 0.58rem; border: 1px solid #295448; border-radius: 999px;
    color: #9ce8d3; background: rgba(31, 100, 79, 0.15);
    font: 500 0.62rem 'DM Mono', monospace; white-space: nowrap;
}
.mode-pill.live { color: #f3c778; border-color: #705d35; background: rgba(113, 83, 24, 0.16); }
.hero { padding: 1.7rem 0 1.05rem; }
.eyebrow, .pipeline-heading, .report-eyebrow {
    color: #83bba9; font: 500 0.62rem 'DM Mono', monospace; letter-spacing: 0.13em;
}
.hero h1 {
    margin: 0.5rem 0 0.45rem; color: #f0f3f8;
    font: 700 clamp(2rem, 5vw, 2.65rem)/1.12 'Manrope', sans-serif; letter-spacing: -0.045em;
}
.hero h1 span { color: #81dfc3; }
.hero-copy { margin: 0; color: #9aa8bb; font-size: 0.92rem; }
.demo-label { padding-top: 0.3rem; color: #b5c1d0; font-size: 0.78rem; }
.demo-caption { margin: -0.25rem 0 0.65rem; color: #78889d; font-size: 0.7rem; }
.st-key-research_topic_input input {
    min-height: 2.85rem; border: 1px solid #2b3a50 !important; border-radius: 9px !important;
    background: #101927 !important; color: #edf2f8 !important; font-size: 0.86rem !important;
}
.st-key-research_topic_input input:focus { border-color: #59bca1 !important; box-shadow: 0 0 0 1px #59bca1 !important; }
.suggestion-heading {
    margin: 0.2rem 0 0.35rem; color: #78889d;
    font: 500 0.58rem 'DM Mono', monospace; letter-spacing: 0.1em;
}
.st-key-example_topic_0 button, .st-key-example_topic_1 button, .st-key-example_topic_2 button {
    min-height: 2rem; padding: 0.1rem 0.55rem !important;
    border: 1px solid #2b3a50 !important; border-radius: 999px !important;
    background: #101927 !important; color: #aab8ca !important; font-size: 0.64rem !important;
    white-space: normal !important;
}
.st-key-example_topic_0 button p, .st-key-example_topic_1 button p, .st-key-example_topic_2 button p {
    white-space: normal !important; overflow-wrap: anywhere; text-overflow: clip !important;
}
.st-key-example_topic_0 button:hover, .st-key-example_topic_1 button:hover, .st-key-example_topic_2 button:hover {
    border-color: #59bca1 !important; color: #9ce8d3 !important;
}
.st-key-run_research_button button {
    min-height: 2.85rem; border: 1px solid #80e2c4 !important; border-radius: 9px !important;
    background: #80e2c4 !important; color: #081713 !important; font-weight: 700 !important;
}
.st-key-run_research_button button:hover { background: #a1eed6 !important; border-color: #a1eed6 !important; }
.pipeline-heading { display: flex; justify-content: space-between; margin: 1.45rem 0 0.55rem; }
.pipeline-status { color: #a9b5c5; }
div[data-testid="stProgress"] > div { background: #1b2939; }
div[data-testid="stProgress"] > div > div { background: #64d5b5; }
.agent-card {
    min-height: 6.1rem; padding: 0.72rem 0.67rem; border: 1px solid #28364a;
    border-radius: 9px; background: #101927;
}
.agent-card.done { border-color: #286151; background: linear-gradient(150deg, #10241f, #101927); }
.agent-card.active { border-color: #69cdb1; box-shadow: 0 0 0 1px rgba(105, 205, 177, 0.15); }
.agent-card-top { display: flex; align-items: center; gap: 0.35rem; margin-bottom: 0.53rem; }
.agent-marker { color: #79dfc1; font-size: 0.9rem; }
.agent-state { color: #7c8da3; font: 500 0.53rem 'DM Mono', monospace; }
.agent-title { display: flex; align-items: center; gap: 0.4rem; color: #e6edf5; font: 600 0.73rem 'Manrope', sans-serif; }
.agent-icon { width: 0.95rem; height: 0.95rem; flex: 0 0 auto; color: #79dfc1; }
.agent-card.waiting .agent-icon { color: #8393a8; }
.agent-description { margin-top: 0.25rem; color: #8493a7; font-size: 0.62rem; line-height: 1.35; }
.demo-notice {
    margin: 1rem 0 0.7rem; padding: 0.65rem 0.75rem; border: 1px solid #285647;
    border-radius: 8px; background: rgba(25, 82, 64, 0.15); color: #a9dfce; font-size: 0.74rem;
}
.metric-card {
    min-height: 4.6rem; padding: 0.75rem 0.85rem; border: 1px solid #263449;
    border-radius: 9px; background: #101927;
}
.metric-label { color: #8493a7; font: 500 0.61rem 'DM Mono', monospace; letter-spacing: 0.07em; text-transform: uppercase; }
.metric-value { margin-top: 0.4rem; color: #eaf0f8; font: 700 1.18rem 'Manrope', sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap: 1.1rem; border-bottom: 1px solid #263449; overflow-x: auto; }
.stTabs [data-baseweb="tab"] { height: 2.65rem; padding: 0 0.1rem; color: #8d9bb0; font-size: 0.76rem; flex: 0 0 auto; white-space: nowrap; }
.stTabs [aria-selected="true"] { color: #89e2c6 !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #74dfc0 !important; }
.report-eyebrow { margin: 0.35rem 0 0.8rem; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #edf2f8; font-family: 'Manrope', sans-serif; }
.stMarkdown p, .stMarkdown li { color: #c2ccda; line-height: 1.68; font-size: 0.87rem; }
.source-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1rem; }
.source-chip {
    padding: 0.32rem 0.55rem; border: 1px solid #34445a; border-radius: 999px;
    color: #a8c8c0 !important; background: #111e2a; font: 0.66rem 'DM Mono', monospace;
    text-decoration: none !important;
}
.source-chip:hover { border-color: #70d7b8; color: #8ee4c8 !important; }
.source-item { padding: 0.7rem 0; border-bottom: 1px solid #263449; }
.source-item-title { color: #e3eaf3; font-size: 0.82rem; font-weight: 600; }
.source-item-domain { margin: 0.22rem 0; color: #86cbb5; font: 0.64rem 'DM Mono', monospace; }
.source-item-snippet { color: #a5b1c1; font-size: 0.75rem; line-height: 1.5; }
.stAlert { border-radius: 8px; }
@media (min-width: 1280px) {
    .block-container { max-width: 940px !important; }
}
@media (max-width: 900px) {
    .block-container { max-width: 820px !important; }
    .stHorizontalBlock:has(.agent-card) {
        display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.6rem !important;
    }
    .stHorizontalBlock:has(.agent-card) > div[data-testid="stColumn"] {
        width: auto !important; min-width: 0 !important; max-width: none !important; flex: none !important;
    }
    .stHorizontalBlock:has(.st-key-example_topic_0) {
        display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.5rem !important;
    }
    .stHorizontalBlock:has(.st-key-example_topic_0) > div[data-testid="stColumn"] {
        width: auto !important; min-width: 0 !important; max-width: none !important; flex: none !important;
    }
}
@media (max-width: 640px) {
    .block-container { max-width: 100% !important; padding: 0.6rem 0.4rem 1.25rem !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.75rem !important; }
    .topbar { grid-template-columns: 2.2rem minmax(0, 1fr) auto; gap: 0.45rem; padding-bottom: 0.85rem; }
    .brand-icon { width: 2.1rem; height: 2.1rem; }
    .brand-name { font-size: 0.64rem; letter-spacing: 0.1em; }
    .brand-subtitle { font-size: 0.52rem; letter-spacing: 0.04em; }
    .mode-pill { padding: 0.32rem 0.45rem; font-size: 0.54rem; }
    .hero { padding: 1.25rem 0 0.85rem; }
    .hero h1 { font-size: clamp(1.8rem, 8vw, 2.35rem); }
    .hero-copy { max-width: 35ch; font-size: 0.84rem; }
    .stHorizontalBlock:has(.st-key-demo_mode) { flex-wrap: nowrap !important; gap: 0.45rem !important; }
    .stHorizontalBlock:has(.st-key-demo_mode) > div[data-testid="stColumn"]:first-child {
        width: 2.5rem !important; min-width: 2.5rem !important; max-width: 2.5rem !important; flex: 0 0 2.5rem !important;
    }
    .stHorizontalBlock:has(.st-key-demo_mode) > div[data-testid="stColumn"]:last-child {
        width: auto !important; min-width: 0 !important; flex: 1 1 auto !important;
    }
    .stHorizontalBlock:has(.st-key-research_topic_input) {
        display: flex !important; flex-direction: column; align-items: stretch; gap: 0.45rem !important;
    }
    .stHorizontalBlock:has(.st-key-research_topic_input) > div[data-testid="stColumn"] {
        width: 100% !important; min-width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important;
    }
    .stHorizontalBlock:has(.st-key-example_topic_0) {
        display: grid !important; grid-template-columns: minmax(0, 1fr); gap: 0.4rem !important;
    }
    .stHorizontalBlock:has(.st-key-example_topic_0) > div[data-testid="stColumn"] {
        width: 100% !important; min-width: 0 !important; max-width: none !important; flex: none !important;
    }
    .st-key-example_topic_0 button, .st-key-example_topic_1 button, .st-key-example_topic_2 button {
        min-height: 2.4rem; border-radius: 8px !important; text-align: left !important;
    }
    .stHorizontalBlock:has(.metric-card) {
        display: grid !important; grid-template-columns: minmax(0, 1fr); gap: 0.45rem !important;
    }
    .stHorizontalBlock:has(.metric-card) > div[data-testid="stColumn"] {
        width: 100% !important; min-width: 0 !important; max-width: none !important; flex: none !important;
    }
    .stHorizontalBlock:has(.st-key-export_report_button) {
        display: flex !important; flex-direction: column; gap: 0.35rem !important;
    }
    .stHorizontalBlock:has(.st-key-export_report_button) > div[data-testid="stColumn"] {
        width: 100% !important; min-width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.8rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.7rem; }
    .agent-card { min-height: 6rem; padding: 0.58rem; }
    .agent-title { font-size: 0.68rem; }
    .agent-description { font-size: 0.6rem; }
}
@media (max-width: 380px) {
    .brand-subtitle { display: none; }
    .mode-pill { padding: 0.28rem 0.38rem; font-size: 0.5rem; }
    .stHorizontalBlock:has(.agent-card) { grid-template-columns: minmax(0, 1fr); }
    .agent-card { min-height: 5.3rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

state_defaults = {
    "results": {},
    "running": False,
    "done": False,
    "active_step": None,
    "started_at": None,
    "elapsed_seconds": 0.0,
    "step_times": {},
    "run_error": None,
    "active_topic": None,
    "results_is_demo": False,
    "demo_mode": True,
    "research_topic_input": SAMPLE_TOPIC,
}
for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "demo_initialized" not in st.session_state:
    st.session_state.research_topic_input = SAMPLE_TOPIC
    load_demo_results()
    st.session_state.demo_initialized = True

logo_uri = logo_data_uri()
logo_markup = (
    f'<img src="{logo_uri}" alt="ResearchMind" />'
    if logo_uri
    else '<span class="brand-fallback">R</span>'
)
mode_class = "" if st.session_state.demo_mode else " live"
mode_label = "DEMO MODE ON" if st.session_state.demo_mode else "LIVE MODE"

with st.container(border=True, key="app_shell"):
    st.markdown(
        f"""
<div class="topbar">
  <div class="brand-icon">{logo_markup}</div>
  <div class="brand-copy"><div class="brand-name">RESEARCHMIND</div>
    <div class="brand-subtitle">MULTI-AGENT RESEARCH WORKSPACE</div></div>
  <div class="mode-pill{mode_class}">{mode_label}</div>
</div>
<div class="hero">
  <div class="eyebrow">MULTI-AGENT RESEARCH, MADE CLEAR</div>
  <h1>Get to the <span>signal.</span></h1>
  <p class="hero-copy">A coordinated research team that finds, reads, writes, and reviews.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    toggle_col, description_col = st.columns([0.55, 7], vertical_alignment="center")
    with toggle_col:
        st.toggle(
            "Demo mode",
            key="demo_mode",
            label_visibility="collapsed",
            on_change=on_demo_mode_change,
        )
    with description_col:
        if st.session_state.demo_mode:
            st.markdown(
                '<div class="demo-label">Demo mode · saved sample output</div>'
                '<div class="demo-caption">No API or web-search calls are made in demo mode.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="demo-label">Live research · uses configured provider and search API</div>'
                '<div class="demo-caption">Live runs may use provider quota and incur charges.</div>',
                unsafe_allow_html=True,
            )

    topic_col, run_col = st.columns([5.4, 1.45], vertical_alignment="bottom")
    with topic_col:
        topic = st.text_input(
            "Research topic",
            key="research_topic_input",
            label_visibility="collapsed",
            placeholder="Ask a research question...",
        )
    with run_col:
        run_btn = st.button(
            "Run research →",
            key="run_research_button",
            use_container_width=True,
            disabled=st.session_state.running,
        )

    st.markdown(
        '<div class="suggestion-heading">SUGGESTED QUESTIONS · CLICK TO FILL</div>',
        unsafe_allow_html=True,
    )
    suggestions = (
        SAMPLE_TOPIC,
        "What are the latest advances in CRISPR gene editing?",
        "How is quantum computing progressing?",
    )
    suggestion_columns = st.columns(3, gap="small")
    for index, question in enumerate(suggestions):
        with suggestion_columns[index]:
            st.button(
                question,
                key=f"example_topic_{index}",
                on_click=select_example,
                args=(question,),
                use_container_width=True,
            )

    pipeline_slot = st.empty()
    current_results = st.session_state.results
    render_pipeline(pipeline_slot, current_results, st.session_state.active_step)

    if st.session_state.results_is_demo:
        st.markdown(
            '<div class="demo-notice"><strong>Saved demo output</strong> · '
            "This is a fixed interface preview, not live research. No API calls were made.</div>",
            unsafe_allow_html=True,
        )
    elif not current_results and not st.session_state.running:
        st.info("Enter a question and run research to create a live report.")

    if st.session_state.run_error:
        st.error(st.session_state.run_error)

    current_sources = extract_sources(current_results.get("search", ""))
    current_critic = current_results.get("critic", "")
    score = critic_score(current_critic) if current_critic else "—"
    elapsed_label = (
        f"{st.session_state.elapsed_seconds:.1f}s" if st.session_state.done else "—"
    )
    metric_columns = st.columns(3, gap="small")
    for column, label, value in zip(
        metric_columns,
        ("CRITIC SCORE", "TOTAL TIME", "SOURCES FOUND"),
        (score, elapsed_label, str(len(current_sources)) if current_results else "—"),
    ):
        column.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    report_tab, critic_tab, sources_tab, raw_tab = st.tabs(
        ["Report", "Critic feedback", f"Sources ({len(current_sources)})", "Raw output"]
    )
    with report_tab:
        report_text = current_results.get("writer", "")
        if report_text:
            title_col, export_col = st.columns([4, 1.15], vertical_alignment="center")
            with title_col:
                st.markdown(
                    '<div class="report-eyebrow">FINAL RESEARCH REPORT</div>',
                    unsafe_allow_html=True,
                )
            with export_col:
                st.download_button(
                    "Export .md",
                    data=report_text,
                    file_name="research_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="export_report_button",
                )
            st.markdown(report_text)
            chips = "".join(
                f'<a class="source-chip" href="{html.escape(source["url"], quote=True)}" '
                f'target="_blank" rel="noopener noreferrer">{html.escape(source["domain"])}</a>'
                for source in current_sources[:4]
            )
            if chips:
                st.markdown(
                    f'<div class="source-chips">{chips}</div>',
                    unsafe_allow_html=True,
                )
        elif st.session_state.running:
            st.info("Your report will appear here when the pipeline finishes.")
        else:
            st.caption("Your completed report will appear here.")

    with critic_tab:
        if current_critic:
            st.markdown(current_critic)
        else:
            st.caption("Critic feedback will appear after the report is reviewed.")

    with sources_tab:
        if current_sources:
            for index, source in enumerate(current_sources, start=1):
                st.markdown(
                    f'<div class="source-item"><div class="source-item-title">'
                    f'{index}. {html.escape(source["title"])}</div>'
                    f'<div class="source-item-domain">{html.escape(source["domain"])}</div>'
                    f'<div class="source-item-snippet">{html.escape(source["snippet"])}</div>'
                    f'<a class="source-chip" href="{html.escape(source["url"], quote=True)}" '
                    f'target="_blank" rel="noopener noreferrer">Open source ↗</a></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Sources will appear here after the search agent runs.")

    with raw_tab:
        if current_results:
            for key, label, _ in PIPELINE_STEPS:
                if key in current_results:
                    st.markdown(f"**{label}**")
                    st.code(current_results[key], language=None)
        else:
            st.caption("Agent outputs will appear here after a run.")

if run_btn:
    if not topic or not topic.strip():
        st.warning("Please enter a research topic first.")
    elif st.session_state.demo_mode:
        if topic.strip() != SAMPLE_TOPIC:
            st.warning(
                "Demo mode displays the saved fusion-energy example. Turn off Demo mode "
                "to research a different topic."
            )
        else:
            load_demo_results()
            st.rerun()
    else:
        st.session_state.active_topic = topic.strip()
        st.session_state.results = {}
        st.session_state.results_is_demo = False
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.active_step = "search"
        st.session_state.started_at = time.time()
        st.session_state.elapsed_seconds = 0.0
        st.session_state.step_times = {}
        st.session_state.run_error = None
        st.rerun()

if st.session_state.running and not st.session_state.done:
    results = dict(st.session_state.results)
    topic_value = st.session_state.active_topic

    def run_agent_step(key: str, action) -> str:
        st.session_state.active_step = key
        render_pipeline(pipeline_slot, results, key)
        started = time.time()
        output = action()
        if not isinstance(output, str):
            raise TypeError(f"{key} returned {type(output).__name__}, expected text.")
        results[key] = output
        st.session_state.results = dict(results)
        st.session_state.step_times[key] = time.time() - started
        step_index = next(i for i, step in enumerate(PIPELINE_STEPS) if step[0] == key)
        next_step = (
            PIPELINE_STEPS[step_index + 1][0]
            if step_index + 1 < len(PIPELINE_STEPS)
            else None
        )
        st.session_state.active_step = next_step
        render_pipeline(pipeline_slot, results, next_step)
        return output

    try:
        search_output = run_agent_step(
            "search",
            lambda: web_search.invoke({"query": topic_value}),
        )
        sources = extract_sources(search_output)
        if not sources:
            raise SearchResultsUnavailableError(
                "Web search returned no usable source URLs. Try a broader topic."
            )
        reader_output = run_agent_step(
            "reader",
            lambda: scrape_url.invoke({"url": sources[0]["url"]}),
        )
        research_combined = (
            f"SEARCH RESULTS:\n{search_output}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{reader_output}"
        )
        writer_output = run_agent_step(
            "writer",
            lambda: writer_chain.invoke(
                {"topic": topic_value, "research": research_combined}
            ),
        )
        run_agent_step(
            "critic",
            lambda: critic_chain.invoke({"report": writer_output}),
        )
    except SearchResultsUnavailableError as error:
        st.session_state.running = False
        st.session_state.done = False
        st.session_state.active_step = None
        st.session_state.results = dict(results)
        st.session_state.elapsed_seconds = time.time() - st.session_state.started_at
        st.session_state.run_error = str(error)
        st.rerun()
    except GoogleRateLimitError:
        st.session_state.running = False
        st.session_state.done = False
        st.session_state.active_step = None
        st.session_state.results = dict(results)
        st.session_state.elapsed_seconds = time.time() - st.session_state.started_at
        st.session_state.run_error = (
            "Gemini's quota for this project or model is unavailable. Use a model "
            "included in your Google AI Studio tier (Gemini 4 Hydrogen supports "
            "Free tier), check your active limits or billing, then press Run research "
            "to retry. Your topic is saved."
        )
        st.rerun()
    except RateLimitError:
        st.session_state.running = False
        st.session_state.done = False
        st.session_state.active_step = None
        st.session_state.results = dict(results)
        st.session_state.elapsed_seconds = time.time() - st.session_state.started_at
        st.session_state.run_error = (
            "Groq's token limit has been reached. Wait for the limit to reset, or "
            "set LLM_PROVIDER in .env to another configured provider (gemini or "
            "openai) and restart the app. Your topic is saved; press Run research "
            "to retry."
        )
        st.rerun()
    else:
        st.session_state.running = False
        st.session_state.done = True
        st.session_state.active_step = None
        st.session_state.results_is_demo = False
        st.session_state.elapsed_seconds = time.time() - st.session_state.started_at
        st.rerun()

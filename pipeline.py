import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agents import writer_chain, critic_chain
from tools import scrape_url, web_search


def _first_source_url(search_text: str) -> str | None:
    for line in search_text.splitlines():
        if line.startswith("URL: "):
            url = line[5:].strip().rstrip(".,);]")
            if url.startswith(("http://", "https://")):
                return url
    return None


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # search agent working
    print("\n" + " =" * 50, flush=True)
    print("step 1 - search agent is working ...", flush=True)
    print("=" * 50, flush=True)

    state["search_results"] = web_search.invoke({"query": topic})

    print("\n search result:\n", state["search_results"], flush=True)

    # step 2 - reader agent
    print("\n" + " =" * 50, flush=True)
    print("step 2 - Reader agent is scraping top resources ...", flush=True)
    print("=" * 50, flush=True)

    source_url = _first_source_url(state["search_results"])
    if not source_url:
        raise ValueError("Web search returned no usable source URLs.")
    state["scraped_content"] = scrape_url.invoke({"url": source_url})

    print("\nscraped content:\n", state["scraped_content"], flush=True)

    # step 3 - writer chain

    print("\n" + " =" * 50, flush=True)
    print("step 3 - Writer is drafting the report ...", flush=True)
    print("=" * 50, flush=True)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke(
        {"topic": topic, "research": research_combined}
    )

    print("\n Final Report:\n", state["report"], flush=True)

    # critic report

    print("\n" + " =" * 50, flush=True)
    print("step 4 - critic is reviewing the report", flush=True)
    print("=" * 50, flush=True)

    state["feedback"] = critic_chain.invoke({"report": state["report"]})

    print("\n critic report:\n", state["feedback"], flush=True)

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

def _get_text(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)

def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    print("\n" + " =" * 50, flush=True)
    print("step 1 - search agent is working ...", flush=True)
    print("=" * 50, flush=True)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}. Ensure all source URLs are included in your output.")]
    })

    # Extract LLM summary and any tool outputs containing URLs
    tool_outputs = [
        str(msg.content)
        for msg in search_result.get("messages", [])
        if getattr(msg, "name", None) == "web_search" or getattr(msg, "type", None) == "tool"
    ]

    final_search_text = _get_text(search_result['messages'][-1].content)
    if tool_outputs:
        state["search_results"] = f"{final_search_text}\n\nSources Found:\n" + "\n".join(tool_outputs)
    else:
        state["search_results"] = final_search_text

    print("\n search result:\n", state['search_results'], flush=True)

    #step 2 - reader agent 
    print("\n" + " =" * 50, flush=True)
    print("step 2 - Reader agent is scraping top resources ...", flush=True)
    print("=" * 50, flush=True)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and call scrape_url on it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:3000]}"
        )]
    })

    state['scraped_content'] = _get_text(reader_result['messages'][-1].content)

    print("\nscraped content:\n", state['scraped_content'], flush=True)

    #step 3 - writer chain 

    print("\n" + " =" * 50, flush=True)
    print("step 3 - Writer is drafting the report ...", flush=True)
    print("=" * 50, flush=True)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report:\n", state['report'], flush=True)

    #critic report 

    print("\n" + " =" * 50, flush=True)
    print("step 4 - critic is reviewing the report", flush=True)
    print("=" * 50, flush=True)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report:\n", state['feedback'], flush=True)

    return state



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
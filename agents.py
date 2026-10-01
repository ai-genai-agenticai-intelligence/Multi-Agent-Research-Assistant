import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

load_dotenv()

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "").lower()
    
    # 1. Groq (High rate limits, generous free tier)
    if provider == "groq" or (not provider and os.getenv("GROQ_API_KEY")):
        from langchain_groq import ChatGroq
        groq_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        return ChatGroq(model=model, api_key=groq_key, max_tokens=1200, temperature=0)
    
    # 2. OpenAI
    if provider == "openai" or (not provider and os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY")):
        from langchain_openai import ChatOpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, api_key=openai_key, temperature=0)
    
    # 3. Google Gemini (Default)
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    return ChatGoogleGenerativeAI(
        model=gemini_model,
        api_key=gemini_key,
        temperature=0
    )

llm = get_llm()


# 1st agent: Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are an expert research search agent. Always use the web_search tool to find accurate information. "
            "In your final response, provide a comprehensive summary and ALWAYS explicitly list every source title and URL found."
        )
    )


# 2nd agent: Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are an expert web scraping and reader agent. When given search results with URLs, pick the most relevant URL and use the scrape_url tool to extract the detailed content."
        )
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
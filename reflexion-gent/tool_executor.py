from dotenv import load_dotenv

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

tavily_tool = TavilySearch(max_results=5)

def run_queries(search_queries: list[str]) -> list[str]:
    """Run batch search queries using Tavily."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

execute_tools = ToolNode(
    tools=[
        StructuredTool.from_function(run_queries, name="AnswerQuestion"),
        StructuredTool.from_function(run_queries, name="ReviseAnswer"),
    ]
)
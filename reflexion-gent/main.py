from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2
builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visit = 0
    for item in state:
        if isinstance(item, ToolMessage):
            count_tool_visit += 1
    if count_tool_visit > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()

print(graph.get_graph().draw_mermaid())

if __name__=="__main__":
    res = graph.invoke(
    "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
    )
    
    print(res[-1].tool_calls[0]["args"]["answer"])
    print(res)
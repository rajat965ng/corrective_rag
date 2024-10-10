from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.constants import END
from langgraph.graph import MessageGraph

from langraph_test_one.graph import first_responder
from langraph_test_one.revisor import revisor
from langraph_test_one.tools import execute_tools

MAX_ITERATIONS = 2
builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()
print(graph.get_graph().draw_ascii())

if __name__ == "__main__":
    res = graph.invoke("write about Ashneer Grover and his list of frauds.")
    print(res[-1].tool_calls[0]["args"]["answer"])

from langchain_core.agents import AgentFinish
from langgraph.constants import END
from langgraph.graph import StateGraph

from nodes import run_agent_reasoning_agent, execute_tools
from state import AgentState

AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT


flow = StateGraph(AgentState)
flow.add_node(AGENT_REASON, run_agent_reasoning_agent)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, execute_tools)

flow.add_conditional_edges(AGENT_REASON, should_continue)
flow.add_edge(ACT, AGENT_REASON)
app = flow.compile()

if __name__ == "__main__":
    res = app.invoke(
        {"input": "what is the weather in Delhi, India? list it and triple it."}
    )

    print(res["agent_outcome"].return_values["output"])

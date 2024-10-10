import operator
from typing import TypedDict, Union, Annotated, List, Tuple

from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], operator.add]

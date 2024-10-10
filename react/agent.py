from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

react_prompt: PromptTemplate = hub.pull("hwchase17/react")


@tool
def triple(num: float) -> float:
    """Use this to triple weather information."""
    return 3 * float(num)


tools = [TavilySearchResults(max_results=1), triple]
llm = ChatOpenAI()
react_agent_runnable = create_react_agent(llm, tools, react_prompt)

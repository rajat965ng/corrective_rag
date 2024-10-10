import json
from collections import defaultdict
from typing import List

from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from langraph_test_one.graph import parser
from langraph_test_one.schema import AnswerQuestion, Reflection

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tool_executor = ToolExecutor([tavily_tool])


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tools_calls = parser.invoke(tool_invocation)
    ids = []
    tool_invocations = []
    for parsed_call in parsed_tools_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="tavily_search_result_json",
                    tool_input=query,
                )
            )
            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations)
    output_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        output_map[id_][invocation.tool_input] = output

    tool_messages = []
    for id_, mapped_output in output_map.items():
        tool_messages.append(
            ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_)
        )

    return tool_messages


if __name__ == "__main__":
    message = HumanMessage(
        content="Write about AI powered SOC / autonomous soc problem domain,"
        "list startups that do that and raised capital"
    )
    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_andMandKaTola",
    )
    raw_res = execute_tools(
        state=[
            message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_andMandKaTola",
                    }
                ],
            ),
        ]
    )
    print(raw_res)

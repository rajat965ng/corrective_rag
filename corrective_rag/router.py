from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from corrective_rag.state import QueryRouter, GraphState

llm = ChatOpenAI(temperature=0)

query_router = llm.with_structured_output(QueryRouter)
query_router_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at routing a user question to a vectorstore or websearch.
            The vectorstore contains documents related to java interview questions and cooking recipes.
            Use vectorstore for questions on those topics. For anything else use websearch.""",
        ),
        ("human", "User question is {question}"),
    ]
)
query_router_prompt_chain = query_router_prompt_template | query_router


def query_router_conditional_edge(state: GraphState) -> str:
    print("Invoked::::query_router_conditional_edge")
    question = state["question"]
    query_router_result = query_router_prompt_chain.invoke(input={"question": question})
    destination = query_router_result.destination
    print("Query router destination is {}".format(destination))
    return destination

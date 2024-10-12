from typing import Dict, Any

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from corrective_rag.state import GraphState

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(temperature=0)

generated_prompt_chain = prompt | llm | StrOutputParser()


def generate_node(state: GraphState) -> Dict[str, Any]:
    print("Invoked::::generate_node")
    question = state["question"]
    documents = state["documents"]

    generation = generated_prompt_chain.invoke(
        {"context": documents, "question": question}
    )
    return {"documents": documents, "question": question, "generation": generation}

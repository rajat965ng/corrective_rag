from typing import Dict, Any

from ingestion import retriever
from corrective_rag.state import GraphState


def retriever_node(state: GraphState) -> Dict[str, Any]:
    print("Invoked::::retriever_node")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"question": question, "documents": documents}

from typing import Dict, Any

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.documents import Document

from corrective_rag.state import GraphState


def websearch_node(state: GraphState) -> Dict[str, Any]:
    print("Invoked::::websearch_node")
    question = state["question"]

    search = TavilySearchAPIWrapper()
    result = search.results(query=question, max_results=3)
    documents = []
    [
        documents.append(
            Document(page_content=doc["content"], metadata={"source": doc["url"]})
        )
        for doc in result
    ]
    return {"question": question, "documents": documents}

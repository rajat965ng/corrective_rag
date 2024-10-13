from langgraph.constants import END
from langgraph.graph import StateGraph

from corrective_rag.constant import GRADE_DOCS
from corrective_rag.constant import (
    RETRIEVER,
    GENERATE,
    WEBSEARCH,
    NOT_SUPPORTED,
    NOT_USEFUL,
    USEFUL,
    VECTORSTORE,
)
from corrective_rag.generate import generate_node
from corrective_rag.grading import (
    grading_node,
    grade_conditional_node,
    grade_generation_grounded_in_documents_and_question,
)
from corrective_rag.retriever import retriever_node
from corrective_rag.router import query_router_conditional_edge
from corrective_rag.state import GraphState
from corrective_rag.websearch import websearch_node

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVER, retriever_node)
workflow.add_node(GRADE_DOCS, grading_node)
workflow.add_node(GENERATE, generate_node)
workflow.add_node(WEBSEARCH, websearch_node)

workflow.set_conditional_entry_point(
    query_router_conditional_edge, {VECTORSTORE: RETRIEVER, WEBSEARCH: WEBSEARCH}
)

workflow.add_edge(RETRIEVER, GRADE_DOCS)
workflow.add_conditional_edges(
    GRADE_DOCS,
    grade_conditional_node,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {NOT_SUPPORTED: GENERATE, NOT_USEFUL: WEBSEARCH, USEFUL: END},
)

graph = workflow.compile()
graph.get_graph().draw_png(output_file_path="graph.jpg")

if __name__ == "__main__":
    res = graph.invoke(input={"question": "how to calculate XIRR in mutual funds?"})
    print(res)

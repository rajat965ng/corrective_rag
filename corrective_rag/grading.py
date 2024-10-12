from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from corrective_rag.constant import WEBSEARCH, GENERATE
from corrective_rag.state import GraphState, Grade

system = (
    "As a professional grading assistant find out whether the retrieved documents are relevant to the question "
    "asked."
)
llm = ChatOpenAI(temperature=0)


def grading_node(state: GraphState) -> Dict[str, Any]:
    print("Invoked::::grading_node")
    filter_docs = []
    websearch = False

    grade_template = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "user question: {question}, retrieved document: {document}"),
        ]
    )
    grade = llm.with_structured_output(Grade)
    template = grade_template | grade
    question = state["question"]
    documents = state["documents"]

    for doc in documents:
        result = template.invoke({"question": question, "document": doc.page_content})
        if result.bool_score == "yes":
            filter_docs.append(doc)

    if len(filter_docs) < 2:
        websearch = True

    return {"question": question, "documents": filter_docs, "websearch": websearch}


def grade_conditional_node(state: GraphState) -> str:
    print("Invoked::::grade_conditional_node")
    websearch = state["websearch"]
    print("value of websearch flag is {}".format(websearch))
    if websearch:
        return WEBSEARCH
    else:
        return GENERATE

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from corrective_rag.constant import (
    WEBSEARCH,
    GENERATE,
    USEFUL,
    NOT_USEFUL,
    NOT_SUPPORTED,
)
from corrective_rag.state import (
    GraphState,
    Grade,
    HallucinationGrade,
    GenerationAnswerGrade,
)

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


hallucination_grader = llm.with_structured_output(HallucinationGrade)
hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional grader. Return 'yes' if the generation is grounded in documents else return 'no'. ",
        ),
        (
            "human",
            "set of facts: \n\n {documents} \n\n" "LLM generation: {generation}",
        ),
    ]
)

hallucination_grader_prompt_chain = hallucination_grader_prompt | hallucination_grader


generation_answer_grader = llm.with_structured_output(GenerationAnswerGrade)
generation_answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an answer addresses/resolves a question. \n Give a binary score 'yes' "
            "or 'no'. 'yes' means that the answer resolves the question.",
        ),
        (
            "human",
            "User question: \n\n {question} \n\n" "generation: {generation}",
        ),
    ]
)

generation_answer_grader_prompt_chain = (
    generation_answer_grader_prompt | generation_answer_grader
)


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("Invoked::::grade_generation_grounded_in_documents_and_question")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_result = hallucination_grader_prompt_chain.invoke(
        input={"generation": generation, "documents": documents}
    )
    print(
        "hallucination_result.bool_score is {}".format(hallucination_result.bool_score)
    )
    if hallucination_result.bool_score == "yes":
        generation_answer_grader_result = generation_answer_grader_prompt_chain.invoke(
            input={"generation": generation, "question": question}
        )
        print(
            "generation_answer_grader_result.bool_score is {}".format(
                generation_answer_grader_result.bool_score
            )
        )
        if generation_answer_grader_result.bool_score == "yes":
            return USEFUL
        else:
            return NOT_USEFUL
    else:
        return NOT_SUPPORTED

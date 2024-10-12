from typing import TypedDict, List

from pydantic.v1 import BaseModel, Field


class GraphState(TypedDict):
    question: str
    generation: str
    websearch: bool
    documents: List[str]


class Grade(BaseModel):
    bool_score: str = Field(
        description="Check whether the documents retrieved are relevant to the question asked."
        "If they are relevant then answer 'yes', else 'no'."
    )


class HallucinationGrade(BaseModel):
    bool_score: str = Field(
        description="Check if the generation is grounded into documents then return 'yes' else 'no'"
    )


class GenerationAnswerGrade(BaseModel):
    bool_score: str = Field(
        description="Check whether the generation retrieved are relevant to the question asked."
        "If they are relevant then answer 'yes', else 'no'."
    )

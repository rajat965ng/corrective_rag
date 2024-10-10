from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superflous")


class AnswerQuestion(BaseModel):
    """Answer the Question"""

    answer: str = Field(description="~250 words detailed answer to the question")
    reflection: Reflection = Field(description="your reflection on the initial answer")
    search_queries: List = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    references: List = Field(description="Citations motivating your updated answer")

import datetime

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schema import AnswerQuestion

llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """you are an expert researcher.
        current time: {time}
        1. {first_introduction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="message"),
        ("system", "answer the user's questions above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_introduction="Provide a detailed ~250 words answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write a java program to check if number is a happy number or not. What are the names of organizations that asks this question in their interviews?"
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    res = chain.invoke(input={"message": [human_message]})
    print(res)

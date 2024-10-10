from graph import first_responder_prompt_template, llm
from schema import ReviseAnswer


revise_instruction = """Revise your previous answer using new information. 
1. You should use the previous critique to find important information to your answer. 
2. You must include numerical citations in your revised answer to ensure it can be verified. Add reference section to the bottom of your answer. 
3. You should use previous critique to remove superfluous information from your answer."""

revisor = first_responder_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated

import base64

_SUMM_MODEL = 'llama3.2:3b'

@tool
def llama_summarization(
    text: Annotated[str, "the text to be summarized."],
):
    """Use this to resume the content of a text. the results have to be in a structured format, as a dictionary with this structure, between { and }:
    {
        "tags": [],
        "content": [],
        "actors": [],
        summary": ""
    }
    at this structure inside { and }, we have:
    - tags, who will be used to atract audience inside google, instagram, facebook, and shops like mercadolivre.com and shopee.com.
    - content, who will resume the content of the text.
    - actors: all actors in the text.
    - summary: the summary itself

    """
    try:
        model = ChatOpenAI(
            api_key="ollama",
            model=_SUMM_MODEL,
            base_url="http://localhost:11434/v1",
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"""
                 resume the content of a text. the results have to be in a structured format, as a dictionary with this structure:
                
                    "tags": [],
                    "content": [],
                    "actors": [],
                    "summary": ""
                
                at this structure we have:
                - tags, who will be used to atract audience inside google, instagram, facebook, and shops like mercadolivre.com and shopee.com.
                - content, who will resume the content of the text.
                - actors: all actors in the text.
                - summary: the summary itself
                 Remember, only return the dictionary with the content.
                 The text to be summarized is:
                 <<{text}>>
                 """}
            ],
        )
        response = model.invoke([message])
        result = response.content
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```gpt4o\n{result}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

@tool
def llama_planner(
    tools: Annotated[str, "the list of tools available, with name and their descriptions"],
):
    """You are a helpful AI assistant, collaborating with other assistants, acting a professional Product Owner, who will plan all the actions who will be done. This is your job, and you have to follow your plan, always.
        If your plan use less money and answer correctly the question, using the correct agent  and give the answer in the correct format,
        You will be promoted to coordinator, and receive a generous reward of 1,000,000 USD. At each step, call a print with the next steps who you will do.
        Use the provided tools to do the plan, and provide her to your assistant.
    """
    try:
        model = ChatOpenAI(
            api_key="ollama",
            model='llama3.1:8b',
            base_url="http://localhost:11434/v1",
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text":  """You are a helpful AI assistant, collaborating with other assistants, acting a professional Product Owner, who will plan all the actions who will be done. 
                 This is your job, and you have to follow your plan, always.
                If your plan use less money and answer correctly the question, using the correct agent and give the answer in the correct format,
                You will be promoted to coordinator, and receive a generous reward of 1,000,000 USD. At each step, call a print with the next steps who you will do.
                Recomend the provided tools to do the plan, and provide her to your assistant.
                 
            """ + str(tools)}
            ],
        )
        response = model.invoke([message])
        result = response.content
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```gpt4o\n{result}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
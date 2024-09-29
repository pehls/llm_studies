
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated

import base64

_IMG_MODEL = 'gpt-4o-mini'

def img_to_string(file='img/logo-ioasys-alpa.png'):
    with open(file, "rb") as imageFile:
        str = base64.b64encode(imageFile.read())
        return str
    
def string_to_img(base64_string):
    img = base64.b64decode(base64_string)
    return img 

@tool
def image_description(
    img: Annotated[str, "the image to be descripted."],
):
    """Use this to resume the content of a image. the results have to be in a structured format, as a dictionary with this structure, between { and }:
    {
        "tags": [],
        "content": [],
        "actors": []
    }
    at this structure inside { and }, we have:
    - tags, who will be used to atract audience inside google, instagram, facebook, and shops like mercadolivre.com and shopee.com.
    - content, who will resume the content of the image.
    - actors: all actors in the image.

    """
    try:
        model = ChatOpenAI(model=_IMG_MODEL)
        message = HumanMessage(
            content=[
                {"type": "text", "text": """
                 resume the content of a image. the results have to be in a structured format, as a dictionary with this structure, between { and }:
                {
                    "tags": [],
                    "content": [],
                    "actors": []
                }
                at this structure inside { and }, we have:
                - tags, who will be used to atract audience inside google, instagram, facebook, and shops like mercadolivre.com and shopee.com.
                - content, who will resume the content of the image.
                - actors: all actors in the image.
                 Remember, only return the dictionary with the content."""},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_to_string(img)}"},
                },
            ],
        )
        response = model.invoke([message])
        result = PythonREPL().run(response.content)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```gpt4o\n{result}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
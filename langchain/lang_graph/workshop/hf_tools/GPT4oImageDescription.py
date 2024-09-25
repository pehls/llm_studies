
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_experimental.utilities import PythonREPL
from transformers import Tool
import base64

def img_to_string(file='img/logo-ioasys-alpa.png'):
    with open(file, "rb") as imageFile:
        str = base64.b64encode(imageFile.read())
        return str
    
def string_to_img(base64_string):
    img = base64.b64decode(base64_string)
    return img 

class image_description(Tool):
    name = 'image_description'
    description = (
        """
        Use this to resume the content of a image. the results have to be in a structured format, as a dictionary with this structure, between { and }, for example:
        {
            "tags": [],
            "content": [],
            "actors": []
        }
        at this structure inside { and }, we have:
        - tags, who will be used to atract audience inside google, instagram, facebook, and shops like mercadolivre.com and shopee.com.
        - content, who will resume the content of the image.
        - actors: all actors in the image.
        always remember to install every python package who will be used with '!pip install <package name>"
        Remember, only return the dictionary with the content."""
    )
    task = 'Use this to resume the content of a image'
    repo_id = None
    inputs = ["img"]
    outputs = ["result"]
    def __call__():
        try:
            model = ChatOpenAI(model='gpt-4o-mini')
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
            return result
        except Exception as exc:
            return "Error - "+str(exc)

from langchain_experimental.utilities import PythonREPL
from transformers import Tool

class python_repl(Tool):
    name = 'python_repl'
    description = (
        """
        Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user. 
        always remember to install every python package who will be used with '!pip install <package name>"
        It takes a python code, a string, and execute it using PythonREPL"""
    )
    task='Use this to execute python code'
    repo_id = None
    inputs = ["code"]
    outputs = ["result"]
    def __call__(self, code:str):
        try:
            result = PythonREPL().run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return (
            result
        )
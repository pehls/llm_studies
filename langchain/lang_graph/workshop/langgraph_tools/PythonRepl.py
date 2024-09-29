from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to save your files, or every python interaction."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. 
    always remember to install every python package who will be used with '!pip install <package name>'"""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
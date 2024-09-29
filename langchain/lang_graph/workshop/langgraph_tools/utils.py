from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    AIMessage,
)
from langchain_experimental.plan_and_execute import agent_executor
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START

from typing import Annotated, Sequence, TypedDict
import operator
import functools

from langgraph_tools.GPT4oImageDescription import *
from langgraph_tools.PythonRepl import *
from langgraph_tools.Similarities import *
from langgraph_tools.LLamaSummarization import *

_ROUTER_MODEL = 'gpt-4o-mini'

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " will be think step by step, and set a plan to be executed, and follow him. "
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop, and include a python dictionary with comparations done with the tools, the news who you search, and named metrics."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join(['\n name: ' + tool.name + ' description: ' + tool.description + '\n' for tool in tools]))
    return prompt | llm.bind_tools(tools)

## Define State
## We first define the state of the graph. This will just a list of messages, along with a key to track the most recent sender


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }
# Helper to create workflow
def create_workflow(
        llm = ChatOpenAI(model=_ROUTER_MODEL),
        tools = [
            TavilySearchResults(max_results=5), 
            python_repl, 
            cosine_similarity, 
            levenshtein_similarity, 
            jaccard_similarity, 
            llama_summarization,
            llama_planner
            ],
        system_message = ""
        ):
    # Research agent and node
    research_agent = create_agent(
        llm,
        tools,
        system_message=system_message,
    )
    
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
    ## Define Tool Node
    # node to run the tools

    tool_node = ToolNode(tools)
    ## Define the Graph

    workflow = StateGraph(AgentState)

    workflow.add_node("Researcher", research_node)
    workflow.add_node("tools", tool_node)

    workflow.add_conditional_edges(
        "tools",
        # Each agent node updates the 'sender' field
        # the tool calling node does not, meaning
        # this edge will route back to the original agent
        # who invoked the tool
        lambda x: x["sender"],
        {
            "Researcher": "Researcher",
            
        },
    )

    workflow.add_conditional_edges(
        "Researcher",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    workflow.add_edge(START, "Researcher")
    graph = workflow.compile()
    return graph
from dotenv import load_dotenv
import os
from langchain.globals import set_debug

from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description

from typing import Any, Dict, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnablePassthrough

set_debug(False)

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: Dict[str, Any]


def invoke_tool(
    tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None
):
    """A function that we can use the perform a tool invocation.

    Args:
        tool_call_request: a dict that contains the keys name and arguments.
            The name must match the name of a tool that exists.
            The arguments are the arguments to that tool.
        config: This is configuration information that LangChain uses that contains
            things like callbacks, metadata, etc.See LCEL documentation about RunnableConfig.

    Returns:
        output from the requested tool
    """
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)


def parse_output(output: Dict) -> Dict:

    role = "assistant"
    text = output["output"]["content"]
    url = output["output"]["url"]

    content = f"{text} \n You can find more information at: {url}"

    return {"role": role, "content": content}


def chatbot(state: State):
    return {"messages": [chain.invoke(state["messages"])]}


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1]["content"])


# Tools - Tavily Search
@tool
def TavilyRunQuery(query: str) -> str:
    """
    This function will run a query in TavilySearch and return the search result.
    """
    tool = TavilySearchResults(max_results=3)

    return tool.invoke(query)[0]


tools = [TavilyRunQuery]

rendered_tools = render_text_description(tools)

system_prompt = f"""
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

The `arguments` should be a dictionary, with keys corresponding to the argument names and the values corresponding to the requested values.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

llm = ChatOllama(
    model="SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M",
    api_key="ollama",
    base_url="http://127.0.0.1:11434",
)

chain = (
    prompt
    | llm
    | JsonOutputParser()
    | RunnablePassthrough.assign(output=invoke_tool)
    | parse_output
)

# LangGraph

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()


# Run Bielik Chatbot with Tavily Search with LangGraph

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

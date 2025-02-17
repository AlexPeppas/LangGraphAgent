import envUtils
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import TavilySearchResults
from IPython.display import Image, display
from pprint import pformat

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import interrupt, Command, Interrupt

from typing import Literal

from agentTools import BasicToolNode # this is a replica ToolNode of mine for experimentation. Use the prebuilt below.
from langgraph.prebuilt import ToolNode 

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables import chain    
from promptBuilder import PromptBuilder

from pydantic import BaseModel

envUtils.init_env()

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    ask_human: bool
    approved: bool

def is_critical_tool(ai_message: object) -> bool:
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "human_assistance_tool":
            return True
        
    return False

def route_tools(state : State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if (is_critical_tool(ai_message)):
            return "critical_tools"
        
        return "tools" # if found tools then route to tools node
    return END # otherwise end the graph traversal

graph_builder = StateGraph(State)

@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance_tool(is_approved: bool, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Request assistance from human."""
    # If the information is correct, update the state as-is.
    if is_approved:
        response = "User approved the request."
    else:
        response = "User rejected the request."
     # Otherwise, receive information from the human reviewer.
    
    state_update = {
        "approved": is_approved,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)

search_tool = TavilySearchResults(
    max_results = 2,
    search_depth = "advanced",
    include_answer = True,
    include_raw_content = True,
    include_images = True)

tools = [search_tool]
critical_tools = [human_assistance_tool]

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.3)
llm_with_tools = llm.bind_tools(tools= [search_tool, human_assistance_tool])

def llm_invoke(state: State):
    #messages = state.get("messages", [])
    #last_user_input = messages[-1].content
    #prompt = PromptBuilder(input= last_user_input, history= messages)
    #chain = llm_with_tools | prompt
    #output = {"messages": [chain.invoke(state["messages"])]}
    bot_response = llm_with_tools.invoke(state["messages"])
    assert(len(bot_response.tool_calls) <= 1)
    return {"messages": [bot_response]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("llm", llm_invoke)

tool_node = ToolNode(tools = tools)
graph_builder.add_node("tools", tool_node)

critical_tool_node = ToolNode(tools = critical_tools)
graph_builder.add_node("critical_tools", critical_tool_node)

graph_builder.add_conditional_edges(
    "llm",
    route_tools,
    {"tools" : "tools", END: END, "critical_tools" : "critical_tools"}
)

graph_builder.add_edge("critical_tools", "llm")
graph_builder.add_edge("tools", "llm")
graph_builder.add_edge(START, "llm")


memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory, interrupt_before= ["critical_tools"])

try:
    display(Image(graph.get_graph().draw_png()))
except Exception:
    pass

def stream_graph_updates(user_input: str, config: RunnableConfig):
    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode= "values"):
        for value in event.values():
            if isinstance(value, dict) and "messages" in value:
                # Access the content safely
                last_message = value["messages"][-1]
                if hasattr(last_message, "content"):
                    "Assistant:", last_message.pretty_print()
                else:
                    print("Assistant: Message has no content.")
            #else:
                # Log unexpected structures for debugging
                #print(f"Unexpected value structure: {value}")
    
    state = graph.get_state(config)
    if ('critical_tools' in state.next):
        # continue execution, it has been interrupted for HITL
        human_response = input("Do you approve the following request? ")
        if (human_response):
            for event in graph.stream(None, config, stream_mode= "values"):
                print (event)
        else:
            print("Rejected")

threadId = input("ThreadId: ")
while True:
    #try:
        config = {"configurable" : {"thread_id": f"{threadId}"}}

        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if user_input.lower() in ["checkpoint"]:
            snapshot = graph.get_state(config)
            print(f"snapshot: {snapshot}")
            continue
        if user_input.lower() in ["next"]:
            snapshot = graph.get_state(config).next
            print(f"snapshot: {snapshot}")
            continue

        stream_graph_updates(user_input, config)
    # except Exception as ex:
    #     print(f"Caught an exception: {ex}")
    #     raise
import envUtils
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import TavilySearchResults
# from IPython.display import Image, display


from typing import Literal
from agentTools import BasicToolNode

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables import chain    
from promptBuilder import PromptBuilder

envUtils.init_env()

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def route_tools(state : State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools" # if found tools then route to tools node
    return END # otherwise end the graph traversal

graph_builder = StateGraph(State)

searchTool = TavilySearchResults(
    max_results = 5,
    search_depth = "advanced",
    include_answer = True,
    include_raw_content = True,
    include_images = True)

tools = [searchTool]

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.5)
llm_with_tools = llm.bind_tools(tools)

def llm_invoke(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("llm", llm_invoke)

tool_node = BasicToolNode(tools = tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "llm",
    route_tools,
    {"tools" : "tools", END: END}
)

graph_builder.add_edge("tools", "llm")
graph_builder.add_edge(START, "llm")

# not needed, the conditional edge tools-> llm is responsible to end that
#graph_builder.add_edge("llm", END)

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass

def stream_graph_updates(user_input: str, config: RunnableConfig):
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content) #.pretty_print()

threadId = input("ThreadId: ")
config = {"configurable" : {"thread_id": f"{threadId}"}}
while True:
    try:
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
    except:
        break
import sys
import types
import json

# Create a dummy langchain_core.messages module with ToolMessage
messages_mod = types.ModuleType("langchain_core.messages")
class ToolMessage:
    def __init__(self, content: str, name: str, tool_call_id: str):
        self.content = content
        self.name = name
        self.tool_call_id = tool_call_id
messages_mod.ToolMessage = ToolMessage
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules["langchain_core.messages"] = messages_mod

from agentTools import BasicToolNode

class DummyTool:
    def __init__(self, name, return_value):
        self.name = name
        self.return_value = return_value

    def invoke(self, args):
        return self.return_value

class DummyMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def test_basic_tool_node_returns_tool_message():
    tool = DummyTool("dummy", {"result": 42})
    node = BasicToolNode([tool])

    message = DummyMessage([
        {"name": "dummy", "args": {}, "id": "1"}
    ])

    result = node({"messages": [message]})

    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.name == "dummy"
    assert json.loads(msg.content) == {"result": 42}

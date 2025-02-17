from langchain_core.messages import ToolMessage
import json

class BasicToolNode:

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {}
        for tool in tools:
            if hasattr(tool, "name"):
                key = tool.name
            elif hasattr(tool, "__name__"):
                key = tool.__name__
            else:
                raise AttributeError(f"Tool {tool} must have either 'name' or '__name__' attribute.")
            
            self.tools_by_name[key] = tool

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            messages = messages[-1]
        else:
            raise ValueError("Messages not found in Inputs")
        outputs = []
        for tool_call in messages.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name = tool_call["name"],
                    tool_call_id = tool_call["id"]
                )
            )
        return {"messages": outputs}
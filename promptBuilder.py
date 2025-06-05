from langchain_core.prompts import ChatPromptTemplate
import datetime


class PromptBuilder:
    """Helper class to assemble a chat prompt."""

    def __init__(self, system_prompt: str, user_input: str, history: list) -> None:
        """Create a prompt template from the given components."""
        today = datetime.datetime.today().strftime("%D")

        if not (system_prompt := (system_prompt or "").strip()):
            system_prompt = (
                "You are a helpful assistant. Today the day is "
                f"{today}. Your goal is to prepare and break down a cohesive plan "
                "with ultimate goal to help the user fulfil a request. If you can "
                "respond to the user without using a given tool go ahead. Otherwise, "
                "invoke the required tool to fulfill your goal. Always reply politely "
                "and with humor."
            )

        self.prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("human", user_input),
                ("placeholder", history),
            ]
        )

    def build(self) -> ChatPromptTemplate:
        """Return the constructed prompt template."""
        return self.prompt

from langchain_core.prompts import ChatPromptTemplate
import datetime

class PromptBuilder:

    def __init__(self, systemPrompt: str, input: str, history: list):
        
        today = datetime.datetime.today().strftime("%D")

        if not (systemPrompt := systemPrompt.strip()):
            systemPrompt = f"You are a helpful assistant. Today the day is {today}. Your goal is to prepare and break down a cohesive plan with ultimate goal to help the user fulfil a request.
                 if you can respond to the user without using a given tool go ahead. Otherwise, invoke the required tool to fulfill your goal. Always reply politely and with humor."
            
        return ChatPromptTemplate(
            [
                ("system", ),
                ("user", input),
                ("placeholder", history)
            ]
        )

import asyncio
from fastapi import HTTPException
from utils.logger import logger

from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import initialize_agent

class LLMService:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0.7,
            model_name="gpt-4-0125-preview",
            streaming=False,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are an advanced AI assistant with multi-step reasoning. "
                    "When additional context is missing, ask clarifying questions before answering. "
                    "Use provided tools to fetch external data if needed."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                SystemMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.agent_executor = initialize_agent(
            agent="chat-conversational-react-description",
            tools=[],
            llm=self.llm,
            verbose=True,
            memory=None,
            return_intermediate_steps=True,
        )

    async def generate_response(self, input_text: str, chat_history) -> str:
        agent_input = {"input": input_text, "chat_history": chat_history}
        logger.info("Invoking LLM agent for response generation.")
        try:
            if hasattr(self.agent_executor, "invoke_async"):
                result = await self.agent_executor.invoke_async(agent_input)
            else:
                result = await asyncio.to_thread(self.agent_executor.invoke, agent_input)
            if isinstance(result, dict) and "output" in result:
                return result["output"]
            return result
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="LLM execution error.")

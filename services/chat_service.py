import asyncio
from utils.logger import logger

class ChatService:
    def __init__(self, llm_service, vector_db, external_search):
        self.llm_service = llm_service
        self.vector_db = vector_db
        self.external_search = external_search

    async def process_chat(self, user_id: str, message: str) -> str:
        chat_history = await self.vector_db.retrieve_context(user_id, message)
        answer = await self.llm_service.generate_response(message, chat_history)
        if len(answer.split()) < 20:
            logger.info("Answer is too short; attempting to augment with external search data.")
            search_query = f"{message} additional context"
            search_result = await self.external_search.search(search_query)
            revised_prompt = (
                f"User asked: '{message}'.\n"
                f"Original answer: '{answer}'.\n"
                f"Additional data: '{search_result}'.\n"
                f"Provide a detailed response."
            )
            answer = await self.llm_service.generate_response(revised_prompt, chat_history)

        await asyncio.gather(
            self.vector_db.store_conversation(user_id, message, "user"),
            self.vector_db.store_conversation(user_id, answer, "assistant")
        )
        return answer

from fastapi import APIRouter, HTTPException
from models.chat import ChatRequest, ChatResponse
from utils.logger import logger

from config.config import Config
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
from services.llm_service import LLMService
from services.external_search_service import ExternalSearchService
from services.chat_service import ChatService

router = APIRouter()

# Initialize service instances.
embedding_service = EmbeddingService(Config.OPENAI_API_KEY)
vector_db_service = VectorDBService(
    api_key=Config.PINECONE_API_KEY,
    env=Config.PINECONE_ENV,
    index_name=Config.PINECONE_INDEX_NAME,
    embedding_service=embedding_service,
)
llm_service = LLMService(Config.OPENAI_API_KEY)
external_search_service = ExternalSearchService(Config.GOOGLE_API_KEY, Config.GOOGLE_CSE_ID)
chat_service_instance = ChatService(llm_service, vector_db_service, external_search_service)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    logger.info(f"Received chat request from user '{payload.user_id}'.")
    try:
        response_text = await chat_service_instance.process_chat(payload.user_id, payload.message.strip())
        logger.info("Returning final chat response.")
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat processing error.")

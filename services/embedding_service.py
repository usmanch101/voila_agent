import asyncio
from typing import Dict, List

from langchain_openai import OpenAIEmbeddings
from utils.logger import logger

class EmbeddingService:
    def __init__(self, openai_api_key: str):
        self.embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.cache: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()

    async def get_embedding(self, text: str) -> List[float]:
        async with self.lock:
            if text in self.cache:
                logger.debug("Using cached embedding for text.")
                return self.cache[text]
        logger.debug("Creating new embedding for text.")
        emb = await asyncio.to_thread(self.embeddings_model.embed_query, text)
        async with self.lock:
            self.cache[text] = emb
        return emb

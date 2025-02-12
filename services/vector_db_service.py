import asyncio
import uuid
from typing import List, Dict

from pinecone import Pinecone, ServerlessSpec, Index
from utils.logger import logger
from langchain.schema import SystemMessage

class VectorDBService:
    def __init__(self, api_key: str, env: str, index_name: str, embedding_service):
        self.embedding_service = embedding_service
        self.api_key = api_key
        self.env = env
        self.index_name = index_name

        self.client = Pinecone(api_key=api_key)
        existing_indexes = [index["name"] for index in self.client.list_indexes()]
        if index_name not in existing_indexes:
            logger.info(f"Index '{index_name}' not found. Creating new index.")
            self.client.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=env),
            )
        index_description = self.client.describe_index(index_name)
        index_host = index_description["host"]
        self.index = Index(host=index_host, api_key=api_key)

    async def store_conversation(self, user_id: str, text: str, speaker: str) -> None:
        logger.debug(f"Storing conversation for user '{user_id}' from '{speaker}'.")
        embedding = await self.embedding_service.get_embedding(text)
        chunk_id = f"{user_id}-{speaker}-{uuid.uuid4()}"
        metadata = {"speaker": speaker, "text": text}
        await asyncio.to_thread(
            self.index.upsert,
            vectors=[(chunk_id, embedding)],
            namespace=user_id,
            metadata=metadata,
        )
        logger.info("Conversation stored in vector database.")

    async def retrieve_context(self, user_id: str, query: str, top_k: int = 3) -> List[SystemMessage]:
        logger.debug(f"Retrieving context for user '{user_id}' with query: {query}")
        embedding = await self.embedding_service.get_embedding(query)
        results = await asyncio.to_thread(
            self.index.query,
            vector=embedding,
            top_k=top_k,
            namespace=user_id,
            include_metadata=True,
        )
        messages = []
        if results.get("matches"):
            logger.debug(f"Found {len(results['matches'])} matching contexts.")
            for match in results["matches"]:
                meta = match.get("metadata", {}) or {}
                speaker = meta.get("speaker", "unknown")
                text = meta.get("text", "")
                messages.append(SystemMessage(content=f"[{speaker}]: {text}"))
        else:
            logger.debug("No matching context found.")
        return messages

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-chat-index")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY")
        if not cls.PINECONE_API_KEY:
            raise ValueError("Missing PINECONE_API_KEY")
        if not cls.PINECONE_ENV:
            raise ValueError("Missing PINECONE_ENV")
        if not cls.GOOGLE_API_KEY:
            raise ValueError("Missing GOOGLE_API_KEY")
        if not cls.GOOGLE_CSE_ID:
            raise ValueError("Missing GOOGLE_CSE_ID")

# Validate required environment variables on import.
Config.validate()

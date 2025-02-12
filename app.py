# app.py
from fastapi import FastAPI
from routes import chat  
from utils.logger import logger

app = FastAPI(title="Agentic AI Chatbot")

app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")

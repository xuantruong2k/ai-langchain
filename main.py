from fastapi import FastAPI
import uvicorn

from services.llm_service import InternalLlmService
from messsages.inbound_http_request import ChatRequest

app = FastAPI(title="FastAPI")

llm_service: InternalLlmService = any

@app.on_event("startup")
async def startup():
    global llm_service
    llm_service = InternalLlmService()

# async def lifespan(app: FastAPI):
#     # load LLM server
#     global llm_service
#     llm_service = InternalLlmService()
#     yield
#     # end
    
# app = FastAPI(lifespan=lifespan)


@app.post("/langchain/conversation/chat", response_model={})
async def lc_conversation_chat(chat_request: ChatRequest):
    return await llm_service.lc_chat(request=chat_request)

if __name__ == '__main__':
    uvicorn.run(f'main:app', host='localhost', port=8000)

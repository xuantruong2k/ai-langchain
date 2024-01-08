
from fastapi import FastApi

from services.llm_service import InternalLlmService

llm_service: InternalLlmService

async def lifespan(app: FastApi):
    # load LLM server
    global llm_service
    llm_service = InternalLlmService()
    yield
    # end

app = FastApi(lifespan=lifespan)

@app.post("/langchain/conversation/chat", response_model={})
async def lc_conversation_chat(chat_request: ChatRequest):
    return await llm_service.lc_chat(request=chat_request)

if __name__ == '__main__':
    uvicorn.run(f'main:app', host='localhost', port=8000)


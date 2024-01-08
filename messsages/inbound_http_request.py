from pydantic import BaseModel

class ChatRequest(BaseModel):
    promt: str
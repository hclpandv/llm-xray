from pydantic import BaseModel


class TokenInfo(BaseModel):
    position: int
    token: str
    token_id: int


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 32


class GenerationResponse(BaseModel):
    model: str
    prompt: str
    tokens: list[TokenInfo]
    generated_text: str
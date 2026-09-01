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


class TensorInfo(BaseModel):
    name: str
    shape: list[int]


class ModelInfo(BaseModel):
    name: str
    layers: int
    hidden_size: int
    vocab_size: int


class NextTokenInfo(BaseModel):
    token: str
    token_id: int
    probability: float


class InspectResponse(BaseModel):
    model: ModelInfo
    prompt: str
    tokens: list[TokenInfo]

    embedding: TensorInfo
    layers: list[dict]

    final_hidden_state: TensorInfo
    logits: TensorInfo

    next_tokens: list[NextTokenInfo]

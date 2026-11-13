from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    created_at: datetime
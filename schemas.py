"""
Database Schemas for TTS App

Each Pydantic model corresponds to one MongoDB collection (lowercased name).
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

# Users
class User(BaseModel):
    email: EmailStr
    password_hash: str
    name: Optional[str] = None
    role: Literal["user", "admin"] = "user"
    credits: int = 0
    provider_ids: Optional[dict] = None  # {"google": str, "apple": str}
    created_at: Optional[datetime] = None

# Voices
class Voice(BaseModel):
    key: str = Field(..., description="Internal voice key for provider")
    name: str
    language: str
    accent: Optional[str] = None
    gender: Optional[Literal["male", "female", "neutral"]] = None
    tags: List[str] = []
    provider: str = "builtin"
    preview_text: str = "Hello, this is a sample voice preview."
    is_active: bool = True

# Audio Files generated
class AudioFile(BaseModel):
    user_id: str
    project_id: Optional[str] = None
    voice_key: str
    input_text: str
    format: Literal["mp3", "wav"] = "wav"
    duration_ms: Optional[int] = None
    bytes_size: Optional[int] = None
    status: Literal["pending", "processing", "completed", "failed"] = "completed"
    url: Optional[str] = None  # signed URL or API path

# Projects
class Project(BaseModel):
    user_id: str
    name: str
    description: Optional[str] = None
    favorite_voice_keys: List[str] = []

# Usage events
class Usage(BaseModel):
    user_id: str
    type: Literal["preview", "synthesis", "upload", "billing"]
    units: int = 0  # characters or seconds
    meta: Optional[dict] = None

# API Keys (for admin-issued server-to-server)
class ApiKey(BaseModel):
    name: str
    key_hash: str
    revoked: bool = False

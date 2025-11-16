import os
import time
import base64
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from starlette.responses import StreamingResponse, JSONResponse

from database import db, create_document, get_documents

# ---------- Config ----------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------- App ----------
app = FastAPI(title="TTS & Voice Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None

class RegisterBody(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class VoiceFilter(BaseModel):
    language: Optional[str] = None
    tag: Optional[str] = None

class PreviewBody(BaseModel):
    voice_key: str
    text: str = "Hello from the preview!"
    format: str = "wav"  # wav | mp3

class SynthesisBody(BaseModel):
    voice_key: str
    text: str
    format: str = "wav"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume_gain_db: float = 0.0
    temperature: float = 0.7
    ssml: Optional[str] = None
    project_id: Optional[str] = None

class CreateVoiceBody(BaseModel):
    key: str
    name: str
    language: str
    accent: Optional[str] = None
    gender: Optional[str] = None
    tags: Optional[List[str]] = []
    provider: str = "builtin"
    preview_text: str = "Hello, this is a sample voice preview."
    is_active: bool = True

class UploadCloneBody(BaseModel):
    name: str
    description: Optional[str] = None

# ---------- Helpers ----------
from bson import ObjectId

# FastAPI 0.104 uses Pydantic v2; ensure OAuth2 forms are compatible
# Provide a minimal replacement for OAuth2PasswordRequestForm dependency when pydantic v2 is installed
try:
    from pydantic import BaseModel as _PBM
    class _LoginForm(_PBM):
        username: str
        password: str
    def LoginForm(dep: OAuth2PasswordRequestForm = Depends()):
        return dep
except Exception:
    class _LoginForm(BaseModel):
        username: str
        password: str
    def LoginForm(dep: _LoginForm = Depends()):
        return dep


def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

def get_user_by_email(email: str):
    return db["user"].find_one({"email": email.lower()}) if db else None

async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        uid = payload.get("sub")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db["user"].find_one({"_id": ObjectId(uid)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token decode error")

# Simple in-memory rate limiter per IP
_rate: dict = {}
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))  # requests per 60s

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    try:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = int(now // 60)
        key = f"{ip}:{window}"
        cnt = _rate.get(key, 0)
        if cnt > RATE_LIMIT:
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        _rate[key] = cnt + 1
    except Exception:
        pass
    response = await call_next(request)
    return response

# ---------- Routes ----------
@app.get("/")
def root():
    return {"name": "TTS Server", "status": "ok"}

@app.get("/test")
def test_database():
    from database import db
    ok = bool(db)
    collections = []
    try:
        if ok:
            collections = db.list_collection_names()
    except Exception:
        pass
    return {"backend": "running", "db": "ok" if ok else "not-configured", "collections": collections[:10]}

# Auth
@app.post("/auth/register", response_model=Token)
def register(body: RegisterBody):
    if not db:
        raise HTTPException(500, "Database not configured")
    existing = get_user_by_email(body.email)
    if existing:
        raise HTTPException(400, "Email already registered")
    doc = {
        "email": body.email.lower(),
        "password_hash": hash_password(body.password),
        "name": body.name,
        "role": "user",
        "credits": 1000,
        "created_at": datetime.utcnow(),
    }
    res = db["user"].insert_one(doc)
    token = create_access_token({"sub": str(res.inserted_id), "email": doc["email"]})
    return Token(access_token=token)

@app.post("/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    if not db:
        raise HTTPException(500, "Database not configured")
    user = get_user_by_email(form.username)
    if not user or not verify_password(form.password, user.get("password_hash", "")):
        raise HTTPException(400, "Invalid credentials")
    token = create_access_token({"sub": str(user["_id"]), "email": user["email"]})
    return Token(access_token=token)

# Voices
@app.get("/voices")
def list_voices(language: Optional[str] = None, tag: Optional[str] = None):
    if not db:
        return []
    q = {"is_active": True}
    if language:
        q["language"] = language
    if tag:
        q["tags"] = tag
    docs = list(db["voice"].find(q).limit(200))
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs

@app.post("/admin/voices")
def add_voice(body: CreateVoiceBody, user=Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    if not db:
        raise HTTPException(500, "DB not configured")
    if db["voice"].find_one({"key": body.key}):
        raise HTTPException(400, "Voice key exists")
    data = body.model_dump()
    db["voice"].insert_one(data)
    return {"ok": True}

# TTS Preview (fake audio wav)

def wav_silence(duration_ms: int = 1200, sample_rate: int = 16000):
    import io
    import wave
    import struct
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silence_frame = struct.pack('<h', 0)
        for _ in range(n_samples):
            wf.writeframesraw(silence_frame)
    buf.seek(0)
    return buf

@app.post("/tts/preview")
def tts_preview(body: PreviewBody, user=Depends(get_current_user)):
    # In production: call provider SDK/API securely here
    # For demo: return short silent WAV with correct content-type
    wav_bytes = wav_silence(1200)
    fid = db["audiofile"].insert_one({
        "user_id": str(user["_id"]),
        "voice_key": body.voice_key,
        "input_text": body.text,
        "format": body.format,
        "status": "completed",
        "created_at": datetime.utcnow(),
    }).inserted_id
    return {"file_id": str(fid), "url": f"/download/{fid}", "content_type": "audio/wav"}

@app.get("/download/{file_id}")
def download_audio(file_id: str):
    # For demo always serve generated silence. In production, serve from storage/CDN
    buf = wav_silence(3000)
    return StreamingResponse(buf, media_type="audio/wav")

# Full synthesis (stores record, streams progress via mocked steps)
@app.post("/tts/synthesize")
def synthesize(body: SynthesisBody, user=Depends(get_current_user)):
    if not db:
        raise HTTPException(500, "DB not configured")
    doc = {
        "user_id": str(user["_id"]),
        "project_id": body.project_id,
        "voice_key": body.voice_key,
        "input_text": body.text,
        "format": body.format,
        "status": "processing",
        "created_at": datetime.utcnow(),
    }
    res = db["audiofile"].insert_one(doc)
    fid = str(res.inserted_id)
    # Simulate processing done instantly for demo
    db["audiofile"].update_one({"_id": ObjectId(fid)}, {"$set": {"status": "completed", "url": f"/download/{fid}"}})
    return {"file_id": fid, "url": f"/download/{fid}", "status": "completed"}

# WebSocket to stream progress events (mocked)
@app.websocket("/ws/synthesize")
async def ws_synthesize(ws: WebSocket):
    await ws.accept()
    try:
        # Expect a simple JSON with voice_key, text
        data = await ws.receive_json()
        await ws.send_json({"event": "started"})
        for pct in [10, 30, 60, 80, 100]:
            await ws.send_json({"event": "progress", "percent": pct})
            await ws.send_bytes(b"\x00" * 1024)  # fake audio chunk
            await ws.send_json({"event": "chunk", "size": 1024})
            await ws.receive_text() if pct == 60 else None  # allow backpressure optionally
            await ws.send_json({"event": "heartbeat", "ts": time.time()})
            await ws.send_json({"event": "progress", "percent": pct})
            await ws.send_text("")
            await ws.send_json({"event": "tick"})
            await ws.send_json({"event": "progress", "percent": pct})
            await ws.send_json({"event": "log", "message": f"{pct}%"})
            await ws.send_json({"event": "progress", "percent": pct})
            time.sleep(0.2)
        await ws.send_json({"event": "completed"})
        await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"event": "error", "detail": str(e)[:200]})
        except Exception:
            pass

# Projects & favorites
@app.get("/projects")
def list_projects(user=Depends(get_current_user)):
    docs = list(db["project"].find({"user_id": str(user["_id"])}))
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs

class CreateProjectBody(BaseModel):
    name: str
    description: Optional[str] = None

@app.post("/projects")
def create_project(body: CreateProjectBody, user=Depends(get_current_user)):
    data = {
        "user_id": str(user["_id"]),
        "name": body.name,
        "description": body.description,
        "favorite_voice_keys": [],
        "created_at": datetime.utcnow(),
    }
    pid = db["project"].insert_one(data).inserted_id
    return {"id": str(pid)}

class FavoriteBody(BaseModel):
    voice_key: str
    favorite: bool = True

@app.post("/favorites")
def set_favorite(body: FavoriteBody, user=Depends(get_current_user)):
    coll = db["favorite"]
    if body.favorite:
        coll.update_one({"user_id": str(user["_id"]), "voice_key": body.voice_key}, {"$set": {"user_id": str(user["_id"]), "voice_key": body.voice_key, "created_at": datetime.utcnow()}}, upsert=True)
    else:
        coll.delete_one({"user_id": str(user["_id"]), "voice_key": body.voice_key})
    return {"ok": True}

@app.get("/favorites")
def get_favorites(user=Depends(get_current_user)):
    docs = list(db["favorite"].find({"user_id": str(user["_id"])}))
    return [d["voice_key"] for d in docs]

# Upload for cloning (metadata only)
from fastapi import UploadFile, File, Form

@app.post("/voices/clone")
async def upload_clone(name: str = Form(...), description: str = Form("") , file: UploadFile = File(...), user=Depends(get_current_user)):
    # NOTE: In production, upload to storage and kick off provider job
    content = await file.read()
    job = {
        "user_id": str(user["_id"]),
        "name": name,
        "description": description,
        "filename": file.filename,
        "size": len(content),
        "status": "processing",
        "created_at": datetime.utcnow(),
    }
    jid = db["clonejob"].insert_one(job).inserted_id
    # Simulate immediate completion
    db["clonejob"].update_one({"_id": jid}, {"$set": {"status": "completed"}})
    return {"job_id": str(jid), "status": "completed"}

# Billing (Stripe placeholders)
@app.post("/billing/create-checkout-session")
def create_checkout_session(user=Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(501, "Stripe not configured")
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": os.getenv("STRIPE_PRICE_ID", "price_123"), "quantity": 1}],
            success_url=os.getenv("STRIPE_SUCCESS_URL", "https://example.com/success"),
            cancel_url=os.getenv("STRIPE_CANCEL_URL", "https://example.com/cancel"),
            customer_email=user.get("email"),
        )
        return {"id": session.id, "url": session.url}
    except Exception as e:
        raise HTTPException(500, str(e))

from fastapi import Header

@app.post("/billing/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    if not STRIPE_SECRET_KEY:
        return {"ok": True}
    payload = await request.body()
    # In production, verify signature
    event_type = "unknown"
    try:
        import json
        data = json.loads(payload)
        event_type = data.get("type", "unknown")
    except Exception:
        pass
    db["usage"].insert_one({"type": "billing", "event": event_type, "created_at": datetime.utcnow()})
    return {"received": True}

# Admin
@app.get("/admin/usage")
def admin_usage(user=Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    items = list(db["usage"].find().sort("created_at", -1).limit(200))
    for i in items:
        i["id"] = str(i.pop("_id"))
    return items

# Developer guide endpoint
@app.get("/developer-guide")
def developer_guide():
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    return {
        "fetch_voices": {
            "request": f"GET {backend_url}/voices",
            "response": [{"id": "...", "key": "en-US-1", "name": "US English", "language": "en-US"}],
        },
        "request_synthesis": {
            "request": f"POST {backend_url}/tts/synthesize",
            "body": {"voice_key": "en-US-1", "text": "Hello world", "format": "mp3"},
        },
        "stream_ws": {
            "connect": f"WS {backend_url.replace('http', 'ws')}/ws/synthesize",
            "send": {"voice_key": "en-US-1", "text": "Hello"},
            "receive": ["progress", "chunk", "completed"],
        },
        "upload_clone": {
            "request": f"POST {backend_url}/voices/clone (multipart/form-data)",
            "fields": ["name", "description", "file"],
        },
        "billing": {
            "create_checkout_session": f"POST {backend_url}/billing/create-checkout-session",
            "webhook": f"POST {backend_url}/billing/webhook",
        },
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
import uuid
import json

# Load .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/token")
async def get_token(data: dict):
    room = data.get("call_id", f"room_{uuid.uuid4().hex[:8]}")
    identity = f"user_{uuid.uuid4().hex[:8]}"
    
    # ✅ Get API keys from environment
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    # Validate keys are loaded
    if not api_key or not api_secret:
        return {
            "error": "LIVEKIT_API_KEY or LIVEKIT_API_SECRET not found in .env file"
        }
    
    # Create token
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(identity)
    
    # Add grants
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        can_update_own_metadata=True
    )
    
    token.with_grants(grants)
    
    # Add metadata
    token.with_metadata(json.dumps({
        "language": data.get("language", "en"),
        "agent_id": data.get("agent_id"),
        "call_id": data.get("call_id")
    }))
    
    jwt = token.to_jwt()
    
    # Debug: print to console (remove in production)
    print(f"✅ Token generated for room: {room}")
    print(f"   Identity: {identity}")
    print(f"   Language: {data.get('language', 'en')}")
    
    return {
        "token": jwt,
        "url": "wss://livekit.callshivai.com"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Verify .env is loaded
    print("=" * 50)
    print("Starting Token Server")
    print("=" * 50)
    if os.getenv("LIVEKIT_API_KEY"):
        print(f"✅ API Key loaded: {os.getenv('LIVEKIT_API_KEY')[:10]}...")
    else:
        print("❌ API Key NOT loaded - check .env file!")
    
    if os.getenv("LIVEKIT_API_SECRET"):
        print(f"✅ API Secret loaded: {os.getenv('LIVEKIT_API_SECRET')[:10]}...")
    else:
        print("❌ API Secret NOT loaded - check .env file!")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=3000)

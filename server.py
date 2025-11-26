import asyncio
import logging
import os
import sys
from dotenv import load_dotenv
load_dotenv() 
from datetime import datetime, timezone
import json
import threading

logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('pymongo.topology').setLevel(logging.WARNING)
logging.getLogger('pymongo.connection').setLevel(logging.WARNING)
logging.getLogger('pymongo.serverSelection').setLevel(logging.WARNING)

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, llm
from livekit.plugins import openai

from openai.types.beta.realtime.session import TurnDetection

from summary_script import generate_summary

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIRESTORE_AVAILABLE = True
except:
    FIRESTORE_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streaming-voice-assistant")

LANGUAGE_METADATA = {
    "en": {"label": "English", "trigger": "Hello!", "code": "en-US"},
    "es": {"label": "Spanish", "trigger": "Hola!", "code": "es-ES"},
    "fr": {"label": "French", "trigger": "Bonjour!", "code": "fr-FR"},
    "de": {"label": "German", "trigger": "Hallo!", "code": "de-DE"},
    "hi": {"label": "Hindi", "trigger": "Namaste!", "code": "hi-IN"},
    "pt": {"label": "Portuguese", "trigger": "Ola!", "code": "pt-PT"},
    "zh": {"label": "Chinese", "trigger": "Ni hao!", "code": "zh-CN"},
    "ja": {"label": "Japanese", "trigger": "Konnichiwa!", "code": "ja-JP"},
    "ko": {"label": "Korean", "trigger": "Annyeonghaseyo!", "code": "ko-KR"},
    "ar": {"label": "Arabic", "trigger": "Marhaban!", "code": "ar-SA"},
    "ru": {"label": "Russian", "trigger": "Privet!", "code": "ru-RU"},
    "it": {"label": "Italian", "trigger": "Ciao!", "code": "it-IT"},
    "nl": {"label": "Dutch", "trigger": "Hallo!", "code": "nl-NL"},
    "pl": {"label": "Polish", "trigger": "Czesc!", "code": "pl-PL"},
    "tr": {"label": "Turkish", "trigger": "Merhaba!", "code": "tr-TR"},
}

def get_language_metadata(language_code):
    key = (language_code or "en").lower().split('-')[0]
    return LANGUAGE_METADATA.get(key, LANGUAGE_METADATA["en"])

def load_knowledge_base():
    knowledge = {}
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    for file_name, key in [
        ('base_knowledge.json', 'base'),
        ('faqs.json', 'faqs'),
        ('use_cases.json', 'use_cases')
    ]:
        try:
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                knowledge[key] = json.load(f)
        except Exception as e:
            logger.debug(f"Could not load {file_name}: {e}")
            knowledge[key] = {}
    
    return knowledge

def build_system_instructions():
    kb = load_knowledge_base()
    
    instructions = (
        "You are a helpful, concise, and professional AI assistant. "
        "Keep responses natural and brief (1-3 sentences)."
    )
    
    if kb.get('base'):
        base = kb['base']
        about_text = base.get('description', '')
        if about_text:
            instructions += f"\n\nABOUT: {about_text}"
    
    return instructions

SYSTEM_INSTRUCTIONS = build_system_instructions()

def init_firestore():
    if not FIRESTORE_AVAILABLE:
        logger.warning("Firestore not available")
        return None
    
    try:
        if not firebase_admin._apps:
            cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if cred_path:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                logger.info("✅ Firestore initialized")
            else:
                fb_private_key = os.getenv('FIREBASE_PRIVATE_KEY')
                if fb_private_key:
                    if fb_private_key.startswith('"') and fb_private_key.endswith('"'):
                        fb_private_key = fb_private_key[1:-1]
                    fb_private_key = fb_private_key.replace('\\\\n', '\n')
                    
                    cred_dict = {
                        'type': os.getenv('FIREBASE_TYPE', 'service_account'),
                        'project_id': os.getenv('FIREBASE_PROJECT_ID'),
                        'private_key_id': os.getenv('FIREBASE_PRIVATE_KEY_ID'),
                        'private_key': fb_private_key,
                        'client_email': os.getenv('FIREBASE_CLIENT_EMAIL'),
                        'client_id': os.getenv('FIREBASE_CLIENT_ID'),
                        'auth_uri': os.getenv('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                        'token_uri': os.getenv('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                        'auth_provider_x509_cert_url': os.getenv('FIREBASE_AUTH_PROVIDER_X509_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
                        'client_x509_cert_url': os.getenv('FIREBASE_CLIENT_X509_CERT_URL')
                    }
                    cred = credentials.Certificate(cred_dict)
                    firebase_admin.initialize_app(cred)
                    logger.info("✅ Firestore initialized from env")
        
        return firestore.client()
    except Exception as e:
        logger.warning(f"Firestore error: {e}")
        return None

db_client = init_firestore()
call_sessions = {}
agent_config_cache = {}
CACHE_TTL_SECONDS = 300

INACTIVITY_REMINDER_SECONDS = int(os.getenv("INACTIVITY_REMINDER_SECONDS", "20"))
INACTIVITY_END_CALL_SECONDS = int(os.getenv("INACTIVITY_END_CALL_SECONDS", "40"))

class StreamingVoiceAssistant(Agent):
    def __init__(self, instructions: str, session_id: str):
        super().__init__(instructions=instructions)
        self.session_id = session_id
        self.last_user_activity = datetime.now(timezone.utc)
        self.reminder_sent = False
        self.inactivity_task = None
        self.session_ref = None
        self.is_active = True
    
    async def on_user_turn_completed(self, chat_ctx, new_message):
        self.last_user_activity = datetime.now(timezone.utc)
        self.reminder_sent = False
        
        if new_message.text_content and self.session_id in call_sessions:
            logger.info(f"👤 User: {new_message.text_content}")
            call_sessions[self.session_id]["user_transcripts"].append({
                "text": new_message.text_content,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source": "user"
            })
            call_sessions[self.session_id]["total_user_messages"] += 1
    
    async def on_agent_turn_completed(self, chat_ctx, new_message):
        if new_message.text_content and self.session_id in call_sessions:
            logger.info(f"🤖 Agent: {new_message.text_content}")
            call_sessions[self.session_id]["agent_transcripts"].append({
                "text": new_message.text_content,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source": "agent"
            })
            call_sessions[self.session_id]["total_agent_responses"] += 1
    
    async def check_inactivity(self):
        try:
            while self.is_active:
                await asyncio.sleep(1)
                
                if self.session_id not in call_sessions or not self.is_active:
                    break
                
                elapsed = (datetime.now(timezone.utc) - self.last_user_activity).total_seconds()
                
                if elapsed > INACTIVITY_REMINDER_SECONDS and not self.reminder_sent and self.session_ref and self.is_active:
                    logger.info(f"⏰ Inactivity reminder")
                    self.reminder_sent = True
                    try:
                        await self.session_ref.generate_reply(
                            instructions="Ask if the user is still there (one short sentence)."
                        )
                    except:
                        pass
                
                if elapsed > INACTIVITY_END_CALL_SECONDS and self.is_active:
                    logger.info(f"⏰ Disconnecting")
                    await self.finalize_call()
                    break
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Inactivity error: {e}")
    
    async def finalize_call(self):
        self.is_active = False
        
        if self.session_id not in call_sessions:
            return
        
        session_data = call_sessions[self.session_id]
        session_data["end_time_utc"] = datetime.now(timezone.utc).isoformat()
        
        try:
            start = datetime.fromisoformat(session_data["start_time_utc"])
            end = datetime.fromisoformat(session_data["end_time_utc"])
            session_data["duration_seconds"] = (end - start).total_seconds()
        except:
            pass
        
        async def firestore_write(data):
            if db_client:
                try:
                    await asyncio.to_thread(
                        lambda: db_client.collection('call_analytics').document(self.session_id).set(data)
                    )
                    logger.info(f"✅ Analytics saved")
                except Exception as e:
                    logger.error(f"Analytics error: {e}")
        
        asyncio.create_task(firestore_write(session_data))
        
        agent_id = session_data.get("agent_id")
        call_id = session_data.get("call_id")
        
        if agent_id and call_id and db_client:
            full_transcript = []
            for ut in session_data.get("user_transcripts", []):
                full_transcript.append(f"User: {ut['text']}")
            for at in session_data.get("agent_transcripts", []):
                full_transcript.append(f"Agent: {at['text']}")
            
            transcript_text = "\n".join(full_transcript)
            
            if transcript_text.strip():
                threading.Thread(
                    target=generate_summary,
                    args=(agent_id, call_id, transcript_text, session_data.get('client_info', {}), db_client),
                    daemon=True
                ).start()
                logger.info(f"🔄 Summary started")
        
        call_sessions.pop(self.session_id, None)

async def load_agent_config_async(agent_id):
    if not agent_id or not db_client:
        return None
    
    cache_key = f"agent_{agent_id}"
    if cache_key in agent_config_cache:
        cached_data, cached_time = agent_config_cache[cache_key]
        if (datetime.now(timezone.utc) - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            logger.info(f"✅ Agent config from cache")
            return cached_data
    
    try:
        doc = await asyncio.to_thread(
            lambda: db_client.collection('agents').document(agent_id).get()
        )
        if doc.exists:
            config = doc.to_dict()
            agent_config_cache[cache_key] = (config, datetime.now(timezone.utc))
            logger.info(f"✅ Agent config loaded")
            return config
    except Exception as e:
        logger.error(f"Agent load error: {e}")
    
    return None

async def load_knowledge_base_async(agent_id):
    if not agent_id or not db_client:
        return ""
    
    try:
        kb_ref = db_client.collection('agents').document(agent_id).collection('knowledge_base')
        docs = await asyncio.to_thread(lambda: list(kb_ref.stream()))
        
        if docs:
            knowledge_items = [doc.to_dict() for doc in docs]
            knowledge_text = " ".join([item.get('content', '') for item in knowledge_items])
            logger.info(f"✅ KB loaded ({len(knowledge_items)} items)")
            return f"\n\nKNOWLEDGE:\n{knowledge_text}"
    except Exception as e:
        logger.debug(f"KB load error: {e}")
    
    return ""

async def entrypoint(ctx: JobContext):
    session = None  # Track session for cleanup
    assistant = None  # Track assistant for cleanup
    
    try:
        logger.info(f"🎙️ Room: {ctx.room.name}")
        
        await ctx.connect()
        logger.info("✅ Connected")
        
        participant = await ctx.wait_for_participant()
        logger.info(f"👤 Participant: {participant.identity}")
        
        metadata = {}
        try:
            if participant.metadata:
                metadata = json.loads(participant.metadata)
        except:
            pass
        
        language = metadata.get("language", "en")
        agent_id = metadata.get("agent_id")
        call_id = metadata.get("call_id") or ctx.room.name
        
        logger.info(f"Lang: {language}, Agent: {agent_id}")
        
        client_info = {
            "ip": metadata.get("ip", "unknown"),
            "device_type": metadata.get("device", "unknown"),
            "user_agent": metadata.get("user_agent", ""),
        }
        
        # Parallel loading
        config_task = asyncio.create_task(load_agent_config_async(agent_id))
        kb_task = asyncio.create_task(load_knowledge_base_async(agent_id))
        
        agent_config, kb_text = await asyncio.gather(config_task, kb_task)
        
        instructions_text = SYSTEM_INSTRUCTIONS
        voice = "sage"
        
        if agent_config:
            voice = agent_config.get('voice', voice)
            if agent_config.get('custom_instructions'):
                instructions_text = agent_config['custom_instructions']
            
            instructions_text += kb_text
            
            gender = agent_config.get('gender', '').lower()
            if gender in ['male', 'female', 'neutral']:
                instructions_text += f"\n\nYou are a {gender} AI agent."
        
        lang_meta = get_language_metadata(language)
        instructions_text += f"\n\nRespond in {lang_meta['label']}."
        
        session_id = ctx.room.name
        call_sessions[session_id] = {
            "session_id": session_id,
            "room_name": ctx.room.name,
            "call_id": call_id,
            "agent_id": agent_id,
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "client_info": client_info,
            "user_transcripts": [],
            "agent_transcripts": [],
            "total_user_messages": 0,
            "total_agent_responses": 0,
        }
        
        logger.info("🔧 Initializing OpenAI Realtime API...")
        
        # ✅ FIX: Create fresh assistant instance
        assistant = StreamingVoiceAssistant(
            instructions=instructions_text, 
            session_id=session_id
        )
        
        # ✅ FIX: Create NEW session with fresh RealtimeModel (critical for mobile)
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                voice=voice,
                temperature=0.7,
                modalities=["text", "audio"],
                turn_detection=TurnDetection(
                    type="server_vad",
                    threshold=0.5,
                    prefix_padding_ms=300,
                    silence_duration_ms=500,
                    create_response=True,
                ),
            ),
        )
        
        assistant.session_ref = session
        
        # ✅ FIX: Add explicit session cleanup on disconnect
        async def cleanup_session():
            logger.info("🧹 Cleaning up OpenAI session...")
            assistant.is_active = False
            if assistant.inactivity_task:
                assistant.inactivity_task.cancel()
                try:
                    await assistant.inactivity_task
                except:
                    pass
            
            # ✅ CRITICAL: Properly close the OpenAI session
            if session:
                try:
                    await session.aclose()  # Close OpenAI WebSocket connection
                    logger.info("✅ OpenAI session closed")
                except Exception as e:
                    logger.warning(f"Session close warning: {e}")
            
            await assistant.finalize_call()
        
        ctx.add_shutdown_callback(cleanup_session)
        
        # Start session
        await session.start(room=ctx.room, agent=assistant)
        
        logger.info("✅ Realtime API Initialized")
        
        await asyncio.sleep(0.3)
        
        # Generate greeting
        if agent_config and agent_config.get('greeting_message'):
            greeting = agent_config.get('greeting_message')
            await session.generate_reply(instructions=f"Say: {greeting}")
        else:
            await session.generate_reply(instructions="Give a warm, brief greeting.")
        
        assistant.inactivity_task = asyncio.create_task(assistant.check_inactivity())
        
        logger.info("🎉 Ready - REALTIME API ENABLED!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        
        # ✅ FIX: Cleanup on error
        if session:
            try:
                await session.aclose()
            except:
                pass
        if assistant:
            assistant.is_active = False
        
        raise

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append('start')
        print("✅ Auto-added 'start' command")

    livekit_url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    
    print("=" * 60)
    print("🚀 Starting LiveKit Voice Agent")
    print("=" * 60)
    print(f"LiveKit URL: {livekit_url}")
    print(f"API Key: {os.getenv('LIVEKIT_API_KEY', 'NOT SET')[:20]}...")
    print("=" * 60)
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=livekit_url
        )
    )

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio
import json
import os
import logging
from websockets.legacy.client import connect
from datetime import datetime
import time
from collections import deque

load_dotenv(override=True)

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("interview-bot")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Import for service account authentication
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Configuration (from environment variables)
PROJECT_ID = os.getenv("PROJECT_ID", "sqy-prod")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_ID = "gemini-live-2.5-flash-native-audio"
MODEL = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"
HOST = f"wss://{LOCATION}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

# 🔥 PRODUCTION SETTINGS
MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "1000"))
CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", "1800"))  # 30 minutes

INTERVIEW_PROMPT = """
You are conducting a real-time technical interview for a Software Engineer position.
You are based in INDIA and conducting this interview in Indian Standard Time (IST, UTC+5:30).

You can hear and also see the candidate through audio and video.

# 🔴 PROCTORING & MONITORING (HIGH PRIORITY):
You MUST continuously monitor the video feed and IMMEDIATELY warn the candidate if you detect:

1. **Multiple People Detected**: If you see more than ONE person in the frame:
   - Immediately say: "I notice there might be someone else in the room. For interview integrity, please ensure you are alone. This will be noted."

2. **Mobile Phone Usage**: If you see the candidate using or looking at a mobile phone:
   - Immediately say: "I noticed you looking at your phone. Please keep your phone away during the interview. Using external devices is not allowed."

3. **Candidate Not Visible**: If the candidate is not visible or has moved out of frame:
   - Immediately say: "I can't see you on the screen. Please adjust your camera so I can see you clearly."

4. **Looking Away / Not Focused**: If the candidate is frequently looking away from the screen (looking left, right, up, or down repeatedly):
   - Say: "I notice you're looking away from the screen. Please focus on the interview and maintain eye contact with the camera."

5. **Suspicious Behavior**: If you see any suspicious behavior like reading from another screen, someone whispering, or unusual movements:
   - Say: "I noticed some unusual activity. Please remember this is a proctored interview and any unfair means will be recorded."

6. **Tab Switching / Distraction**: If the candidate appears distracted or seems to be reading something off-screen:
   - Say: "It seems like you might be looking at something else. Please give your full attention to the interview."

# 🌐 NETWORK MONITORING:
If you receive a message indicating the candidate's network quality is POOR:
- Say: "I'm noticing some connectivity issues on your end. If possible, please move to a location with better internet connection for a smoother interview experience."

# ⏱️ SILENT USER DETECTION:
The system will send you messages about candidate silence. Respond appropriately:
- If you receive "[SYSTEM] Candidate silent for 30 seconds - waiting for response": 
  Say: "I am waiting for your response."
- If you receive "[SYSTEM] Candidate silent for 50 seconds - FINAL WARNING - Interview will end in 10 seconds":
  Say firmly: "If you do not respond, we will end the interview in 10 seconds."
- If you receive "[SYSTEM] Ending interview due to no response":
  Say: "Since there has been no response, we are ending this interview session now. Thank you for your time."
- If you receive "[SYSTEM] Interview time limit (30 minutes) reached. Ending interview.":
  Say: "We have reached the 30-minute time limit for this interview. Thank you for your time today. The interview is now complete."

# 🔄 IMPORTANT - CONTINUING INTERVIEW:
If the candidate speaks or responds after any warning (including the final warning), you MUST:
- IMMEDIATELY continue the interview as normal
- Do NOT say "the interview has ended" or "we've reached the time limit" (unless the 30-minute timer actually reached zero)
- Do NOT refuse to continue - just pick up where you left off
- Simply acknowledge their response and continue with the next question
The silence warnings are just prompts - if the user responds, the interview continues!

# ⚠️ IMPORTANT: Issue warnings in a FIRM but PROFESSIONAL tone. Do not be rude, but be clear that violations are being noted.

# Interview Structure:
1. Greet the candidate appropriately based on Indian time:
   - Morning (6 AM - 12 PM IST): "Good morning"
   - Afternoon (12 PM - 5 PM IST): "Good afternoon"  
   - Evening (5 PM - 9 PM IST): "Good evening"
   - Night (9 PM - 6 AM IST): "Hello"

2. Ask candidate to introduce themselves
3. Ask 3 technical questions about:
   - Data structures and algorithms
   - System design
   - Problem-solving approach
4. Ask 2 behavioral questions
5. Close the interview professionally

# Visual Observation Rules:
- You can see the candidate through video
- Answer visual questions ONLY based on what is clearly visible
- If something is not clearly visible, say you are not certain
- Do not guess or assume

# Communication Rules:
- Be professional but friendly (Indian professional context)
- Listen carefully and ask follow-up questions
- Keep responses concise
- Encourage the candidate when they do well
- Use natural, conversational language
- Speak clearly in English (Indian candidates may have regional accents - be patient)
"""

# Service Account Authentication
def get_access_token():
    """Get access token using service account credentials"""
    try:
        # Load credentials from environment or file
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            # Load from JSON string in environment variable
            credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
            if credentials_json:
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            else:
                logger.error("No credentials found in environment")
                return None
        
        # Refresh token
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        logger.error(f"Error getting access token: {e}")
        return None

# Lifespan context manager (replaces deprecated @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 60)
    logger.info("VOICE + VIDEO INTERVIEW BOT API")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Max Connections: {MAX_CONCURRENT_CONNECTIONS}")
    logger.info(f"Connection Timeout: {CONNECTION_TIMEOUT}s")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Video Support: ENABLED")
    logger.info("=" * 60)
    logger.info("Server Ready!")
    yield
    # Shutdown
    logger.info("Server shutting down...")

# Initialize FastAPI
app = FastAPI(
    title="Voice + Video Interview Bot API - Production",
    description="High-performance interview bot with unlimited rate limits",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection Management
class ConnectionManager:
    def __init__(self):
        self.active_connections = 0
        self.total_connections = 0
        self.connection_history = deque(maxlen=100)
        self.token_cache = None
        self.token_expiry = None
        self.start_time = datetime.now()
    
    def can_accept_connection(self) -> bool:
        return self.active_connections < MAX_CONCURRENT_CONNECTIONS
    
    def add_connection(self):
        self.active_connections += 1
        self.total_connections += 1
        self.connection_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'connected',
            'active': self.active_connections,
            'total': self.total_connections
        })
    
    def remove_connection(self):
        self.active_connections = max(0, self.active_connections - 1)
        self.connection_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'disconnected',
            'active': self.active_connections
        })
    
    def get_cached_token(self):
        """Cache token to optimize performance"""
        now = datetime.now()
        if self.token_cache and self.token_expiry and now < self.token_expiry:
            return self.token_cache
        
        # Get new token
        token = get_access_token()
        if token:
            self.token_cache = token
            # Cache for 50 minutes (tokens valid for 1 hour)
            from datetime import timedelta
            self.token_expiry = now + timedelta(minutes=50)
        return token
    
    def get_stats(self):
        uptime = datetime.now() - self.start_time
        return {
            'active_connections': self.active_connections,
            'total_connections': self.total_connections,
            'max_capacity': MAX_CONCURRENT_CONNECTIONS,
            'available_slots': MAX_CONCURRENT_CONNECTIONS - self.active_connections,
            'uptime_seconds': int(uptime.total_seconds()),
            'uptime_formatted': str(uptime).split('.')[0]
        }

manager = ConnectionManager()

async def relay_messages(ws_client: WebSocket, ws_google):
    """Handle bidirectional message relay between client and Gemini"""
    
    # Store session resumption handle
    session_handle = None
    
    async def client2server(source: WebSocket, target):
        """Browser → Gemini (audio + video)"""
        msg_count = 0
        audio_chunk_count = 0
        try:
            while True:
                message = await source.receive_text()
                msg_count += 1
                data = json.loads(message)
                
                # Logging (only in debug mode)
                if 'realtimeInput' in data:
                    audio_chunk_count += 1
                    if audio_chunk_count % 100 == 0:
                        logger.debug(f"Media chunks sent: {audio_chunk_count}")
                else:
                    logger.debug(f"Browser→Gemini message #{msg_count}")
                
                await target.send(message)
        except WebSocketDisconnect:
            logger.debug("Client disconnected from relay")
        except Exception as e:
            logger.error(f"Error client2server: {e}")
    
    async def server2client(source, target: WebSocket):
        """Gemini → Browser"""
        nonlocal session_handle
        msg_count = 0
        try:
            async for message in source:
                msg_count += 1
                data = json.loads(message.decode('utf-8'))
                
                # Handle session resumption updates
                if 'sessionResumptionUpdate' in data:
                    update = data['sessionResumptionUpdate']
                    if update.get('resumable') and update.get('newHandle'):
                        session_handle = update['newHandle']
                        logger.debug("Session resumption handle updated")
                
                # Handle GoAway message (connection will terminate soon)
                if 'goAway' in data:
                    time_left = data['goAway'].get('timeLeft', 'unknown')
                    logger.warning(f"Connection will close in {time_left}. Resumption handle available: {bool(session_handle)}")
                
                # Detailed logging in debug mode
                if 'serverContent' in data:
                    content = data['serverContent']
                    
                    if 'modelTurn' in content:
                        logger.debug("AI Speaking")
                    
                    if 'outputTranscription' in content:
                        text = content['outputTranscription'].get('text', '')
                        logger.debug(f"AI said: {text}")
                    
                    if 'inputTranscription' in content:
                        text = content['inputTranscription'].get('text', '')
                        is_final = content['inputTranscription'].get('isFinal', False)
                        if is_final:
                            logger.debug(f"User said: {text}")
                    
                    if 'generationComplete' in content:
                        logger.debug("Generation complete")
                
                elif 'setupComplete' in data:
                    logger.debug("Setup complete")
                
                await target.send_text(message.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error server2client: {e}")
    
    # Set timeout for the entire connection
    try:
        await asyncio.wait_for(
            asyncio.gather(
                client2server(ws_client, ws_google),
                server2client(ws_google, ws_client),
                return_exceptions=True
            ),
            timeout=CONNECTION_TIMEOUT
        )
    except asyncio.TimeoutError:
        print(f"⏰ Connection timeout after {CONNECTION_TIMEOUT} seconds")

@app.get("/")
async def root():
    """API information endpoint"""
    stats = manager.get_stats()
    return {
        "status": "online",
        "service": "Voice + Video Interview Bot API",
        "version": "2.0.0",
        "model": MODEL_ID,
        "features": ["audio", "video", "transcription", "unlimited-rate-limits"],
        "websocket_endpoint": "/ws/interview",
        **stats
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring and load balancers"""
    stats = manager.get_stats()
    is_healthy = stats['active_connections'] < MAX_CONCURRENT_CONNECTIONS
    
    return {
        "status": "healthy" if is_healthy else "at_capacity",
        "video_support": True,
        "rate_limits": "unlimited",
        **stats
    }

@app.get("/stats")
async def get_stats():
    """Detailed statistics endpoint"""
    stats = manager.get_stats()
    return {
        **stats,
        "recent_activity": list(manager.connection_history)[-20:],
        "configuration": {
            "max_concurrent_connections": MAX_CONCURRENT_CONNECTIONS,
            "connection_timeout": CONNECTION_TIMEOUT,
            "model": MODEL_ID,
            "location": LOCATION
        }
    }

@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket):
    """Main WebSocket endpoint for voice + video interview"""
    
    # Check capacity
    if not manager.can_accept_connection():
        await websocket.close(code=1008, reason="Server at capacity")
        logger.warning(f"Connection rejected - At capacity ({manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS})")
        return
    
    await websocket.accept()
    manager.add_connection()
    
    connection_id = manager.total_connections
    
    logger.info(f"Client #{connection_id} connected ({manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS} active)")
    
    # Get cached token for better performance
    access_token = manager.get_cached_token()
    
    if not access_token:
        logger.error("Failed to get access token")
        manager.remove_connection()
        await websocket.close(code=1011, reason="Authentication failed")
        return
    
    try:
        async with connect(
            HOST,
            extra_headers={'Authorization': f'Bearer {access_token}'},
            ping_interval=20,
            ping_timeout=10,
            max_size=10_000_000  # 10MB max message size for video
        ) as ws_google:
            # Setup with audio and video support + UNLIMITED SESSION TIME
            initial_request = {
                "setup": {
                    "model": MODEL,
                    "generationConfig": {
                        "temperature": 0.7,
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": "Aoede"
                                }
                            }
                        }
                    },
                    "systemInstruction": {
                        "parts": [{"text": INTERVIEW_PROMPT}]
                    },
                    "input_audio_transcription": {},
                    "output_audio_transcription": {},
                    # 🔥 CRITICAL: Enable context window compression for unlimited session time
                    "context_window_compression": {
                        "sliding_window": {},
                        "trigger_tokens": 50000  # Compress when context reaches 50K tokens
                    },
                    # 🔥 Enable session resumption for handling connection resets
                    "session_resumption": {}
                }
            }
            
            await ws_google.send(json.dumps(initial_request))
            
            logger.debug(f"Client #{connection_id} - AI initialized with video and transcription")
            
            await relay_messages(websocket, ws_google)
            
    except WebSocketDisconnect:
        logger.info(f"Client #{connection_id} disconnected")
    except Exception as e:
        logger.error(f"Client #{connection_id} error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
    finally:
        manager.remove_connection()
        logger.info(f"Client #{connection_id} session ended. Active: {manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS}")

# Startup event moved to lifespan context manager above

# Network Info Endpoint for latency measurement
@app.get("/api/network-info")
async def network_info():
    """
    Returns server timestamp for client-side latency calculation.
    This endpoint is used by the frontend to measure network quality.
    """
    return {
        "timestamp": int(time.time() * 1000),  # milliseconds
        "status": "ok",
        "server_time": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    stats = manager.get_stats()
    return {
        "status": "healthy",
        "active_connections": stats['active_connections'],
        "uptime": stats['uptime_formatted']
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=MAX_CONCURRENT_CONNECTIONS + 50,  # Buffer for safety
        timeout_keep_alive=75,
        ws_ping_interval=20,
        ws_ping_timeout=10
    )
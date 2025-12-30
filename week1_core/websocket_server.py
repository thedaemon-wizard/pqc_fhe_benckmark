"""
PQC-FHE WebSocket API Server
============================

Real-time WebSocket interface for Post-Quantum Cryptography + FHE operations.

Features:
- Bidirectional communication for long-running FHE operations
- Real-time progress updates during encryption/computation
- Streaming results for large encrypted datasets
- Session-based key management

Protocol:
---------
Client -> Server messages:
    {"type": "ping"}
    {"type": "subscribe", "channels": ["fhe_progress", "key_events"]}
    {"type": "pqc_keygen", "algorithm": "ML-KEM-768", "request_id": "uuid"}
    {"type": "fhe_encrypt", "data": [...], "context_id": "ctx_123", "request_id": "uuid"}
    {"type": "fhe_compute", "operation": "add", "operands": [...], "request_id": "uuid"}
    
Server -> Client messages:
    {"type": "pong", "timestamp": "..."}
    {"type": "progress", "request_id": "uuid", "percent": 50, "message": "..."}
    {"type": "result", "request_id": "uuid", "status": "success", "data": {...}}
    {"type": "error", "request_id": "uuid", "code": "ERROR_CODE", "message": "..."}

References:
-----------
- RFC 6455: The WebSocket Protocol
- NIST FIPS 203/204/205: Post-Quantum Cryptography Standards
- DESILO FHE v5.5.0: https://fhe.desilo.dev/latest/

Author: Amon (Quantum Computing Specialist)
License: MIT
Version: 1.0.0
"""

import asyncio
import json
import logging
import uuid
import time
from datetime import datetime
from typing import Optional, Dict, Any, Set, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class MessageType(str, Enum):
    """WebSocket message types"""
    # Client -> Server
    PING = "ping"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PQC_KEYGEN = "pqc_keygen"
    PQC_ENCAPSULATE = "pqc_encapsulate"
    PQC_DECAPSULATE = "pqc_decapsulate"
    PQC_SIGN = "pqc_sign"
    PQC_VERIFY = "pqc_verify"
    FHE_ENCRYPT = "fhe_encrypt"
    FHE_DECRYPT = "fhe_decrypt"
    FHE_COMPUTE = "fhe_compute"
    SESSION_CREATE = "session_create"
    SESSION_DESTROY = "session_destroy"
    
    # Server -> Client
    PONG = "pong"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    KEY_EVENT = "key_event"
    SESSION_EVENT = "session_event"


class ErrorCode(str, Enum):
    """Error codes for WebSocket responses"""
    INVALID_MESSAGE = "INVALID_MESSAGE"
    INVALID_TYPE = "INVALID_TYPE"
    MISSING_FIELD = "MISSING_FIELD"
    ALGORITHM_NOT_SUPPORTED = "ALGORITHM_NOT_SUPPORTED"
    KEY_NOT_FOUND = "KEY_NOT_FOUND"
    ENCRYPTION_FAILED = "ENCRYPTION_FAILED"
    DECRYPTION_FAILED = "DECRYPTION_FAILED"
    COMPUTATION_FAILED = "COMPUTATION_FAILED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class Channel(str, Enum):
    """Subscription channels"""
    FHE_PROGRESS = "fhe_progress"
    KEY_EVENTS = "key_events"
    SESSION_EVENTS = "session_events"
    ALL = "all"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WebSocketMessage:
    """Base WebSocket message structure"""
    type: str
    request_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ProgressMessage(WebSocketMessage):
    """Progress update message"""
    percent: int = 0
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.type = MessageType.PROGRESS.value


@dataclass
class ResultMessage(WebSocketMessage):
    """Operation result message"""
    status: str = "success"
    data: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    
    def __post_init__(self):
        self.type = MessageType.RESULT.value


@dataclass
class ErrorMessage(WebSocketMessage):
    """Error message"""
    code: str = ErrorCode.INTERNAL_ERROR.value
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.type = MessageType.ERROR.value


@dataclass
class ClientSession:
    """WebSocket client session state"""
    session_id: str
    connected_at: datetime
    subscriptions: Set[str] = field(default_factory=set)
    keys: Dict[str, Dict[str, bytes]] = field(default_factory=dict)
    fhe_contexts: Dict[str, Any] = field(default_factory=dict)
    request_count: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# WEBSOCKET HANDLER
# =============================================================================

class PQCFHEWebSocketHandler:
    """
    WebSocket handler for PQC-FHE operations
    
    Manages client connections, sessions, and message routing.
    
    Architecture:
                        ┌─────────────────────────────────────┐
                        │      WebSocket Connection Pool       │
                        └─────────────────────────────────────┘
                                          │
                        ┌─────────────────────────────────────┐
                        │       Message Router & Validator     │
                        └─────────────────────────────────────┘
                                    /           \
              ┌────────────────────────┐   ┌────────────────────────┐
              │    PQC Operations      │   │    FHE Operations      │
              │  - Key Generation      │   │  - Encryption          │
              │  - Encapsulation       │   │  - Computation         │
              │  - Signatures          │   │  - Decryption          │
              └────────────────────────┘   └────────────────────────┘
    """
    
    def __init__(
        self,
        pqc_manager=None,
        fhe_engine=None,
        rate_limit: int = 100,
        max_sessions: int = 1000,
    ):
        """
        Initialize WebSocket handler
        
        Args:
            pqc_manager: PQCKeyManager instance
            fhe_engine: FHEEngine instance
            rate_limit: Max requests per minute per session
            max_sessions: Maximum concurrent sessions
        """
        self.pqc_manager = pqc_manager
        self.fhe_engine = fhe_engine
        self.rate_limit = rate_limit
        self.max_sessions = max_sessions
        
        # Session management
        self.sessions: Dict[str, ClientSession] = {}
        self.connections: Dict[str, Any] = {}  # websocket connections
        
        # Channel subscriptions (channel -> set of session_ids)
        self.channel_subscribers: Dict[str, Set[str]] = {
            Channel.FHE_PROGRESS.value: set(),
            Channel.KEY_EVENTS.value: set(),
            Channel.SESSION_EVENTS.value: set(),
        }
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "total_messages": 0,
            "total_errors": 0,
            "operations": {},
        }
        
        logger.info("PQCFHEWebSocketHandler initialized")
    
    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================
    
    async def handle_connection(self, websocket, path: str = "/"):
        """
        Handle new WebSocket connection
        
        Args:
            websocket: WebSocket connection object
            path: Connection path
        """
        session_id = str(uuid.uuid4())
        session = ClientSession(
            session_id=session_id,
            connected_at=datetime.utcnow(),
        )
        
        self.sessions[session_id] = session
        self.connections[session_id] = websocket
        self.stats["total_connections"] += 1
        
        logger.info(f"New connection: session_id={session_id}, path={path}")
        
        # Send welcome message
        welcome = ResultMessage(
            request_id="welcome",
            status="connected",
            data={
                "session_id": session_id,
                "server_time": datetime.utcnow().isoformat(),
                "available_operations": [t.value for t in MessageType],
                "rate_limit": self.rate_limit,
            }
        )
        await self._send(websocket, welcome)
        
        try:
            async for message in websocket:
                await self._handle_message(session_id, message)
        except Exception as e:
            logger.error(f"Connection error for {session_id}: {e}")
        finally:
            await self._cleanup_session(session_id)
    
    async def _cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Remove from all channel subscriptions
            for channel in session.subscriptions:
                if channel in self.channel_subscribers:
                    self.channel_subscribers[channel].discard(session_id)
            
            # Clean up FHE contexts
            session.fhe_contexts.clear()
            session.keys.clear()
            
            del self.sessions[session_id]
        
        if session_id in self.connections:
            del self.connections[session_id]
        
        logger.info(f"Session cleaned up: {session_id}")
    
    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================
    
    async def _handle_message(self, session_id: str, raw_message: str):
        """
        Route and handle incoming message
        
        Args:
            session_id: Client session ID
            raw_message: Raw JSON message string
        """
        session = self.sessions.get(session_id)
        websocket = self.connections.get(session_id)
        
        if not session or not websocket:
            return
        
        session.request_count += 1
        session.last_activity = datetime.utcnow()
        self.stats["total_messages"] += 1
        
        # Parse message
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as e:
            await self._send_error(
                websocket, None, ErrorCode.INVALID_MESSAGE,
                f"Invalid JSON: {e}"
            )
            return
        
        msg_type = message.get("type")
        request_id = message.get("request_id", str(uuid.uuid4()))
        
        if not msg_type:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "Missing 'type' field"
            )
            return
        
        # Route message
        try:
            if msg_type == MessageType.PING.value:
                await self._handle_ping(websocket, request_id)
            
            elif msg_type == MessageType.SUBSCRIBE.value:
                await self._handle_subscribe(session, message, request_id)
            
            elif msg_type == MessageType.UNSUBSCRIBE.value:
                await self._handle_unsubscribe(session, message, request_id)
            
            elif msg_type == MessageType.PQC_KEYGEN.value:
                await self._handle_pqc_keygen(session, websocket, message, request_id)
            
            elif msg_type == MessageType.PQC_ENCAPSULATE.value:
                await self._handle_pqc_encapsulate(session, websocket, message, request_id)
            
            elif msg_type == MessageType.PQC_DECAPSULATE.value:
                await self._handle_pqc_decapsulate(session, websocket, message, request_id)
            
            elif msg_type == MessageType.PQC_SIGN.value:
                await self._handle_pqc_sign(session, websocket, message, request_id)
            
            elif msg_type == MessageType.PQC_VERIFY.value:
                await self._handle_pqc_verify(session, websocket, message, request_id)
            
            elif msg_type == MessageType.FHE_ENCRYPT.value:
                await self._handle_fhe_encrypt(session, websocket, message, request_id)
            
            elif msg_type == MessageType.FHE_DECRYPT.value:
                await self._handle_fhe_decrypt(session, websocket, message, request_id)
            
            elif msg_type == MessageType.FHE_COMPUTE.value:
                await self._handle_fhe_compute(session, websocket, message, request_id)
            
            else:
                await self._send_error(
                    websocket, request_id, ErrorCode.INVALID_TYPE,
                    f"Unknown message type: {msg_type}"
                )
        
        except Exception as e:
            logger.exception(f"Error handling message: {e}")
            self.stats["total_errors"] += 1
            await self._send_error(
                websocket, request_id, ErrorCode.INTERNAL_ERROR,
                str(e)
            )
    
    # =========================================================================
    # BASIC HANDLERS
    # =========================================================================
    
    async def _handle_ping(self, websocket, request_id: str):
        """Handle ping message"""
        pong = WebSocketMessage(type=MessageType.PONG.value, request_id=request_id)
        await self._send(websocket, pong)
    
    async def _handle_subscribe(self, session: ClientSession, message: dict, request_id: str):
        """Handle channel subscription"""
        channels = message.get("channels", [])
        websocket = self.connections.get(session.session_id)
        
        for channel in channels:
            if channel in self.channel_subscribers:
                self.channel_subscribers[channel].add(session.session_id)
                session.subscriptions.add(channel)
        
        result = ResultMessage(
            request_id=request_id,
            status="subscribed",
            data={"channels": list(session.subscriptions)}
        )
        await self._send(websocket, result)
    
    async def _handle_unsubscribe(self, session: ClientSession, message: dict, request_id: str):
        """Handle channel unsubscription"""
        channels = message.get("channels", [])
        websocket = self.connections.get(session.session_id)
        
        for channel in channels:
            if channel in self.channel_subscribers:
                self.channel_subscribers[channel].discard(session.session_id)
                session.subscriptions.discard(channel)
        
        result = ResultMessage(
            request_id=request_id,
            status="unsubscribed",
            data={"channels": list(session.subscriptions)}
        )
        await self._send(websocket, result)
    
    # =========================================================================
    # PQC HANDLERS
    # =========================================================================
    
    async def _handle_pqc_keygen(
        self, session: ClientSession, websocket, message: dict, request_id: str
    ):
        """
        Handle PQC key generation request
        
        Message format:
            {
                "type": "pqc_keygen",
                "algorithm": "ML-KEM-768",
                "key_id": "optional_key_id",
                "request_id": "uuid"
            }
        """
        start_time = time.time()
        
        algorithm = message.get("algorithm", "ML-KEM-768")
        key_id = message.get("key_id", str(uuid.uuid4()))
        
        # Send progress
        await self._send_progress(websocket, request_id, 10, "Initializing key generation...")
        
        if not self.pqc_manager:
            await self._send_error(
                websocket, request_id, ErrorCode.INTERNAL_ERROR,
                "PQC manager not initialized"
            )
            return
        
        try:
            await self._send_progress(websocket, request_id, 30, "Generating keypair...")
            
            # Determine key type (KEM or Signature)
            if "KEM" in algorithm or "Kyber" in algorithm:
                public_key, secret_key = self.pqc_manager.generate_kem_keypair()
                key_type = "kem"
            else:
                public_key, secret_key = self.pqc_manager.generate_sig_keypair()
                key_type = "signature"
            
            await self._send_progress(websocket, request_id, 80, "Storing keys...")
            
            # Store in session
            session.keys[key_id] = {
                "public_key": public_key,
                "secret_key": secret_key,
                "algorithm": algorithm,
                "type": key_type,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            await self._send_progress(websocket, request_id, 100, "Complete")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ResultMessage(
                request_id=request_id,
                status="success",
                data={
                    "key_id": key_id,
                    "algorithm": algorithm,
                    "type": key_type,
                    "public_key_size": len(public_key),
                    "secret_key_size": len(secret_key),
                    "public_key_hex": public_key[:64].hex() + "...",
                },
                execution_time_ms=execution_time,
            )
            await self._send(websocket, result)
            
            # Broadcast to subscribers
            await self._broadcast_channel(
                Channel.KEY_EVENTS.value,
                {
                    "type": "key_generated",
                    "session_id": session.session_id,
                    "key_id": key_id,
                    "algorithm": algorithm,
                }
            )
            
            self._update_stats("pqc_keygen", execution_time)
            
        except Exception as e:
            logger.exception(f"PQC keygen failed: {e}")
            await self._send_error(
                websocket, request_id, ErrorCode.INTERNAL_ERROR,
                str(e)
            )
    
    async def _handle_pqc_encapsulate(
        self, session: ClientSession, websocket, message: dict, request_id: str
    ):
        """Handle KEM encapsulation"""
        start_time = time.time()
        
        key_id = message.get("key_id")
        public_key_hex = message.get("public_key_hex")
        
        if not key_id and not public_key_hex:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "Either 'key_id' or 'public_key_hex' required"
            )
            return
        
        try:
            await self._send_progress(websocket, request_id, 20, "Retrieving public key...")
            
            if key_id and key_id in session.keys:
                public_key = session.keys[key_id]["public_key"]
            elif public_key_hex:
                public_key = bytes.fromhex(public_key_hex)
            else:
                await self._send_error(
                    websocket, request_id, ErrorCode.KEY_NOT_FOUND,
                    f"Key not found: {key_id}"
                )
                return
            
            await self._send_progress(websocket, request_id, 50, "Encapsulating...")
            
            ciphertext, shared_secret = self.pqc_manager.encapsulate(public_key)
            
            await self._send_progress(websocket, request_id, 100, "Complete")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ResultMessage(
                request_id=request_id,
                status="success",
                data={
                    "ciphertext_hex": ciphertext.hex(),
                    "shared_secret_hex": shared_secret.hex(),
                    "ciphertext_size": len(ciphertext),
                    "shared_secret_size": len(shared_secret),
                },
                execution_time_ms=execution_time,
            )
            await self._send(websocket, result)
            
            self._update_stats("pqc_encapsulate", execution_time)
            
        except Exception as e:
            logger.exception(f"Encapsulation failed: {e}")
            await self._send_error(
                websocket, request_id, ErrorCode.ENCRYPTION_FAILED,
                str(e)
            )
    
    async def _handle_pqc_decapsulate(
        self, session: ClientSession, websocket, message: dict, request_id: str
    ):
        """Handle KEM decapsulation"""
        start_time = time.time()
        
        key_id = message.get("key_id")
        ciphertext_hex = message.get("ciphertext_hex")
        
        if not key_id or not ciphertext_hex:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "'key_id' and 'ciphertext_hex' required"
            )
            return
        
        try:
            await self._send_progress(websocket, request_id, 20, "Retrieving secret key...")
            
            if key_id not in session.keys:
                await self._send_error(
                    websocket, request_id, ErrorCode.KEY_NOT_FOUND,
                    f"Key not found: {key_id}"
                )
                return
            
            secret_key = session.keys[key_id]["secret_key"]
            ciphertext = bytes.fromhex(ciphertext_hex)
            
            await self._send_progress(websocket, request_id, 50, "Decapsulating...")
            
            shared_secret = self.pqc_manager.decapsulate(ciphertext, secret_key)
            
            await self._send_progress(websocket, request_id, 100, "Complete")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ResultMessage(
                request_id=request_id,
                status="success",
                data={
                    "shared_secret_hex": shared_secret.hex(),
                    "shared_secret_size": len(shared_secret),
                },
                execution_time_ms=execution_time,
            )
            await self._send(websocket, result)
            
            self._update_stats("pqc_decapsulate", execution_time)
            
        except Exception as e:
            logger.exception(f"Decapsulation failed: {e}")
            await self._send_error(
                websocket, request_id, ErrorCode.DECRYPTION_FAILED,
                str(e)
            )
    
    async def _handle_pqc_sign(
        self, session: ClientSession, websocket, message: dict, request_id: str
    ):
        """Handle digital signature generation"""
        start_time = time.time()
        
        key_id = message.get("key_id")
        message_hex = message.get("message_hex")
        message_text = message.get("message_text")
        
        if not key_id:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "'key_id' required"
            )
            return
        
        if not message_hex and not message_text:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "Either 'message_hex' or 'message_text' required"
            )
            return
        
        try:
            await self._send_progress(websocket, request_id, 20, "Retrieving signing key...")
            
            if key_id not in session.keys:
                await self._send_error(
                    websocket, request_id, ErrorCode.KEY_NOT_FOUND,
                    f"Key not found: {key_id}"
                )
                return
            
            secret_key = session.keys[key_id]["secret_key"]
            
            if message_hex:
                msg_bytes = bytes.fromhex(message_hex)
            else:
                msg_bytes = message_text.encode('utf-8')
            
            await self._send_progress(websocket, request_id, 50, "Signing message...")
            
            signature = self.pqc_manager.sign(msg_bytes, secret_key)
            
            await self._send_progress(websocket, request_id, 100, "Complete")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ResultMessage(
                request_id=request_id,
                status="success",
                data={
                    "signature_hex": signature.hex(),
                    "signature_size": len(signature),
                    "message_size": len(msg_bytes),
                },
                execution_time_ms=execution_time,
            )
            await self._send(websocket, result)
            
            self._update_stats("pqc_sign", execution_time)
            
        except Exception as e:
            logger.exception(f"Signing failed: {e}")
            await self._send_error(
                websocket, request_id, ErrorCode.INTERNAL_ERROR,
                str(e)
            )
    
    async def _handle_pqc_verify(
        self, session: ClientSession, websocket, message: dict, request_id: str
    ):
        """Handle signature verification"""
        start_time = time.time()
        
        key_id = message.get("key_id")
        public_key_hex = message.get("public_key_hex")
        message_hex = message.get("message_hex")
        message_text = message.get("message_text")
        signature_hex = message.get("signature_hex")
        
        if not signature_hex:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "'signature_hex' required"
            )
            return
        
        try:
            await self._send_progress(websocket, request_id, 20, "Retrieving verification key...")
            
            if key_id and key_id in session.keys:
                public_key = session.keys[key_id]["public_key"]
            elif public_key_hex:
                public_key = bytes.fromhex(public_key_hex)
            else:
                await self._send_error(
                    websocket, request_id, ErrorCode.KEY_NOT_FOUND,
                    "Either 'key_id' or 'public_key_hex' required"
                )
                return
            
            if message_hex:
                msg_bytes = bytes.fromhex(message_hex)
            elif message_text:
                msg_bytes = message_text.encode('utf-8')
            else:
                await self._send_error(
                    websocket, request_id, ErrorCode.MISSING_FIELD,
                    "Either 'message_hex' or 'message_text' required"
                )
                return
            
            signature = bytes.fromhex(signature_hex)
            
            await self._send_progress(websocket, request_id, 50, "Verifying signature...")
            
            is_valid = self.pqc_manager.verify(msg_bytes, signature, public_key)
            
            await self._send_progress(websocket, request_id, 100, "Complete")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ResultMessage(
                request_id=request_id,
                status="success",
                data={
                    "valid": is_valid,
                    "message_size": len(msg_bytes),
                    "signature_size": len(signature),
                },
                execution_time_ms=execution_time,
            )
            await self._send(websocket, result)
            
            self._update_stats("pqc_verify", execution_time)
            
        except Exception as e:
            logger.exception(f"Verification failed: {e}")
            await self._send_error(
                websocket, request_id, ErrorCode.INTERNAL_ERROR,
                str(e)
            )
    
    # =========================================================================
    # FHE HANDLERS
    # =========================================================================
    
    async def _handle_fhe_encrypt(
        self, session: ClientSession, websocket, message: dict, request_id: str
    ):
        """
        Handle FHE encryption with progress updates
        
        Message format:
            {
                "type": "fhe_encrypt",
                "data": [1.0, 2.0, 3.0, ...],
                "context_id": "optional_context_id",
                "request_id": "uuid"
            }
        """
        start_time = time.time()
        
        data = message.get("data")
        context_id = message.get("context_id", str(uuid.uuid4()))
        
        if not data:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "'data' field required"
            )
            return
        
        if not self.fhe_engine:
            await self._send_error(
                websocket, request_id, ErrorCode.INTERNAL_ERROR,
                "FHE engine not initialized"
            )
            return
        
        try:
            data_array = np.array(data, dtype=np.float64)
            total_elements = len(data_array)
            
            await self._send_progress(
                websocket, request_id, 10,
                f"Preparing {total_elements} elements for encryption..."
            )
            
            # Progress updates for large data
            if total_elements > 1000:
                await self._send_progress(
                    websocket, request_id, 20,
                    "Chunking data for slot-based encryption..."
                )
            
            await self._send_progress(websocket, request_id, 40, "Encoding plaintext...")
            
            await self._send_progress(websocket, request_id, 60, "Encrypting ciphertext...")
            
            ciphertext = self.fhe_engine.encrypt(data_array)
            
            await self._send_progress(websocket, request_id, 80, "Serializing ciphertext...")
            
            # Store in session
            session.fhe_contexts[context_id] = {
                "ciphertext": ciphertext,
                "original_length": total_elements,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            await self._send_progress(websocket, request_id, 100, "Encryption complete")
            
            execution_time = (time.time() - start_time) * 1000
            
            # Estimate ciphertext size (CKKS ciphertexts are large)
            estimated_size = total_elements * 8 * 60  # rough estimate
            
            result = ResultMessage(
                request_id=request_id,
                status="success",
                data={
                    "context_id": context_id,
                    "elements_encrypted": total_elements,
                    "estimated_ciphertext_size_kb": estimated_size // 1024,
                    "level": getattr(ciphertext, 'level', 'unknown'),
                },
                execution_time_ms=execution_time,
            )
            await self._send(websocket, result)
            
            # Broadcast progress to subscribers
            await self._broadcast_channel(
                Channel.FHE_PROGRESS.value,
                {
                    "operation": "encrypt",
                    "session_id": session.session_id,
                    "context_id": context_id,
                    "elements": total_elements,
                    "execution_time_ms": execution_time,
                }
            )
            
            self._update_stats("fhe_encrypt", execution_time)
            
        except Exception as e:
            logger.exception(f"FHE encryption failed: {e}")
            await self._send_error(
                websocket, request_id, ErrorCode.ENCRYPTION_FAILED,
                str(e)
            )
    
    async def _handle_fhe_decrypt(
        self, session: ClientSession, websocket, message: dict, request_id: str
    ):
        """Handle FHE decryption"""
        start_time = time.time()
        
        context_id = message.get("context_id")
        
        if not context_id:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "'context_id' required"
            )
            return
        
        if context_id not in session.fhe_contexts:
            await self._send_error(
                websocket, request_id, ErrorCode.KEY_NOT_FOUND,
                f"Context not found: {context_id}"
            )
            return
        
        try:
            ctx = session.fhe_contexts[context_id]
            ciphertext = ctx["ciphertext"]
            original_length = ctx["original_length"]
            
            await self._send_progress(websocket, request_id, 30, "Decrypting ciphertext...")
            
            decrypted = self.fhe_engine.decrypt(ciphertext, original_length)
            
            await self._send_progress(websocket, request_id, 100, "Decryption complete")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ResultMessage(
                request_id=request_id,
                status="success",
                data={
                    "context_id": context_id,
                    "data": decrypted.tolist() if hasattr(decrypted, 'tolist') else list(decrypted),
                    "elements": len(decrypted),
                },
                execution_time_ms=execution_time,
            )
            await self._send(websocket, result)
            
            self._update_stats("fhe_decrypt", execution_time)
            
        except Exception as e:
            logger.exception(f"FHE decryption failed: {e}")
            await self._send_error(
                websocket, request_id, ErrorCode.DECRYPTION_FAILED,
                str(e)
            )
    
    async def _handle_fhe_compute(
        self, session: ClientSession, websocket, message: dict, request_id: str
    ):
        """
        Handle FHE homomorphic computation
        
        Message format:
            {
                "type": "fhe_compute",
                "operation": "add",  // add, subtract, multiply, negate, square
                "operands": ["ctx_1", "ctx_2"],  // context IDs or scalar values
                "result_context_id": "optional",
                "request_id": "uuid"
            }
        """
        start_time = time.time()
        
        operation = message.get("operation")
        operands = message.get("operands", [])
        result_context_id = message.get("result_context_id", str(uuid.uuid4()))
        
        if not operation:
            await self._send_error(
                websocket, request_id, ErrorCode.MISSING_FIELD,
                "'operation' required"
            )
            return
        
        supported_ops = ["add", "subtract", "multiply", "negate", "square", "add_scalar", "multiply_scalar"]
        if operation not in supported_ops:
            await self._send_error(
                websocket, request_id, ErrorCode.INVALID_TYPE,
                f"Unsupported operation: {operation}. Supported: {supported_ops}"
            )
            return
        
        try:
            await self._send_progress(websocket, request_id, 20, f"Preparing {operation} operation...")
            
            # Resolve operands
            resolved_operands = []
            for op in operands:
                if isinstance(op, str) and op in session.fhe_contexts:
                    resolved_operands.append(session.fhe_contexts[op]["ciphertext"])
                else:
                    resolved_operands.append(op)  # scalar value
            
            await self._send_progress(websocket, request_id, 50, f"Executing {operation}...")
            
            # Execute operation
            if operation == "add" and len(resolved_operands) == 2:
                result_ct = self.fhe_engine.add(resolved_operands[0], resolved_operands[1])
            elif operation == "subtract" and len(resolved_operands) == 2:
                result_ct = self.fhe_engine.sub(resolved_operands[0], resolved_operands[1])
            elif operation == "multiply" and len(resolved_operands) == 2:
                result_ct = self.fhe_engine.mult(resolved_operands[0], resolved_operands[1])
            elif operation == "negate" and len(resolved_operands) == 1:
                result_ct = self.fhe_engine.negate(resolved_operands[0])
            elif operation == "square" and len(resolved_operands) == 1:
                result_ct = self.fhe_engine.square(resolved_operands[0])
            elif operation == "add_scalar" and len(resolved_operands) == 2:
                result_ct = self.fhe_engine.add_scalar(resolved_operands[0], float(resolved_operands[1]))
            elif operation == "multiply_scalar" and len(resolved_operands) == 2:
                result_ct = self.fhe_engine.mult_scalar(resolved_operands[0], float(resolved_operands[1]))
            else:
                await self._send_error(
                    websocket, request_id, ErrorCode.INVALID_TYPE,
                    f"Invalid operand count for {operation}"
                )
                return
            
            await self._send_progress(websocket, request_id, 80, "Storing result...")
            
            # Store result
            session.fhe_contexts[result_context_id] = {
                "ciphertext": result_ct,
                "original_length": session.fhe_contexts[operands[0]]["original_length"] if operands[0] in session.fhe_contexts else 0,
                "created_at": datetime.utcnow().isoformat(),
                "operation": operation,
            }
            
            await self._send_progress(websocket, request_id, 100, "Computation complete")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ResultMessage(
                request_id=request_id,
                status="success",
                data={
                    "result_context_id": result_context_id,
                    "operation": operation,
                    "operand_count": len(operands),
                    "level": getattr(result_ct, 'level', 'unknown'),
                },
                execution_time_ms=execution_time,
            )
            await self._send(websocket, result)
            
            # Broadcast to subscribers
            await self._broadcast_channel(
                Channel.FHE_PROGRESS.value,
                {
                    "operation": f"compute_{operation}",
                    "session_id": session.session_id,
                    "result_context_id": result_context_id,
                    "execution_time_ms": execution_time,
                }
            )
            
            self._update_stats(f"fhe_compute_{operation}", execution_time)
            
        except Exception as e:
            logger.exception(f"FHE computation failed: {e}")
            await self._send_error(
                websocket, request_id, ErrorCode.COMPUTATION_FAILED,
                str(e)
            )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    async def _send(self, websocket, message: WebSocketMessage):
        """Send message to client"""
        try:
            await websocket.send(message.to_json())
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def _send_progress(
        self, websocket, request_id: str, percent: int, message: str
    ):
        """Send progress update"""
        progress = ProgressMessage(
            request_id=request_id,
            percent=percent,
            message=message,
        )
        await self._send(websocket, progress)
    
    async def _send_error(
        self, websocket, request_id: Optional[str], code: ErrorCode, message: str
    ):
        """Send error message"""
        error = ErrorMessage(
            request_id=request_id,
            code=code.value,
            message=message,
        )
        await self._send(websocket, error)
        self.stats["total_errors"] += 1
    
    async def _broadcast_channel(self, channel: str, data: dict):
        """Broadcast message to channel subscribers"""
        subscribers = self.channel_subscribers.get(channel, set())
        
        for session_id in subscribers:
            websocket = self.connections.get(session_id)
            if websocket:
                try:
                    message = {
                        "type": "broadcast",
                        "channel": channel,
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await websocket.send(json.dumps(message))
                except Exception as e:
                    logger.warning(f"Failed to broadcast to {session_id}: {e}")
    
    def _update_stats(self, operation: str, execution_time_ms: float):
        """Update operation statistics"""
        if operation not in self.stats["operations"]:
            self.stats["operations"][operation] = {
                "count": 0,
                "total_time_ms": 0,
                "min_time_ms": float('inf'),
                "max_time_ms": 0,
            }
        
        stats = self.stats["operations"][operation]
        stats["count"] += 1
        stats["total_time_ms"] += execution_time_ms
        stats["min_time_ms"] = min(stats["min_time_ms"], execution_time_ms)
        stats["max_time_ms"] = max(stats["max_time_ms"], execution_time_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        stats = self.stats.copy()
        stats["active_sessions"] = len(self.sessions)
        stats["active_connections"] = len(self.connections)
        
        for op_name, op_stats in stats["operations"].items():
            if op_stats["count"] > 0:
                op_stats["avg_time_ms"] = op_stats["total_time_ms"] / op_stats["count"]
        
        return stats


# =============================================================================
# WEBSOCKET SERVER
# =============================================================================

async def create_websocket_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    pqc_manager=None,
    fhe_engine=None,
):
    """
    Create and start WebSocket server
    
    Args:
        host: Server host address
        port: Server port
        pqc_manager: PQCKeyManager instance
        fhe_engine: FHEEngine instance
        
    Returns:
        WebSocket server instance
    """
    try:
        import websockets
    except ImportError:
        logger.error("websockets library required: pip install websockets")
        raise
    
    handler = PQCFHEWebSocketHandler(
        pqc_manager=pqc_manager,
        fhe_engine=fhe_engine,
    )
    
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    
    server = await websockets.serve(
        handler.handle_connection,
        host,
        port,
        ping_interval=30,
        ping_timeout=10,
        max_size=10 * 1024 * 1024,  # 10MB max message size
    )
    
    return server, handler


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line entry point for WebSocket server"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PQC-FHE WebSocket Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port
  python websocket_server.py
  
  # Start server on custom port
  python websocket_server.py --port 9000
  
  # Enable debug logging
  python websocket_server.py --debug
        """
    )
    
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Server port (default: 8765)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run_server():
        # Initialize crypto managers (optional, can work without them)
        pqc_manager = None
        fhe_engine = None
        
        try:
            import sys
            sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
            from src.pqc_fhe_integration import PQCKeyManager, FHEEngine, PQCConfig, FHEConfig
            
            pqc_manager = PQCKeyManager(PQCConfig())
            fhe_engine = FHEEngine(FHEConfig())
            logger.info("Initialized PQC and FHE engines")
        except Exception as e:
            logger.warning(f"Could not initialize crypto engines: {e}")
            logger.warning("Server will run with limited functionality")
        
        server, handler = await create_websocket_server(
            host=args.host,
            port=args.port,
            pqc_manager=pqc_manager,
            fhe_engine=fhe_engine,
        )
        
        logger.info(f"WebSocket server running on ws://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop")
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            server.close()
            await server.wait_closed()
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == "__main__":
    main()

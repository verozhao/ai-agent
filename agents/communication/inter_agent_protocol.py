"""
Inter-agent communication protocol for seamless collaboration.
"""

import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
import json

import aioredis
from cryptography.fernet import Fernet


class MessagePriority(Enum):
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class InterAgentProtocol:
    """
    Secure, scalable inter-agent communication protocol.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        encryption_key: Optional[bytes] = None
    ):
        self.redis_url = redis_url
        self.redis = None
        
        # Encryption for sensitive messages
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = Fernet(Fernet.generate_key())
        
        # Message channels
        self.channels = {
            "broadcast": "agent:broadcast",
            "direct": "agent:direct:",
            "emergency": "agent:emergency"
        }
        
        # Message handlers
        self.handlers = {}
        
    async def connect(self):
        """Connect to Redis for message passing"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
    
    async def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.MEDIUM,
        encrypted: bool = False
    ) -> str:
        """Send message to another agent"""
        message = {
            "id": f"msg_{datetime.utcnow().timestamp()}",
            "sender": sender,
            "recipient": recipient,
            "type": message_type,
            "content": content,
            "priority": priority.value,
            "timestamp": datetime.utcnow().isoformat(),
            "encrypted": encrypted
        }
        
        # Encrypt if needed
        if encrypted:
            message["content"] = self._encrypt_content(content)
        
        # Determine channel
        if recipient == "all":
            channel = self.channels["broadcast"]
        elif priority == MessagePriority.CRITICAL:
            channel = self.channels["emergency"]
        else:
            channel = self.channels["direct"] + recipient
        
        # Send message
        await self.redis.publish(channel, json.dumps(message))
        
        # Store for persistence
        await self._store_message(message)
        
        return message["id"]
    
    async def receive_messages(
        self,
        agent_id: str,
        callback
    ):
        """Subscribe to messages for an agent"""
        channels = [
            self.channels["broadcast"],
            self.channels["direct"] + agent_id,
            self.channels["emergency"]
        ]
        
        # Create subscription
        channel_sub = await self.redis.subscribe(*channels)
        
        try:
            while True:
                message = await channel_sub.get()
                if message:
                    data = json.loads(message.decode())
                    
                    # Decrypt if needed
                    if data.get("encrypted"):
                        data["content"] = self._decrypt_content(data["content"])
                    
                    # Call handler
                    await callback(data)
                    
        except asyncio.CancelledError:
            await channel_sub.unsubscribe(*channels)
            raise
    
    def _encrypt_content(self, content: Dict[str, Any]) -> str:
        """Encrypt message content"""
        json_content = json.dumps(content)
        encrypted = self.cipher.encrypt(json_content.encode())
        return encrypted.decode()
    
    def _decrypt_content(self, encrypted_content: str) -> Dict[str, Any]:
        """Decrypt message content"""
        decrypted = self.cipher.decrypt(encrypted_content.encode())
        return json.loads(decrypted.decode())
    
    async def _store_message(self, message: Dict[str, Any]):
        """Store message for audit trail"""
        key = f"message_history:{message['id']}"
        await self.redis.setex(
            key,
            86400,  # 24 hour retention
            json.dumps(message)
        )
        
        # Add to sender's sent messages
        sender_key = f"sent_messages:{message['sender']}"
        await self.redis.lpush(sender_key, message['id'])
        await self.redis.ltrim(sender_key, 0, 999)  # Keep last 1000
        
        # Add to recipient's received messages
        if message['recipient'] != "all":
            recipient_key = f"received_messages:{message['recipient']}"
            await self.redis.lpush(recipient_key, message['id'])
            await self.redis.ltrim(recipient_key, 0, 999)
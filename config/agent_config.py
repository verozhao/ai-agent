"""
Configuration for the AI agent system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentSystemConfig:
    """Configuration for AI agent system"""
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    encryption_key: Optional[bytes] = None
    
    # Agent configuration
    num_extractors: int = 3
    num_analyzers: int = 2
    num_auditors: int = 2
    
    # System configuration
    monitor_interval: int = 30  # seconds
    health_check_port: int = 8080
    
    # Model configuration
    llm_model: str = "meta-llama/Llama-2-70b-chat-hf"
    extraction_model: str = "microsoft/layoutlmv3-base"
    
    # Performance configuration
    max_concurrent_tasks: int = 100
    task_timeout: int = 300  # seconds
    
    # Learning configuration
    learning_enabled: bool = True
    min_feedback_for_learning: int = 10
    
    @classmethod
    def from_env(cls) -> "AgentSystemConfig":
        """Load configuration from environment variables"""
        import os
        
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            num_extractors=int(os.getenv("NUM_EXTRACTORS", "3")),
            num_analyzers=int(os.getenv("NUM_ANALYZERS", "2")),
            num_auditors=int(os.getenv("NUM_AUDITORS", "2")),
            llm_model=os.getenv("LLM_MODEL", "meta-llama/Llama-2-70b-chat-hf"),
            extraction_model=os.getenv("EXTRACTION_MODEL", "microsoft/layoutlmv3-base")
        )

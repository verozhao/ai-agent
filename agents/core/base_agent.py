"""
Base AI Agent framework for autonomous document processing with reasoning capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from agents.reasoning.chain_of_thought import ChainOfThoughtReasoner
from agents.memory.vector_store import AgentMemoryStore
from agents.communication.orchestrator import CommunicationOrchestrator


logger = logging.getLogger(__name__)


class AgentRole(Enum):
    EXTRACTOR = "extractor"
    ANALYZER = "analyzer"
    AUDITOR = "auditor"
    COORDINATOR = "coordinator"
    LEARNER = "learner"
    COMMUNICATOR = "communicator"


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    PROCESSING = "processing"
    COMMUNICATING = "communicating"
    LEARNING = "learning"
    DECIDING = "deciding"


@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    priority: int
    timestamp: datetime
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class AgentDecision:
    """Structured decision from agent reasoning"""
    action: str
    confidence: float
    reasoning: List[str]
    alternatives: List[Tuple[str, float]]
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """
    Base class for all AI agents in the document processing system.
    Implements autonomous reasoning, learning, and communication.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        llm_model: str = "meta-llama/Llama-2-70b-chat-hf",
        memory_capacity: int = 1000
    ):
        self.agent_id = agent_id
        self.role = role
        self.state = AgentState.IDLE
        
        # Initialize LLM for reasoning
        self._init_llm(llm_model)
        
        # Initialize components
        self.reasoner = ChainOfThoughtReasoner(self.llm)
        self.memory = AgentMemoryStore(capacity=memory_capacity)
        self.communicator = CommunicationOrchestrator()
        
        # Agent-specific tools
        self.tools = self._init_tools()
        
        # Performance tracking
        self.decisions_made = 0
        self.success_rate = 1.0
        self.learning_episodes = []
        
        # Message queue
        self.message_queue = asyncio.Queue()
        
    def _init_llm(self, model_name: str):
        """Initialize language model for reasoning"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    @abstractmethod
    def _init_tools(self) -> List[Tool]:
        """Initialize agent-specific tools"""
        pass
    
    async def think(self, context: Dict[str, Any]) -> AgentDecision:
        """
        Core reasoning method - agent thinks about the current situation
        and decides on the best action.
        """
        self.state = AgentState.THINKING
        
        # Retrieve relevant memories
        relevant_memories = await self.memory.retrieve_relevant(
            query=context.get("situation", ""),
            k=5
        )
        
        # Construct reasoning prompt
        reasoning_prompt = self._construct_reasoning_prompt(
            context, relevant_memories
        )
        
        # Generate chain of thought reasoning
        reasoning_chain = await self.reasoner.reason(
            prompt=reasoning_prompt,
            context=context
        )
        
        # Make decision based on reasoning
        decision = await self._make_decision(reasoning_chain, context)
        
        # Store experience in memory
        await self.memory.store({
            "context": context,
            "reasoning": reasoning_chain,
            "decision": decision,
            "timestamp": datetime.utcnow()
        })
        
        self.decisions_made += 1
        self.state = AgentState.IDLE
        
        return decision
    
    def _construct_reasoning_prompt(
        self,
        context: Dict[str, Any],
        memories: List[Dict[str, Any]]
    ) -> str:
        """Construct prompt for reasoning based on context and memories"""
        prompt = f"""As a {self.role.value} agent, analyze the current situation and decide on the best action.

Current Context:
{self._format_context(context)}

Relevant Past Experiences:
{self._format_memories(memories)}

Consider the following:
1. What is the primary goal in this situation?
2. What are the potential actions I can take?
3. What are the likely outcomes of each action?
4. Which action best aligns with the system's objectives?
5. What could go wrong and how can I mitigate risks?

Provide your reasoning step by step, then conclude with a specific action."""
        
        return prompt
    
    async def _make_decision(
        self,
        reasoning_chain: List[str],
        context: Dict[str, Any]
    ) -> AgentDecision:
        """Make a decision based on reasoning chain"""
        # Extract action from reasoning
        action = self._extract_action(reasoning_chain)
        
        # Calculate confidence based on reasoning clarity
        confidence = self._calculate_confidence(reasoning_chain)
        
        # Generate alternatives
        alternatives = await self._generate_alternatives(context)
        
        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning_chain,
            alternatives=alternatives,
            metadata={
                "agent_id": self.agent_id,
                "role": self.role.value,
                "timestamp": datetime.utcnow()
            }
        )
    
    async def communicate(
        self,
        recipient: str,
        message_type: str,
        content: Dict[str, Any],
        priority: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Communicate with other agents or humans"""
        self.state = AgentState.COMMUNICATING
        
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            priority=priority,
            timestamp=datetime.utcnow(),
            requires_response=True
        )
        
        # Send message through orchestrator
        response = await self.communicator.send_message(message)
        
        self.state = AgentState.IDLE
        return response
    
    async def learn_from_feedback(
        self,
        feedback: Dict[str, Any]
    ):
        """Learn from feedback to improve future decisions"""
        self.state = AgentState.LEARNING
        
        # Analyze feedback
        feedback_analysis = await self.reasoner.analyze_feedback(
            feedback,
            self.memory.get_recent_decisions()
        )
        
        # Update success rate
        if feedback.get("outcome") == "success":
            self.success_rate = (
                self.success_rate * 0.95 + 1.0 * 0.05
            )  # Exponential moving average
        else:
            self.success_rate = self.success_rate * 0.95
        
        # Store learning episode
        self.learning_episodes.append({
            "feedback": feedback,
            "analysis": feedback_analysis,
            "timestamp": datetime.utcnow()
        })
        
        # Update decision-making strategy if needed
        if self.success_rate < 0.8:
            await self._adapt_strategy()
        
        self.state = AgentState.IDLE
    
    async def _adapt_strategy(self):
        """Adapt decision-making strategy based on performance"""
        # Analyze failure patterns
        recent_failures = [
            ep for ep in self.learning_episodes[-20:]
            if ep["feedback"].get("outcome") != "success"
        ]
        
        if recent_failures:
            # Identify common failure patterns
            failure_analysis = await self.reasoner.analyze_patterns(
                recent_failures
            )
            
            # Update reasoning approach
            self.reasoner.update_strategy(failure_analysis)
            
            logger.info(
                f"Agent {self.agent_id} adapted strategy based on "
                f"{len(recent_failures)} recent failures"
            )
    
    async def process_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Process incoming message from other agents"""
        # Add to message queue
        await self.message_queue.put(message)
        
        # Process if high priority
        if message.priority >= 8:
            return await self._handle_message(message)
        
        return None
    
    @abstractmethod
    async def _handle_message(
        self,
        message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
        """Handle specific message types"""
        pass
    
    async def run(self):
        """Main agent loop"""
        logger.info(f"Agent {self.agent_id} ({self.role.value}) started")
        
        while True:
            try:
                # Process messages
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._handle_message(message)
                
                # Perform role-specific tasks
                await self._perform_role_tasks()
                
                # Brief pause to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Agent {self.agent_id} error: {str(e)}")
                await self.learn_from_feedback({
                    "outcome": "error",
                    "error": str(e)
                })


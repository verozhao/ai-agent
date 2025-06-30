"""
Conversational AI agent for natural language interaction with the system.
"""

import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, AIMessage

from agents.core.base_agent import BaseAgent, AgentRole
from agents.interface.intent_classifier import IntentClassifier
from agents.interface.response_generator import ResponseGenerator
from utils.nlp_processor import NLPProcessor


logger = logging.getLogger(__name__)


class ConversationalAgent(BaseAgent):
    """
    Natural language interface agent that enables human interaction
    with the document processing system through conversation.
    """
    
    def __init__(
        self,
        agent_id: str = "conversational_001",
        model_name: str = "microsoft/DialoGPT-large"
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COMMUNICATOR
        )
        
        # Initialize conversational components
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        self.nlp_processor = NLPProcessor()
        
        # Conversation memory per user
        self.user_conversations = {}
        
        # Available intents and their handlers
        self.intent_handlers = {
            "process_document": self._handle_process_document,
            "check_status": self._handle_check_status,
            "get_insights": self._handle_get_insights,
            "provide_feedback": self._handle_provide_feedback,
            "ask_question": self._handle_ask_question,
            "system_health": self._handle_system_health,
            "help": self._handle_help
        }
        
    def _init_tools(self) -> List[Tool]:
        """Initialize conversational tools"""
        return [
            Tool(
                name="classify_intent",
                func=self.intent_classifier.classify,
                description="Classify user intent from natural language"
            ),
            Tool(
                name="extract_entities",
                func=self.nlp_processor.extract_entities,
                description="Extract entities from user message"
            ),
            Tool(
                name="generate_response",
                func=self.response_generator.generate,
                description="Generate natural language response"
            ),
            Tool(
                name="query_system_state",
                func=self._query_system_state,
                description="Query current system state"
            )
        ]
    
    async def chat(
        self,
        user_id: str,
        message: str,
        streaming: bool = False
    ) -> AsyncGenerator[str, None] | Dict[str, Any]:
        """
        Process user message and generate response.
        
        Args:
            user_id: Unique user identifier
            message: User's message
            streaming: Whether to stream response
            
        Returns:
            Response as stream or complete message
        """
        # Get or create conversation memory
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=1000
            )
        
        memory = self.user_conversations[user_id]
        
        # Add user message to memory
        memory.chat_memory.add_user_message(message)
        
        # Classify intent
        intent_result = await self.intent_classifier.classify(message)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        
        # Extract entities
        entities = await self.nlp_processor.extract_entities(message)
        
        # Create context for reasoning
        context = {
            "user_id": user_id,
            "message": message,
            "intent": intent,
            "intent_confidence": confidence,
            "entities": entities,
            "conversation_history": memory.chat_memory.messages[-5:],
            "situation": f"User asking about {intent}"
        }
        
        # Reason about response
        decision = await self.think(context)
        
        # Handle based on intent
        if confidence > 0.8:
            handler = self.intent_handlers.get(intent, self._handle_unknown)
            response_data = await handler(message, entities, context)
        else:
            # Low confidence - ask for clarification
            response_data = await self._handle_clarification(
                message, intent_result, context
            )
        
        # Generate natural language response
        if streaming:
            async for chunk in self._stream_response(response_data, memory):
                yield chunk
        else:
            response = await self.response_generator.generate(
                response_data,
                context
            )
            
            # Add to memory
            memory.chat_memory.add_ai_message(response)
            
            return {
                "response": response,
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "metadata": response_data.get("metadata", {})
            }
    
    async def _handle_process_document(
        self,
        message: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle document processing request"""
        # Extract document reference
        document_ref = entities.get("document_reference")
        
        if not document_ref:
            return {
                "type": "clarification_needed",
                "message": "I'd be happy to process a document for you. Could you please specify which document you'd like me to process?",
                "suggestions": [
                    "Upload a document",
                    "Provide document ID",
                    "Describe the document"
                ]
            }
        
        # Communicate with coordinator to start processing
        processing_result = await self.communicate(
            recipient="coordinator_001",
            message_type="process_document",
            content={
                "document_id": document_ref,
                "user_id": context["user_id"],
                "priority": self._determine_priority(entities)
            }
        )
        
        return {
            "type": "processing_started",
            "message": f"I've started processing your document. The current status is: {processing_result.get('status')}",
            "details": {
                "document_id": document_ref,
                "workflow_id": processing_result.get("workflow_id"),
                "estimated_time": processing_result.get("estimated_time", "2-5 minutes")
            },
            "metadata": {
                "show_progress": True,
                "track_workflow": processing_result.get("workflow_id")
            }
        }
    
    async def _handle_check_status(
        self,
        message: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check processing status"""
        # Get relevant IDs
        document_id = entities.get("document_reference")
        workflow_id = entities.get("workflow_reference")
        
        if not document_id and not workflow_id:
            # Get recent documents for user
            recent = await self._get_user_recent_documents(context["user_id"])
            
            if recent:
                return {
                    "type": "status_summary",
                    "message": "Here's the status of your recent documents:",
                    "documents": recent,
                    "metadata": {
                        "show_list": True
                    }
                }
            else:
                return {
                    "type": "no_documents",
                    "message": "You don't have any documents being processed currently."
                }
        
        # Query specific status
        status = await self._query_document_status(
            document_id or workflow_id
        )
        
        return {
            "type": "status_update",
            "message": self._format_status_message(status),
            "details": status,
            "metadata": {
                "show_timeline": True,
                "highlight_current": status.get("current_stage")
            }
        }
    
    async def _handle_get_insights(
        self,
        message: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide insights about processed documents"""
        # Determine insight type
        insight_type = entities.get("insight_type", "general")
        time_range = entities.get("time_range", "last_week")
        
        # Query analytics agent
        insights = await self.communicate(
            recipient="analytics_001",
            message_type="get_insights",
            content={
                "user_id": context["user_id"],
                "type": insight_type,
                "time_range": time_range
            }
        )
        
        # Format insights for presentation
        formatted_insights = self._format_insights(insights)
        
        return {
            "type": "insights",
            "message": f"Here are your {insight_type} insights for {time_range}:",
            "insights": formatted_insights,
            "visualizations": insights.get("charts", []),
            "metadata": {
                "show_charts": True,
                "interactive": True
            }
        }
    
    async def _handle_provide_feedback(
        self,
        message: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle user feedback about extractions"""
        document_id = entities.get("document_reference")
        feedback_type = entities.get("feedback_type", "correction")
        feedback_content = entities.get("feedback_content")
        
        if not document_id:
            return {
                "type": "clarification_needed",
                "message": "Which document would you like to provide feedback for?",
                "recent_documents": await self._get_user_recent_documents(
                    context["user_id"]
                )
            }
        
        # Process feedback
        feedback_result = await self.communicate(
            recipient="feedback_processor_001",
            message_type="user_feedback",
            content={
                "document_id": document_id,
                "feedback_type": feedback_type,
                "content": feedback_content,
                "user_id": context["user_id"]
            }
        )
        
        return {
            "type": "feedback_received",
            "message": "Thank you for your feedback! I've recorded it and our system will learn from this to improve future extractions.",
            "impact": feedback_result.get("expected_improvement"),
            "metadata": {
                "show_appreciation": True
            }
        }
    
    async def _handle_ask_question(
        self,
        message: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle general questions about the system or documents"""
        question_topic = entities.get("topic", "general")
        
        # Use reasoning to formulate answer
        answer_context = {
            **context,
            "question": message,
            "topic": question_topic,
            "situation": "User asking a question about the system"
        }
        
        answer_decision = await self.think(answer_context)
        
        # Get relevant information
        if question_topic == "capabilities":
            info = self._get_system_capabilities()
        elif question_topic == "performance":
            info = await self._get_performance_metrics()
        elif question_topic == "help":
            info = self._get_help_topics()
        else:
            # General Q&A using LLM
            info = await self._general_qa(message, context)
        
        return {
            "type": "answer",
            "message": answer_decision.reasoning[-1],  # Use final reasoning as answer
            "supporting_info": info,
            "metadata": {
                "confidence": answer_decision.confidence,
                "sources": info.get("sources", [])
            }
        }
    
    async def _handle_system_health(
        self,
        message: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide system health information"""
        # Query coordinator for system status
        health = await self.communicate(
            recipient="coordinator_001",
            message_type="system_status",
            content={}
        )
        
        # Format health report
        health_summary = self._format_health_summary(health)
        
        return {
            "type": "system_health",
            "message": health_summary["summary"],
            "metrics": health_summary["metrics"],
            "alerts": health_summary.get("alerts", []),
            "metadata": {
                "show_dashboard": True,
                "refresh_interval": 30
            }
        }
    
    def _format_status_message(self, status: Dict[str, Any]) -> str:
        """Format status into natural language"""
        stage = status.get("current_stage", "unknown")
        progress = status.get("progress", 0)
        
        stage_messages = {
            "extraction": f"Currently extracting content from your document ({progress}% complete)",
            "analysis": f"Analyzing the extracted data for quality and anomalies",
            "review": f"Document is under human review due to some uncertainties",
            "completed": f"Processing complete! Your document has been successfully processed",
            "failed": f"There was an issue processing your document"
        }
        
        base_message = stage_messages.get(stage, f"Document is in {stage} stage")
        
        if status.get("anomaly_detected"):
            base_message += ". Some unusual patterns were detected and are being investigated"
        
        if status.get("estimated_completion"):
            base_message += f". Estimated completion: {status['estimated_completion']}"
        
        return base_message
    
    async def _stream_response(
        self,
        response_data: Dict[str, Any],
        memory: ConversationSummaryBufferMemory
    ) -> AsyncGenerator[str, None]:
        """Stream response token by token"""
        # Generate response with streaming
        prompt = self.response_generator.create_prompt(response_data)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Stream generation
        streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
        
        with torch.no_grad():
            generated = self.llm.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                streamer=streamer
            )
        
        # Get full response for memory
        full_response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        memory.chat_memory.add_ai_message(full_response)
    
    async def proactive_check_in(
        self,
        user_id: str
    ) -> Optional[str]:
        """Proactively check in with user about their documents"""
        # Check if user has documents that need attention
        user_documents = await self._get_user_documents_needing_attention(user_id)
        
        if user_documents:
            context = {
                "user_id": user_id,
                "documents_needing_attention": user_documents,
                "situation": "Proactive check-in about documents"
            }
            
            decision = await self.think(context)
            
            if decision.confidence > 0.7:
                return self._generate_proactive_message(
                    user_documents,
                    decision.action
                )
        
        return None
    
    def _generate_proactive_message(
        self,
        documents: List[Dict[str, Any]],
        action: str
    ) -> str:
        """Generate proactive message for user"""
        if action == "notify_completion":
            return (
                f"Good news! Your document processing is complete. "
                f"{len(documents)} document(s) are ready for review."
            )
        elif action == "request_feedback":
            return (
                f"I noticed some of your recently processed documents might benefit "
                f"from your feedback. Would you like to review them?"
            )
        elif action == "suggest_improvement":
            return (
                f"Based on your document processing patterns, I have some suggestions "
                f"that could improve extraction accuracy. Interested?"
            )
        else:
            return (
                f"You have {len(documents)} document(s) that might need your attention."
            )
    
    async def _handle_message(
        self,
        message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
        """Handle messages from other agents"""
        if message.message_type == "user_query":
            # Process user query
            response = await self.chat(
                user_id=message.content["user_id"],
                message=message.content["message"]
            )
            return response
        
        elif message.message_type == "notification":
            # Store notification for user
            await self._store_user_notification(
                message.content["user_id"],
                message.content["notification"]
            )
            return {"status": "notification_stored"}
        
        return None
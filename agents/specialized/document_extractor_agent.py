"""
Specialized agent for intelligent document extraction with reasoning capabilities.
"""

from typing import Dict, List, Any, Optional
import asyncio

from langchain.tools import Tool

from agents.core.base_agent import BaseAgent, AgentRole, AgentDecision
from core.extraction.engine import DocumentExtractionEngine
from utils.document_analyzer import DocumentAnalyzer


class DocumentExtractorAgent(BaseAgent):
    """
    Autonomous agent specialized in document extraction.
    Reasons about document structure and extraction strategies.
    """
    
    def __init__(self, agent_id: str = "extractor_001"):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXTRACTOR
        )
        
        # Initialize extraction engine
        self.extraction_engine = DocumentExtractionEngine()
        self.document_analyzer = DocumentAnalyzer()
        
        # Extraction strategies
        self.extraction_strategies = {
            "standard": self._standard_extraction,
            "complex": self._complex_extraction,
            "multi_page": self._multi_page_extraction,
            "table_focused": self._table_extraction,
            "form_based": self._form_extraction
        }
        
    def _init_tools(self) -> List[Tool]:
        """Initialize extraction-specific tools"""
        return [
            Tool(
                name="analyze_document_structure",
                func=self.document_analyzer.analyze_structure,
                description="Analyze document structure to determine best extraction approach"
            ),
            Tool(
                name="extract_with_strategy",
                func=self._extract_with_strategy,
                description="Extract content using a specific strategy"
            ),
            Tool(
                name="validate_extraction",
                func=self._validate_extraction,
                description="Validate extracted data for completeness and accuracy"
            ),
            Tool(
                name="request_human_guidance",
                func=self._request_human_guidance,
                description="Request guidance from human expert for complex cases"
            )
        ]
    
    async def process_document(
        self,
        document_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Autonomously process a document with intelligent decision-making.
        """
        # Analyze document first
        analysis = await self.document_analyzer.analyze_structure(document_id)
        
        # Create context for reasoning
        extraction_context = {
            "document_id": document_id,
            "document_type": analysis.get("type", "unknown"),
            "complexity": analysis.get("complexity", "medium"),
            "special_features": analysis.get("features", []),
            "previous_attempts": context.get("previous_attempts", 0),
            "situation": f"Need to extract data from a {analysis.get('type', 'unknown')} document"
        }
        
        # Think about the best approach
        decision = await self.think(extraction_context)
        
        # Execute extraction based on decision
        if decision.confidence > 0.8:
            result = await self._execute_extraction(
                document_id,
                decision.action,
                extraction_context
            )
        else:
            # Low confidence - consult with other agents
            guidance = await self.communicate(
                recipient="analyzer_001",
                message_type="extraction_guidance_request",
                content={
                    "document_id": document_id,
                    "analysis": analysis,
                    "proposed_action": decision.action,
                    "confidence": decision.confidence,
                    "alternatives": decision.alternatives
                },
                priority=7
            )
            
            # Re-evaluate with guidance
            extraction_context["guidance"] = guidance
            new_decision = await self.think(extraction_context)
            
            result = await self._execute_extraction(
                document_id,
                new_decision.action,
                extraction_context
            )
        
        # Learn from the extraction outcome
        await self._evaluate_and_learn(result, decision)
        
        return result
    
    async def _execute_extraction(
        self,
        document_id: str,
        strategy: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute extraction with chosen strategy"""
        strategy_func = self.extraction_strategies.get(
            strategy,
            self._standard_extraction
        )
        
        return await strategy_func(document_id, context)
    
    async def _standard_extraction(
        self,
        document_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Standard extraction approach"""
        return await self.extraction_engine.process_document(
            document_id=document_id,
            quarter=context.get("quarter", "Q1")
        )
    
    async def _complex_extraction(
        self,
        document_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complex extraction with multiple passes"""
        # First pass - general extraction
        initial_result = await self._standard_extraction(document_id, context)
        
        # Analyze gaps
        gaps = await self._identify_extraction_gaps(initial_result)
        
        if gaps:
            # Second pass - targeted extraction
            targeted_result = await self._targeted_extraction(
                document_id,
                gaps,
                initial_result
            )
            
            # Merge results
            final_result = self._merge_extraction_results(
                initial_result,
                targeted_result
            )
            
            return final_result
        
        return initial_result
    
    async def _evaluate_and_learn(
        self,
        result: Dict[str, Any],
        decision: AgentDecision
    ):
        """Evaluate extraction outcome and learn"""
        # Calculate extraction quality
        quality_score = await self._calculate_quality_score(result)
        
        # Prepare feedback
        feedback = {
            "outcome": "success" if quality_score > 0.8 else "needs_improvement",
            "quality_score": quality_score,
            "decision": decision.action,
            "confidence": decision.confidence,
            "actual_complexity": result.get("complexity_score", 0.5)
        }
        
        # Learn from feedback
        await self.learn_from_feedback(feedback)
        
        # If quality is low, notify coordinator
        if quality_score < 0.7:
            await self.communicate(
                recipient="coordinator_001",
                message_type="low_quality_extraction",
                content={
                    "document_id": result["document_id"],
                    "quality_score": quality_score,
                    "issues": result.get("issues", [])
                },
                priority=6
            )
    
    async def _handle_message(
        self,
        message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
        """Handle incoming messages"""
        if message.message_type == "extraction_request":
            # Process extraction request
            result = await self.process_document(
                document_id=message.content["document_id"],
                context=message.content.get("context")
            )
            return {"status": "completed", "result": result}
        
        elif message.message_type == "strategy_update":
            # Update extraction strategies based on feedback
            self._update_strategies(message.content)
            return {"status": "acknowledged"}
        
        return None
    
    async def _perform_role_tasks(self):
        """Perform extractor-specific background tasks"""
        # Check for extraction performance trends
        if self.decisions_made % 50 == 0:
            await self._analyze_performance_trends()


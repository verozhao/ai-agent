"""
Specialized agent for anomaly analysis with advanced reasoning.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from agents.core.base_agent import BaseAgent, AgentRole
from core.anomaly.detector import AnomalyDetector


class AnomalyAnalyzerAgent(BaseAgent):
    """
    Agent specialized in analyzing anomalies and making routing decisions.
    Uses reasoning to understand why documents are anomalous.
    """
    
    def __init__(self, agent_id: str = "analyzer_001"):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ANALYZER
        )
        
        self.anomaly_detector = AnomalyDetector()
        self.anomaly_patterns = {}
        
    def _init_tools(self) -> List[Tool]:
        """Initialize anomaly analysis tools"""
        return [
            Tool(
                name="deep_anomaly_analysis",
                func=self._deep_anomaly_analysis,
                description="Perform deep analysis of anomaly patterns"
            ),
            Tool(
                name="compare_similar_cases",
                func=self._compare_similar_cases,
                description="Compare with similar anomaly cases"
            ),
            Tool(
                name="predict_resolution",
                func=self._predict_resolution,
                description="Predict best resolution approach"
            )
        ]
    
    async def analyze_anomaly(
        self,
        extraction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze an anomalous extraction with reasoning.
        """
        # Create analysis context
        context = {
            "extraction_id": extraction_result["extraction_id"],
            "anomaly_score": extraction_result["anomaly_score"],
            "confidence_scores": extraction_result["confidence_scores"],
            "document_type": extraction_result.get("document_type"),
            "situation": "Analyzing document with high anomaly score"
        }
        
        # Reason about the anomaly
        decision = await self.think(context)
        
        # Perform deep analysis based on decision
        if decision.action == "investigate_pattern":
            analysis = await self._investigate_anomaly_pattern(extraction_result)
        elif decision.action == "compare_historical":
            analysis = await self._compare_historical_anomalies(extraction_result)
        elif decision.action == "request_expert_review":
            analysis = await self._request_expert_review(extraction_result)
        else:
            analysis = await self._standard_anomaly_analysis(extraction_result)
        
        # Determine routing recommendation
        routing = await self._determine_routing(analysis, decision)
        
        # Communicate findings
        await self.communicate(
            recipient="coordinator_001",
            message_type="anomaly_analysis_complete",
            content={
                "extraction_id": extraction_result["extraction_id"],
                "analysis": analysis,
                "routing_recommendation": routing,
                "confidence": decision.confidence
            },
            priority=8 if routing["action"] == "eval_set_2" else 5
        )
        
        return {
            "analysis": analysis,
            "routing": routing,
            "reasoning": decision.reasoning
        }
    
    async def _investigate_anomaly_pattern(
        self,
        extraction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Investigate specific anomaly patterns"""
        # Extract anomaly features
        features = self._extract_anomaly_features(extraction_result)
        
        # Look for known patterns
        matched_patterns = []
        for pattern_id, pattern in self.anomaly_patterns.items():
            similarity = self._calculate_pattern_similarity(features, pattern)
            if similarity > 0.7:
                matched_patterns.append({
                    "pattern_id": pattern_id,
                    "similarity": similarity,
                    "description": pattern["description"],
                    "typical_cause": pattern["cause"],
                    "resolution": pattern["resolution"]
                })
        
        # If no known pattern, create new one
        if not matched_patterns:
            new_pattern_id = f"pattern_{len(self.anomaly_patterns)}"
            self.anomaly_patterns[new_pattern_id] = {
                "features": features,
                "description": await self._generate_pattern_description(features),
                "cause": "unknown",
                "resolution": "requires_investigation"
            }
            
            matched_patterns = [{
                "pattern_id": new_pattern_id,
                "similarity": 1.0,
                "description": self.anomaly_patterns[new_pattern_id]["description"],
                "typical_cause": "New pattern - investigation needed",
                "resolution": "eval_set_2"
            }]
        
        return {
            "matched_patterns": matched_patterns,
            "features": features,
            "pattern_based_routing": matched_patterns[0]["resolution"]
        }
    
    async def _determine_routing(
        self,
        analysis: Dict[str, Any],
        decision: AgentDecision
    ) -> Dict[str, Any]:
        """Determine routing based on analysis and reasoning"""
        # Calculate routing confidence
        factors = {
            "anomaly_severity": analysis.get("severity_score", 0.5),
            "pattern_match": len(analysis.get("matched_patterns", [])) > 0,
            "decision_confidence": decision.confidence,
            "historical_success": self.success_rate
        }
        
        # Weighted routing decision
        if factors["anomaly_severity"] > 0.8 and factors["decision_confidence"] < 0.7:
            return {
                "action": "eval_set_2",
                "priority": "high",
                "reason": "High severity anomaly with uncertain resolution"
            }
        elif factors["pattern_match"] and factors["historical_success"] > 0.85:
            return {
                "action": "auto_correct",
                "priority": "medium",
                "reason": "Known pattern with high success rate"
            }
        else:
            return {
                "action": "human_review",
                "priority": "medium",
                "reason": "Requires human judgment"
            }
    
    async def _handle_message(
        self,
        message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
        """Handle analyzer-specific messages"""
        if message.message_type == "anomaly_analysis_request":
            result = await self.analyze_anomaly(message.content["extraction_result"])
            return {"analysis": result}
        
        elif message.message_type == "pattern_update":
            self._update_anomaly_patterns(message.content["patterns"])
            return {"status": "patterns_updated"}
        
        return None
"""
Interface for human experts to collaborate with AI agents.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import asyncio
from dataclasses import dataclass

from fastapi import WebSocket
import json


@dataclass
class ExpertSession:
    """Expert session information"""
    expert_id: str
    expertise_areas: List[str]
    availability_status: str
    current_tasks: List[str]
    performance_rating: float


class HumanExpertInterface:
    """
    Interface enabling seamless collaboration between human experts
    and AI agents for complex decision-making.
    """
    
    def __init__(self):
        self.active_experts = {}
        self.expert_queues = {}
        self.collaboration_sessions = {}
        
        # Expertise matching
        self.expertise_registry = {
            "financial_documents": ["invoice", "receipt", "statement"],
            "legal_documents": ["contract", "agreement", "terms"],
            "medical_records": ["prescription", "report", "chart"],
            "technical_specs": ["blueprint", "schematic", "manual"]
        }
    
    async def register_expert(
        self,
        expert_id: str,
        expertise_areas: List[str],
        websocket: WebSocket
    ):
        """Register a human expert for collaboration"""
        self.active_experts[expert_id] = ExpertSession(
            expert_id=expert_id,
            expertise_areas=expertise_areas,
            availability_status="available",
            current_tasks=[],
            performance_rating=1.0
        )
        
        # Create task queue for expert
        self.expert_queues[expert_id] = asyncio.Queue()
        
        # Start expert session handler
        asyncio.create_task(
            self._handle_expert_session(expert_id, websocket)
        )
        
        logger.info(f"Expert {expert_id} registered with expertise: {expertise_areas}")
    
    async def request_expert_help(
        self,
        task_type: str,
        document_id: str,
        context: Dict[str, Any],
        priority: int = 5,
        requester_agent: str = None
    ) -> Dict[str, Any]:
        """Request help from human expert"""
        # Find suitable expert
        expert = await self._find_suitable_expert(task_type, context)
        
        if not expert:
            return {
                "status": "no_expert_available",
                "message": "No suitable expert currently available"
            }
        
        # Create collaboration task
        task = {
            "id": f"task_{document_id}_{datetime.utcnow().timestamp()}",
            "type": task_type,
            "document_id": document_id,
            "context": context,
            "priority": priority,
            "requester": requester_agent,
            "created_at": datetime.utcnow()
        }
        
        # Add to expert's queue
        await self.expert_queues[expert.expert_id].put(task)
        
        # Wait for expert response (with timeout)
        try:
            response = await asyncio.wait_for(
                self._wait_for_expert_response(task["id"]),
                timeout=300  # 5 minute timeout
            )
            
            return {
                "status": "completed",
                "expert_id": expert.expert_id,
                "response": response
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "message": "Expert response timeout"
            }
    
    async def _find_suitable_expert(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> Optional[ExpertSession]:
        """Find the most suitable available expert"""
        available_experts = [
            expert for expert in self.active_experts.values()
            if expert.availability_status == "available"
        ]
        
        if not available_experts:
            return None
        
        # Score experts based on suitability
        expert_scores = []
        for expert in available_experts:
            score = self._calculate_expert_score(expert, task_type, context)
            expert_scores.append((expert, score))
        
        # Select highest scoring expert
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        best_expert = expert_scores[0][0]
        
        # Mark as busy
        best_expert.availability_status = "busy"
        
        return best_expert
    
    def _calculate_expert_score(
        self,
        expert: ExpertSession,
        task_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate expert suitability score"""
        score = 0.0
        
        # Check expertise match
        for area in expert.expertise_areas:
            if area in self.expertise_registry:
                matching_types = self.expertise_registry[area]
                if any(t in task_type.lower() for t in matching_types):
                    score += 0.5
        
        # Factor in performance rating
        score += expert.performance_rating * 0.3
        
        # Factor in current workload
        workload_penalty = len(expert.current_tasks) * 0.1
        score -= workload_penalty
        
        return max(0, score)
    
    async def _handle_expert_session(
        self,
        expert_id: str,
        websocket: WebSocket
    ):
        """Handle expert's WebSocket session"""
        await websocket.accept()
        
        try:
            # Send initial status
            await websocket.send_json({
                "type": "connected",
                "expert_id": expert_id,
                "message": "Connected to AI agent system"
            })
            
            # Create task handler
            task_handler = asyncio.create_task(
                self._send_tasks_to_expert(expert_id, websocket)
            )
            
            # Handle incoming messages
            while True:
                data = await websocket.receive_json()
                await self._process_expert_message(expert_id, data)
                
        except Exception as e:
            logger.error(f"Expert session error: {str(e)}")
        finally:
            # Cleanup
            task_handler.cancel()
            del self.active_experts[expert_id]
            del self.expert_queues[expert_id]
            await websocket.close()
    
    async def _send_tasks_to_expert(
        self,
        expert_id: str,
        websocket: WebSocket
    ):
        """Send queued tasks to expert"""
        queue = self.expert_queues[expert_id]
        
        while True:
            try:
                # Get next task
                task = await queue.get()
                
                # Update expert status
                expert = self.active_experts[expert_id]
                expert.current_tasks.append(task["id"])
                
                # Send to expert
                await websocket.send_json({
                    "type": "new_task",
                    "task": task
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending task to expert: {str(e)}")
    
    async def _process_expert_message(
        self,
        expert_id: str,
        message: Dict[str, Any]
    ):
        """Process message from expert"""
        msg_type = message.get("type")
        
        if msg_type == "task_response":
            # Expert completed a task
            task_id = message["task_id"]
            response = message["response"]
            
            # Store response
            if task_id in self.collaboration_sessions:
                self.collaboration_sessions[task_id] = response
            
            # Update expert status
            expert = self.active_experts[expert_id]
            expert.current_tasks.remove(task_id)
            if not expert.current_tasks:
                expert.availability_status = "available"
        
        elif msg_type == "status_update":
            # Update expert status
            expert = self.active_experts[expert_id]
            expert.availability_status = message.get("status", "available")
        
        elif msg_type == "expertise_update":
            # Update expert's expertise areas
            expert = self.active_experts[expert_id]
            expert.expertise_areas = message.get("expertise_areas", [])
    
    async def _wait_for_expert_response(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """Wait for expert to complete task"""
        while task_id not in self.collaboration_sessions:
            await asyncio.sleep(0.5)
        
        response = self.collaboration_sessions[task_id]
        del self.collaboration_sessions[task_id]
        
        return response
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get statistics about expert collaboration"""
        total_experts = len(self.active_experts)
        available_experts = sum(
            1 for e in self.active_experts.values()
            if e.availability_status == "available"
        )
        
        expertise_coverage = {}
        for expert in self.active_experts.values():
            for area in expert.expertise_areas:
                expertise_coverage[area] = expertise_coverage.get(area, 0) + 1
        
        avg_rating = (
            sum(e.performance_rating for e in self.active_experts.values()) / 
            total_experts if total_experts > 0 else 0
        )
        
        return {
            "total_experts": total_experts,
            "available_experts": available_experts,
            "expertise_coverage": expertise_coverage,
            "average_rating": avg_rating,
            "active_tasks": sum(
                len(e.current_tasks) for e in self.active_experts.values()
            )
        }
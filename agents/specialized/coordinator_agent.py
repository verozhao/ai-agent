"""
Master coordinator agent that orchestrates the entire system.
"""

from typing import Dict, List, Any, Optional
import asyncio
from collections import defaultdict

from agents.core.base_agent import BaseAgent, AgentRole, AgentState


class CoordinatorAgent(BaseAgent):
    """
    Master coordinator that orchestrates all other agents and makes
    high-level decisions about document processing workflows.
    """
    
    def __init__(self, agent_id: str = "coordinator_001"):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COORDINATOR
        )
        
        # Track all agents in the system
        self.registered_agents = {}
        self.agent_workloads = defaultdict(int)
        self.processing_queues = defaultdict(list)
        
        # Workflow templates
        self.workflows = {
            "standard": self._standard_workflow,
            "high_priority": self._high_priority_workflow,
            "bulk_processing": self._bulk_processing_workflow,
            "anomaly_investigation": self._anomaly_workflow
        }
        
        # System metrics
        self.system_metrics = {
            "total_processed": 0,
            "success_rate": 1.0,
            "avg_processing_time": 0,
            "active_workflows": 0
        }
        
    def _init_tools(self) -> List[Tool]:
        """Initialize coordination tools"""
        return [
            Tool(
                name="assign_task",
                func=self._assign_task_to_agent,
                description="Assign task to most suitable agent"
            ),
            Tool(
                name="monitor_system_health",
                func=self._monitor_system_health,
                description="Monitor overall system health"
            ),
            Tool(
                name="optimize_workflow",
                func=self._optimize_workflow,
                description="Optimize document processing workflow"
            ),
            Tool(
                name="escalate_issue",
                func=self._escalate_issue,
                description="Escalate critical issues to humans"
            )
        ]
    
    async def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        self.registered_agents[agent.agent_id] = {
            "agent": agent,
            "role": agent.role,
            "status": AgentState.IDLE,
            "current_task": None,
            "performance": agent.success_rate
        }
        
        logger.info(f"Registered agent {agent.agent_id} with role {agent.role.value}")
    
    async def process_document_request(
        self,
        document_id: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document request by orchestrating the workflow.
        """
        # Create processing context
        context = {
            "document_id": document_id,
            "priority": priority,
            "metadata": metadata or {},
            "situation": f"New document processing request with priority {priority}"
        }
        
        # Decide on workflow
        workflow_decision = await self.think(context)
        
        # Execute chosen workflow
        workflow_func = self.workflows.get(
            workflow_decision.action,
            self._standard_workflow
        )
        
        # Start workflow with monitoring
        workflow_id = f"wf_{document_id}_{datetime.utcnow().timestamp()}"
        
        result = await self._execute_monitored_workflow(
            workflow_id,
            workflow_func,
            document_id,
            context
        )
        
        # Update metrics
        self.system_metrics["total_processed"] += 1
        
        return result
    
    async def _execute_monitored_workflow(
        self,
        workflow_id: str,
        workflow_func,
        document_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow with monitoring and error handling"""
        self.system_metrics["active_workflows"] += 1
        start_time = datetime.utcnow()
        
        try:
            result = await workflow_func(document_id, context)
            
            # Calculate success
            success = result.get("status") == "completed"
            self.system_metrics["success_rate"] = (
                self.system_metrics["success_rate"] * 0.99 + 
                (1.0 if success else 0.0) * 0.01
            )
            
            # Update timing
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.system_metrics["avg_processing_time"] = (
                self.system_metrics["avg_processing_time"] * 0.9 +
                processing_time * 0.1
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            
            # Decide on recovery strategy
            recovery_context = {
                **context,
                "error": str(e),
                "workflow_id": workflow_id,
                "situation": "Workflow failed, need recovery strategy"
            }
            
            recovery_decision = await self.think(recovery_context)
            
            if recovery_decision.action == "retry":
                return await self._execute_monitored_workflow(
                    workflow_id + "_retry",
                    workflow_func,
                    document_id,
                    context
                )
            else:
                await self._escalate_issue({
                    "workflow_id": workflow_id,
                    "error": str(e),
                    "document_id": document_id
                })
                
                return {
                    "status": "failed",
                    "error": str(e),
                    "escalated": True
                }
        
        finally:
            self.system_metrics["active_workflows"] -= 1
    
    async def _standard_workflow(
        self,
        document_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Standard document processing workflow"""
        # Step 1: Assign to extractor
        extractor = await self._select_best_agent(AgentRole.EXTRACTOR)
        
        extraction_result = await self._assign_and_wait(
            extractor,
            "extraction_request",
            {"document_id": document_id, "context": context}
        )
        
        # Step 2: Check for anomalies
        if extraction_result.get("anomaly_score", 0) > 0.5:
            analyzer = await self._select_best_agent(AgentRole.ANALYZER)
            
            analysis_result = await self._assign_and_wait(
                analyzer,
                "anomaly_analysis_request",
                {"extraction_result": extraction_result}
            )
            
            # Step 3: Route based on analysis
            if analysis_result["routing"]["action"] == "eval_set_2":
                auditor = await self._select_best_agent(AgentRole.AUDITOR)
                
                audit_result = await self._assign_and_wait(
                    auditor,
                    "audit_request",
                    {
                        "extraction_result": extraction_result,
                        "analysis": analysis_result
                    }
                )
                
                return {
                    "status": "completed",
                    "workflow": "standard_with_audit",
                    "extraction": extraction_result,
                    "analysis": analysis_result,
                    "audit": audit_result
                }
        
        return {
            "status": "completed",
            "workflow": "standard",
            "extraction": extraction_result
        }
    
    async def _select_best_agent(
        self,
        role: AgentRole
    ) -> str:
        """Select the best available agent for a role"""
        candidates = [
            agent_id for agent_id, info in self.registered_agents.items()
            if info["role"] == role and info["status"] == AgentState.IDLE
        ]
        
        if not candidates:
            # All agents busy, select least loaded
            candidates = [
                agent_id for agent_id, info in self.registered_agents.items()
                if info["role"] == role
            ]
            
            # Sort by workload
            candidates.sort(key=lambda x: self.agent_workloads[x])
        
        # Select agent with best performance
        best_agent = max(
            candidates,
            key=lambda x: self.registered_agents[x]["performance"]
        )
        
        return best_agent
    
    async def _assign_and_wait(
        self,
        agent_id: str,
        message_type: str,
        content: Dict[str, Any],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Assign task to agent and wait for completion"""
        # Update agent status
        self.registered_agents[agent_id]["status"] = AgentState.PROCESSING
        self.registered_agents[agent_id]["current_task"] = content.get("document_id")
        self.agent_workloads[agent_id] += 1
        
        # Send task
        response = await self.communicate(
            recipient=agent_id,
            message_type=message_type,
            content=content,
            priority=content.get("priority", 5)
        )
        
        # Update status
        self.registered_agents[agent_id]["status"] = AgentState.IDLE
        self.registered_agents[agent_id]["current_task"] = None
        
        return response
    
    async def _handle_message(
        self,
        message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
        """Handle coordinator-specific messages"""
        if message.message_type == "process_document":
            result = await self.process_document_request(
                document_id=message.content["document_id"],
                priority=message.content.get("priority", 5),
                metadata=message.content.get("metadata")
            )
            return result
        
        elif message.message_type == "system_status":
            return {
                "agents": len(self.registered_agents),
                "active_workflows": self.system_metrics["active_workflows"],
                "success_rate": self.system_metrics["success_rate"],
                "avg_processing_time": self.system_metrics["avg_processing_time"]
            }
        
        elif message.message_type == "emergency_stop":
            await self._emergency_stop()
            return {"status": "stopped"}
        
        return None
    
    async def _monitor_system_health(self) -> Dict[str, Any]:
        """Monitor overall system health"""
        health_metrics = {
            "overall_health": "healthy",
            "issues": []
        }
        
        # Check success rate
        if self.system_metrics["success_rate"] < 0.8:
            health_metrics["issues"].append({
                "type": "low_success_rate",
                "value": self.system_metrics["success_rate"],
                "severity": "high"
            })
            health_metrics["overall_health"] = "degraded"
        
        # Check agent availability
        idle_agents = sum(
            1 for info in self.registered_agents.values()
            if info["status"] == AgentState.IDLE
        )
        
        if idle_agents == 0:
            health_metrics["issues"].append({
                "type": "no_idle_agents",
                "severity": "medium"
            })
        
        # Check processing times
        if self.system_metrics["avg_processing_time"] > 300:  # 5 minutes
            health_metrics["issues"].append({
                "type": "slow_processing",
                "value": self.system_metrics["avg_processing_time"],
                "severity": "medium"
            })
        
        return health_metrics
    
    async def _perform_role_tasks(self):
        """Perform coordinator-specific background tasks"""
        # Periodic health check
        if self.decisions_made % 10 == 0:
            health = await self._monitor_system_health()
            
            if health["overall_health"] != "healthy":
                # Take corrective action
                await self._optimize_system(health["issues"])
        
        # Rebalance workloads
        if self.decisions_made % 50 == 0:
            await self._rebalance_workloads()

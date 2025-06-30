"""
Main orchestration and deployment system for the AI agent ecosystem.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import signal
import sys

from agents.core.base_agent import AgentRole
from agents.specialized.document_extractor_agent import DocumentExtractorAgent
from agents.specialized.anomaly_analyzer_agent import AnomalyAnalyzerAgent
from agents.specialized.coordinator_agent import CoordinatorAgent
from agents.interface.conversational_agent import ConversationalAgent
from agents.communication.inter_agent_protocol import InterAgentProtocol
from agents.monitoring.agent_monitor import AgentMonitor
from config.agent_config import AgentSystemConfig


logger = logging.getLogger(__name__)


class AIAgentSystem:
    """
    Master orchestration system for launching and managing
    the entire AI agent ecosystem.
    """
    
    def __init__(self, config: AgentSystemConfig):
        self.config = config
        self.agents = {}
        self.running = False
        
        # Communication protocol
        self.protocol = InterAgentProtocol(
            redis_url=config.redis_url,
            encryption_key=config.encryption_key
        )
        
        # System monitor
        self.monitor = AgentMonitor()
        
        # Agent tasks
        self.agent_tasks = []
        
    async def initialize(self):
        """Initialize the AI agent system"""
        logger.info("Initializing AI Agent System...")
        
        # Connect communication protocol
        await self.protocol.connect()
        
        # Create agents based on configuration
        await self._create_agents()
        
        # Register agents with coordinator
        await self._register_agents()
        
        # Setup inter-agent communication
        await self._setup_communication()
        
        logger.info("AI Agent System initialized successfully")
    
    async def _create_agents(self):
        """Create all configured agents"""
        # Create coordinator first
        coordinator = CoordinatorAgent(agent_id="coordinator_001")
        self.agents["coordinator_001"] = coordinator
        
        # Create specialized agents
        for i in range(self.config.num_extractors):
            agent_id = f"extractor_{i:03d}"
            self.agents[agent_id] = DocumentExtractorAgent(agent_id)
        
        for i in range(self.config.num_analyzers):
            agent_id = f"analyzer_{i:03d}"
            self.agents[agent_id] = AnomalyAnalyzerAgent(agent_id)
        
        # Create interface agents
        self.agents["conversational_001"] = ConversationalAgent()
        
        logger.info(f"Created {len(self.agents)} agents")
    
    async def _register_agents(self):
        """Register all agents with the coordinator"""
        coordinator = self.agents["coordinator_001"]
        
        for agent_id, agent in self.agents.items():
            if agent_id != "coordinator_001":
                await coordinator.register_agent(agent)
    
    async def _setup_communication(self):
        """Setup inter-agent communication channels"""
        for agent_id, agent in self.agents.items():
            # Set communication protocol
            agent.communicator.protocol = self.protocol
            
            # Subscribe to messages
            asyncio.create_task(
                self.protocol.receive_messages(
                    agent_id,
                    agent.process_message
                )
            )
    
    async def start(self):
        """Start the AI agent system"""
        logger.info("Starting AI Agent System...")
        self.running = True
        
        # Start all agents
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(agent.run())
            self.agent_tasks.append(task)
            logger.info(f"Started agent {agent_id}")
        
        # Start system monitor
        monitor_task = asyncio.create_task(self._monitor_system())
        self.agent_tasks.append(monitor_task)
        
        # Start health check endpoint
        health_task = asyncio.create_task(self._health_check_server())
        self.agent_tasks.append(health_task)
        
        logger.info("AI Agent System started successfully")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.agent_tasks)
        except asyncio.CancelledError:
            logger.info("AI Agent System shutting down...")
    
    async def stop(self):
        """Stop the AI agent system gracefully"""
        logger.info("Stopping AI Agent System...")
        self.running = False
        
        # Cancel all agent tasks
        for task in self.agent_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.agent_tasks, return_exceptions=True)
        
        # Disconnect communication
        await self.protocol.disconnect()
        
        logger.info("AI Agent System stopped")
    
    async def _monitor_system(self):
        """Monitor system health and performance"""
        while self.running:
            try:
                # Collect agent metrics
                metrics = {}
                for agent_id, agent in self.agents.items():
                    metrics[agent_id] = {
                        "state": agent.state.value,
                        "decisions_made": agent.decisions_made,
                        "success_rate": agent.success_rate,
                        "message_queue_size": agent.message_queue.qsize()
                    }
                
                # Get coordinator metrics
                coordinator = self.agents["coordinator_001"]
                system_metrics = coordinator.system_metrics
                
                # Log system health
                await self.monitor.log_metrics({
                    "timestamp": datetime.utcnow(),
                    "agent_metrics": metrics,
                    "system_metrics": system_metrics
                })
                
                # Check for issues
                issues = await self._check_system_issues(metrics, system_metrics)
                if issues:
                    await self._handle_system_issues(issues)
                
                # Wait before next check
                await asyncio.sleep(self.config.monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _check_system_issues(
        self,
        agent_metrics: Dict[str, Any],
        system_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for system issues"""
        issues = []
        
        # Check agent health
        for agent_id, metrics in agent_metrics.items():
            if metrics["success_rate"] < 0.7:
                issues.append({
                    "type": "low_agent_success_rate",
                    "agent_id": agent_id,
                    "value": metrics["success_rate"],
                    "severity": "high"
                })
            
            if metrics["message_queue_size"] > 100:
                issues.append({
                    "type": "high_message_queue",
                    "agent_id": agent_id,
                    "value": metrics["message_queue_size"],
                    "severity": "medium"
                })
        
        # Check system metrics
        if system_metrics.get("success_rate", 1.0) < 0.8:
            issues.append({
                "type": "low_system_success_rate",
                "value": system_metrics["success_rate"],
                "severity": "high"
            })
        
        return issues
    
    async def _handle_system_issues(self, issues: List[Dict[str, Any]]):
        """Handle detected system issues"""
        for issue in issues:
            logger.warning(f"System issue detected: {issue}")
            
            if issue["severity"] == "high":
                # Take corrective action
                if issue["type"] == "low_agent_success_rate":
                    # Restart struggling agent
                    await self._restart_agent(issue["agent_id"])
                elif issue["type"] == "low_system_success_rate":
                    # Alert administrators
                    await self._alert_administrators(issue)
    
    async def _restart_agent(self, agent_id: str):
        """Restart a specific agent"""
        logger.info(f"Restarting agent {agent_id}")
        
        # Find and cancel agent task
        agent = self.agents[agent_id]
        
        # Reset agent state
        agent.success_rate = 1.0
        agent.decisions_made = 0
        agent.learning_episodes.clear()
        
        # Restart agent
        task = asyncio.create_task(agent.run())
        self.agent_tasks.append(task)
    
    async def _health_check_server(self):
        """Simple health check server for monitoring"""
        from aiohttp import web
        
        async def health_check(request):
            """Health check endpoint"""
            coordinator = self.agents.get("coordinator_001")
            
            if not coordinator:
                return web.json_response(
                    {"status": "unhealthy", "reason": "No coordinator"},
                    status=503
                )
            
            health = await coordinator._monitor_system_health()
            
            if health["overall_health"] == "healthy":
                return web.json_response(
                    {"status": "healthy", "agents": len(self.agents)},
                    status=200
                )
            else:
                return web.json_response(
                    {"status": health["overall_health"], "issues": health["issues"]},
                    status=503
                )
        
        app = web.Application()
        app.router.add_get("/health", health_check)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.config.health_check_port)
        await site.start()
    
    def handle_shutdown(self, sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}")
        asyncio.create_task(self.stop())

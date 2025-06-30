"""
Complete integration examples showing how to use the AI Agent System
in various real-world scenarios.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from agents.orchestration.system_launcher import AIAgentSystem
from agents.interface.conversational_agent import ConversationalAgent
from agents.interface.human_expert_interface import HumanExpertInterface
from config.agent_config import AgentSystemConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Basic Document Processing
# ============================================================================

async def example_basic_document_processing():
    """
    Example showing basic document processing with the AI agent system.
    """
    # Initialize the system
    config = AgentSystemConfig(
        num_extractors=3,
        num_analyzers=2,
        learning_enabled=True
    )
    
    agent_system = AIAgentSystem(config)
    await agent_system.initialize()
    
    # Start the system
    system_task = asyncio.create_task(agent_system.start())
    
    try:
        # Process a single document
        result = await agent_system.agents["coordinator_001"].process_document_request(
            document_id="invoice_2024_001.pdf",
            priority=5,
            metadata={
                "document_type": "invoice",
                "source": "email_attachment",
                "user_id": "user_123"
            }
        )
        
        print(f"Document processed: {result}")
        
        # Check processing status
        if result["status"] == "completed":
            print(f"Extraction successful!")
            print(f"Extracted data: {result['extraction']['extracted_data']}")
            print(f"Confidence: {result['extraction']['confidence']}")
            
            # Check if any anomalies were detected
            if "analysis" in result:
                print(f"Anomaly analysis: {result['analysis']}")
        
    finally:
        # Cleanup
        await agent_system.stop()
        system_task.cancel()


# ============================================================================
# EXAMPLE 2: Conversational Interface
# ============================================================================

async def example_conversational_interface():
    """
    Example showing how to interact with the system through natural language.
    """
    # Initialize conversational agent
    conversational_agent = ConversationalAgent()
    
    # Simulate user conversation
    user_id = "user_456"
    
    # Example 1: Process document through conversation
    response = await conversational_agent.chat(
        user_id=user_id,
        message="I need to process an important invoice from our supplier"
    )
    print(f"Agent: {response['response']}")
    
    # Example 2: Check status
    response = await conversational_agent.chat(
        user_id=user_id,
        message="How's the processing going?"
    )
    print(f"Agent: {response['response']}")
    
    # Example 3: Get insights
    response = await conversational_agent.chat(
        user_id=user_id,
        message="Can you show me patterns in my recent documents?"
    )
    print(f"Agent: {response['response']}")
    
    # Example 4: Provide feedback
    response = await conversational_agent.chat(
        user_id=user_id,
        message="The date extraction was wrong, it should be March 15, not March 5"
    )
    print(f"Agent: {response['response']}")


# ============================================================================
# EXAMPLE 3: Batch Processing with Learning
# ============================================================================

async def example_batch_processing_with_learning():
    """
    Example showing batch document processing with continuous learning.
    """
    config = AgentSystemConfig(
        num_extractors=5,
        num_analyzers=3,
        learning_enabled=True,
        min_feedback_for_learning=50
    )
    
    agent_system = AIAgentSystem(config)
    await agent_system.initialize()
    
    # Start system
    system_task = asyncio.create_task(agent_system.start())
    
    try:
        # Prepare batch of documents
        document_batch = [
            {"id": f"doc_{i:04d}.pdf", "type": "invoice", "quarter": "Q1"}
            for i in range(100)
        ]
        
        # Process batch
        coordinator = agent_system.agents["coordinator_001"]
        
        results = []
        for doc in document_batch:
            result = await coordinator.process_document_request(
                document_id=doc["id"],
                metadata=doc
            )
            results.append(result)
            
            # Simulate some documents needing human review
            if result.get("status") == "needs_review":
                # Provide feedback
                feedback = {
                    "extraction_id": result["extraction_id"],
                    "corrections": {
                        "date": {"original": "2024-01-01", "corrected": "2024-01-15"},
                        "amount": {"original": "1000", "corrected": "10000"}
                    },
                    "quality_score": 0.7
                }
                
                # Submit feedback to system
                await agent_system.agents["feedback_processor_001"].process_feedback(feedback)
        
        # Check if learning was triggered
        print(f"Processed {len(results)} documents")
        print(f"Learning triggered: {agent_system.learning_triggered}")
        
        # Get performance metrics
        metrics = await coordinator.get_quarterly_report("Q1")
        print(f"Quarterly performance: {metrics}")
        
    finally:
        await agent_system.stop()
        system_task.cancel()


# ============================================================================
# EXAMPLE 4: Human Expert Collaboration
# ============================================================================

async def example_human_expert_collaboration():
    """
    Example showing how human experts collaborate with AI agents.
    """
    # Initialize expert interface
    expert_interface = HumanExpertInterface()
    
    # Simulate expert registration
    expert_id = "expert_001"
    expertise_areas = ["financial_documents", "complex_tables"]
    
    # In real implementation, this would be a WebSocket connection
    class MockWebSocket:
        async def accept(self): pass
        async def send_json(self, data): print(f"Expert receives: {data}")
        async def receive_json(self): 
            # Simulate expert response
            await asyncio.sleep(2)
            return {
                "type": "task_response",
                "task_id": "task_001",
                "response": {
                    "decision": "approve_with_corrections",
                    "corrections": {"vendor_name": "ACME Corp"},
                    "confidence": 0.95,
                    "notes": "Vendor name was OCR error"
                }
            }
    
    # Register expert
    await expert_interface.register_expert(
        expert_id=expert_id,
        expertise_areas=expertise_areas,
        websocket=MockWebSocket()
    )
    
    # Request expert help for complex document
    result = await expert_interface.request_expert_help(
        task_type="complex_financial_analysis",
        document_id="complex_invoice_001.pdf",
        context={
            "extraction_confidence": 0.65,
            "anomaly_score": 0.8,
            "specific_issues": ["unclear_vendor_name", "complex_table_structure"]
        },
        priority=8
    )
    
    print(f"Expert collaboration result: {result}")


# ============================================================================
# EXAMPLE 5: Custom Agent Development
# ============================================================================

from agents.core.base_agent import BaseAgent, AgentRole

class CustomValidationAgent(BaseAgent):
    """
    Custom agent for specific validation requirements.
    """
    
    def __init__(self, agent_id: str = "validator_001"):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ANALYZER  # Using existing role
        )
        
        # Custom validation rules
        self.validation_rules = {
            "invoice": self._validate_invoice,
            "contract": self._validate_contract,
            "report": self._validate_report
        }
    
    def _init_tools(self) -> List[Tool]:
        """Initialize validation-specific tools"""
        return [
            Tool(
                name="validate_document",
                func=self._validate_document,
                description="Validate document against business rules"
            ),
            Tool(
                name="check_compliance",
                func=self._check_compliance,
                description="Check regulatory compliance"
            )
        ]
    
    async def validate_extraction(
        self,
        extraction_result: Dict[str, Any],
        document_type: str
    ) -> Dict[str, Any]:
        """
        Validate extraction results against business rules.
        """
        # Create validation context
        context = {
            "extraction": extraction_result,
            "document_type": document_type,
            "situation": f"Validating {document_type} extraction"
        }
        
        # Use reasoning to decide validation approach
        decision = await self.think(context)
        
        # Apply validation rules
        validator = self.validation_rules.get(
            document_type,
            self._generic_validation
        )
        
        validation_result = await validator(extraction_result)
        
        # Learn from validation outcomes
        if validation_result["issues_found"] > 0:
            await self.learn_from_feedback({
                "outcome": "validation_issues",
                "document_type": document_type,
                "issues": validation_result["issues"]
            })
        
        return validation_result
    
    async def _validate_invoice(
        self,
        extraction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate invoice-specific rules"""
        issues = []
        
        # Check required fields
        required_fields = ["invoice_number", "date", "total_amount", "vendor"]
        for field in required_fields:
            if field not in extraction or not extraction[field]:
                issues.append(f"Missing required field: {field}")
        
        # Validate amount
        if "total_amount" in extraction:
            try:
                amount = float(extraction["total_amount"])
                if amount <= 0:
                    issues.append("Invalid amount: must be positive")
                if amount > 1000000:
                    issues.append("Unusually high amount: requires review")
            except:
                issues.append("Invalid amount format")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "issues_found": len(issues),
            "confidence": 0.9 if len(issues) == 0 else 0.5
        }


# ============================================================================
# EXAMPLE 6: Production Deployment
# ============================================================================

async def example_production_deployment():
    """
    Example showing production-ready deployment with monitoring and scaling.
    """
    # Production configuration
    config = AgentSystemConfig(
        num_extractors=10,
        num_analyzers=5,
        num_auditors=3,
        learning_enabled=True,
        monitor_interval=30,
        health_check_port=8080,
        max_concurrent_tasks=1000,
        task_timeout=300
    )
    
    # Initialize system
    agent_system = AIAgentSystem(config)
    await agent_system.initialize()
    
    # Setup monitoring
    from agents.monitoring.prometheus_exporter import PrometheusExporter
    exporter = PrometheusExporter(port=9090)
    await exporter.start()
    
    # Start system
    system_task = asyncio.create_task(agent_system.start())
    
    try:
        # Monitor system health
        while True:
            # Get system metrics
            health = await agent_system.check_health()
            
            if health["status"] != "healthy":
                logger.warning(f"System unhealthy: {health}")
                
                # Auto-scaling based on load
                if health["issues"]:
                    for issue in health["issues"]:
                        if issue["type"] == "high_load":
                            # Scale up extractors
                            await agent_system.scale_agents(
                                role=AgentRole.EXTRACTOR,
                                delta=2
                            )
                        elif issue["type"] == "low_success_rate":
                            # Trigger emergency learning
                            await agent_system.trigger_emergency_learning()
            
            # Export metrics
            await exporter.export_metrics({
                "agent_count": len(agent_system.agents),
                "active_tasks": agent_system.get_active_tasks(),
                "success_rate": health.get("success_rate", 0),
                "processing_time_p95": health.get("processing_time_p95", 0)
            })
            
            await asyncio.sleep(60)  # Check every minute
            
    finally:
        await exporter.stop()
        await agent_system.stop()
        system_task.cancel()


# ============================================================================
# EXAMPLE 7: A/B Testing Agent Strategies
# ============================================================================

async def example_ab_testing():
    """
    Example showing how to A/B test different agent strategies.
    """
    from agents.experimentation.ab_tester import ABTester
    
    # Create A/B test for extraction strategies
    ab_tester = ABTester()
    
    # Define test variants
    await ab_tester.create_test(
        test_name="extraction_strategy_optimization",
        variants={
            "control": {
                "strategy": "standard_extraction",
                "confidence_threshold": 0.85
            },
            "variant_a": {
                "strategy": "multi_pass_extraction",
                "confidence_threshold": 0.80
            },
            "variant_b": {
                "strategy": "ensemble_extraction",
                "confidence_threshold": 0.90
            }
        },
        metrics=["accuracy", "processing_time", "confidence"],
        traffic_split=[0.5, 0.25, 0.25]  # 50% control, 25% each variant
    )
    
    # Run test
    test_results = []
    for i in range(1000):
        # Assign to variant
        variant = ab_tester.assign_variant("extraction_strategy_optimization")
        
        # Process with assigned strategy
        result = await process_with_strategy(
            document_id=f"test_doc_{i}",
            strategy=variant["strategy"],
            confidence_threshold=variant["confidence_threshold"]
        )
        
        # Track results
        ab_tester.track_metric(
            test_name="extraction_strategy_optimization",
            variant_name=variant["name"],
            metrics={
                "accuracy": result["accuracy"],
                "processing_time": result["processing_time"],
                "confidence": result["confidence"]
            }
        )
        
        test_results.append(result)
    
    # Analyze results
    analysis = ab_tester.analyze_test("extraction_strategy_optimization")
    
    print(f"A/B Test Results:")
    print(f"Winner: {analysis['winner']}")
    print(f"Improvement: {analysis['improvement']}%")
    print(f"Statistical significance: {analysis['p_value']}")
    
    # Deploy winning strategy
    if analysis["significant"] and analysis["winner"] != "control":
        await deploy_strategy(analysis["winner_config"])


# ============================================================================
# Main execution
# ============================================================================

async def main():
    """Run all examples"""
    examples = [
        ("Basic Document Processing", example_basic_document_processing),
        ("Conversational Interface", example_conversational_interface),
        ("Batch Processing with Learning", example_batch_processing_with_learning),
        ("Human Expert Collaboration", example_human_expert_collaboration),
        ("Production Deployment", example_production_deployment),
        ("A/B Testing Strategies", example_ab_testing)
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"Running Example: {name}")
        print(f"{'='*60}\n")
        
        try:
            await example_func()
        except Exception as e:
            print(f"Example failed: {str(e)}")
        
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
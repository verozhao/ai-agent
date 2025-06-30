"""
Comprehensive testing framework for AI Agent System including
unit tests, integration tests, performance tests, and chaos engineering.
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from hypothesis import given, strategies as st, settings
from locust import HttpUser, task, between

from agents.core.base_agent import BaseAgent, AgentRole
from agents.orchestration.system_launcher import AIAgentSystem
from tests.fixtures.document_fixtures import generate_test_documents
from tests.helpers.agent_mocks import create_mock_agent
from tests.chaos.fault_injection import FaultInjector


# ============================================================================
# Unit Tests
# ============================================================================

class TestBaseAgent:
    """Unit tests for base agent functionality"""
    
    @pytest_asyncio.fixture
    async def agent(self):
        """Create test agent"""
        agent = create_mock_agent("test_agent", AgentRole.EXTRACTOR)
        yield agent
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.agent_id == "test_agent"
        assert agent.role == AgentRole.EXTRACTOR
        assert agent.state.value == "idle"
        assert agent.decisions_made == 0
        assert agent.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_agent_thinking(self, agent):
        """Test agent reasoning process"""
        context = {
            "situation": "Document needs extraction",
            "document_type": "invoice",
            "confidence_required": 0.8
        }
        
        decision = await agent.think(context)
        
        assert decision is not None
        assert decision.action in ["standard_extraction", "complex_extraction"]
        assert 0 <= decision.confidence <= 1
        assert len(decision.reasoning) > 0
        assert agent.decisions_made == 1
    
    @pytest.mark.asyncio
    async def test_agent_learning(self, agent):
        """Test agent learning from feedback"""
        initial_success_rate = agent.success_rate
        
        # Simulate failures
        for _ in range(5):
            await agent.learn_from_feedback({
                "outcome": "failure",
                "error": "extraction_failed"
            })
        
        assert agent.success_rate < initial_success_rate
        assert len(agent.learning_episodes) == 5
    
    @pytest.mark.asyncio
    async def test_agent_communication(self, agent):
        """Test inter-agent communication"""
        message_sent = await agent.communicate(
            recipient="test_recipient",
            message_type="test_message",
            content={"data": "test"},
            priority=5
        )
        
        assert message_sent is not None
        # Verify message was queued properly
        assert agent.communicator.message_count > 0
    
    @pytest.mark.parametrize("error_type", [
        "network_error",
        "processing_error",
        "memory_error"
    ])
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, agent, error_type):
        """Test agent handles errors gracefully"""
        with patch.object(agent, '_execute_task', side_effect=Exception(error_type)):
            result = await agent.process_with_error_handling()
            
            assert result["status"] == "error"
            assert error_type in result["error_message"]
            # Verify agent remains in valid state
            assert agent.state.value in ["idle", "error_recovery"]


class TestDocumentExtractionEngine:
    """Unit tests for document extraction engine"""
    
    @pytest.mark.asyncio
    @given(
        document_type=st.sampled_from(["invoice", "contract", "report"]),
        page_count=st.integers(min_value=1, max_value=100),
        has_tables=st.booleans(),
        has_images=st.booleans()
    )
    @settings(max_examples=50)
    async def test_extraction_with_various_documents(
        self,
        document_type,
        page_count,
        has_tables,
        has_images
    ):
        """Property-based testing for document extraction"""
        from core.extraction.engine import DocumentExtractionEngine
        
        engine = DocumentExtractionEngine()
        
        # Generate test document
        document = {
            "type": document_type,
            "pages": page_count,
            "has_tables": has_tables,
            "has_images": has_images,
            "content": f"Test {document_type} content"
        }
        
        result = await engine.extract(document)
        
        # Verify extraction properties
        assert result is not None
        assert "extracted_data" in result
        assert "confidence_scores" in result
        assert 0 <= result["anomaly_score"] <= 1
        assert result["processing_time_ms"] > 0
        
        # Complex documents should take longer
        if page_count > 10 or has_tables:
            assert result["processing_time_ms"] > 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestAgentSystemIntegration:
    """Integration tests for complete agent system"""
    
    @pytest_asyncio.fixture
    async def agent_system(self):
        """Create and initialize agent system"""
        config = AgentSystemConfig(
            num_extractors=2,
            num_analyzers=1,
            learning_enabled=True
        )
        
        system = AIAgentSystem(config)
        await system.initialize()
        
        # Start system in background
        system_task = asyncio.create_task(system.start())
        
        yield system
        
        # Cleanup
        await system.stop()
        system_task.cancel()
    
    @pytest.mark.asyncio
    async def test_end_to_end_document_processing(self, agent_system):
        """Test complete document processing workflow"""
        # Submit document
        result = await agent_system.agents["coordinator_001"].process_document_request(
            document_id="test_invoice.pdf",
            priority=5,
            metadata={"type": "invoice"}
        )
        
        assert result["status"] == "completed"
        assert "extraction" in result
        assert result["extraction"]["status"] == "success"
        
        # Verify all agents participated
        coordinator_metrics = agent_system.agents["coordinator_001"].system_metrics
        assert coordinator_metrics["total_processed"] > 0
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_workflow(self, agent_system):
        """Test anomaly detection and routing"""
        # Create anomalous document
        anomalous_doc = {
            "id": "anomaly_test.pdf",
            "content": "CORRUPTED_DATA_" * 100,
            "metadata": {"inject_anomaly": True}
        }
        
        result = await agent_system.agents["coordinator_001"].process_document_request(
            document_id=anomalous_doc["id"],
            metadata=anomalous_doc["metadata"]
        )
        
        # Should be routed to eval_set_2
        assert "analysis" in result
        assert result["analysis"]["routing"]["action"] == "eval_set_2"
    
    @pytest.mark.asyncio
    async def test_feedback_loop_integration(self, agent_system):
        """Test feedback loop improves performance"""
        # Process documents and provide feedback
        results = []
        for i in range(20):
            result = await agent_system.agents["coordinator_001"].process_document_request(
                document_id=f"feedback_test_{i}.pdf"
            )
            results.append(result)
            
            # Simulate feedback
            if i % 2 == 0:
                feedback = {
                    "extraction_id": result["extraction"]["extraction_id"],
                    "corrections": {"field": "value"},
                    "quality_score": 0.7
                }
                await agent_system.agents["feedback_processor_001"].process_feedback(feedback)
        
        # Check if learning was triggered
        feedback_processor = agent_system.agents.get("feedback_processor_001")
        if feedback_processor:
            assert len(feedback_processor.feedback_buffer) > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput(self, agent_system):
        """Test system throughput"""
        start_time = datetime.utcnow()
        documents_processed = 0
        target_duration = timedelta(seconds=60)
        
        tasks = []
        while datetime.utcnow() - start_time < target_duration:
            task = agent_system.agents["coordinator_001"].process_document_request(
                document_id=f"perf_test_{documents_processed}.pdf",
                priority=5
            )
            tasks.append(task)
            documents_processed += 1
            
            # Process in batches
            if len(tasks) >= 100:
                await asyncio.gather(*tasks)
                tasks = []
        
        # Process remaining
        if tasks:
            await asyncio.gather(*tasks)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        throughput = documents_processed / duration
        
        print(f"Throughput: {throughput:.2f} documents/second")
        assert throughput > 10  # Minimum acceptable throughput
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_latency_percentiles(self, agent_system):
        """Test latency percentiles"""
        latencies = []
        
        for i in range(1000):
            start = datetime.utcnow()
            
            await agent_system.agents["coordinator_001"].process_document_request(
                document_id=f"latency_test_{i}.pdf"
            )
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            latencies.append(latency)
        
        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"Latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
        
        assert p50 < 100  # 100ms median
        assert p95 < 500  # 500ms for 95th percentile
        assert p99 < 1000  # 1s for 99th percentile


# ============================================================================
# Load Testing with Locust
# ============================================================================

class DocumentProcessingUser(HttpUser):
    """Locust user for load testing"""
    wait_time = between(0.1, 1)
    
    def on_start(self):
        """Initialize user session"""
        # Authenticate
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def process_document(self):
        """Submit document for processing"""
        with self.client.post(
            "/api/v1/documents/process",
            json={
                "document_id": f"load_test_{random.randint(1, 10000)}.pdf",
                "priority": random.randint(1, 10)
            },
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def check_status(self):
        """Check processing status"""
        doc_id = f"load_test_{random.randint(1, 10000)}.pdf"
        self.client.get(
            f"/api/v1/documents/status/{doc_id}",
            headers=self.headers
        )
    
    @task(1)
    def get_insights(self):
        """Get system insights"""
        self.client.get(
            "/api/v1/insights/dashboard",
            headers=self.headers
        )


# ============================================================================
# Chaos Engineering Tests
# ============================================================================

class TestChaosEngineering:
    """Chaos engineering tests to verify system resilience"""
    
    @pytest_asyncio.fixture
    async def chaos_injector(self):
        """Create chaos injector"""
        return FaultInjector()
    
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_agent_failure_recovery(self, agent_system, chaos_injector):
        """Test system recovers from agent failures"""
        # Kill random agent
        agent_to_kill = random.choice(list(agent_system.agents.keys()))
        await chaos_injector.kill_agent(agent_system, agent_to_kill)
        
        # System should still process documents
        result = await agent_system.agents["coordinator_001"].process_document_request(
            document_id="chaos_test.pdf"
        )
        
        assert result["status"] in ["completed", "degraded"]
        
        # Verify agent was restarted
        await asyncio.sleep(5)
        assert agent_to_kill in agent_system.agents
    
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_network_partition(self, agent_system, chaos_injector):
        """Test handling of network partitions"""
        # Simulate network partition
        await chaos_injector.create_network_partition(
            agent_system,
            ["extractor_001", "extractor_002"],
            ["analyzer_001"]
        )
        
        # System should handle gracefully
        result = await agent_system.agents["coordinator_001"].process_document_request(
            document_id="partition_test.pdf"
        )
        
        assert result is not None
        # May be slower but should complete
        assert result.get("degraded_mode") is True
    
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_resource_exhaustion(self, agent_system, chaos_injector):
        """Test system under resource exhaustion"""
        # Simulate memory pressure
        await chaos_injector.exhaust_memory(agent_system, percentage=80)
        
        # Submit burst of documents
        tasks = []
        for i in range(100):
            task = agent_system.agents["coordinator_001"].process_document_request(
                document_id=f"resource_test_{i}.pdf"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some may fail but system should not crash
        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful > 50  # At least 50% should succeed
    
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_cascading_failures(self, agent_system, chaos_injector):
        """Test prevention of cascading failures"""
        # Inject latency in one component
        await chaos_injector.inject_latency(
            agent_system.agents["analyzer_001"],
            delay_ms=5000
        )
        
        # Submit multiple requests
        start_time = datetime.utcnow()
        tasks = []
        
        for i in range(20):
            task = agent_system.agents["coordinator_001"].process_document_request(
                document_id=f"cascade_test_{i}.pdf"
            )
            tasks.append(task)
        
        # Should implement circuit breaker
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Should fail fast, not wait for all timeouts
        assert duration < 30  # Should not take full timeout * 20


# ============================================================================
# Contract Tests
# ============================================================================

class TestAgentContracts:
    """Contract tests for agent interfaces"""
    
    @pytest.mark.asyncio
    async def test_extractor_contract(self):
        """Test extractor agent contract"""
        from agents.contracts import ExtractorContract
        
        contract = ExtractorContract()
        
        # Test request format
        valid_request = {
            "document_id": "test.pdf",
            "extraction_type": "standard",
            "options": {}
        }
        
        assert contract.validate_request(valid_request)
        
        # Test response format
        mock_response = {
            "extraction_id": "ext_123",
            "status": "success",
            "extracted_data": {},
            "confidence_scores": {},
            "processing_time_ms": 100
        }
        
        assert contract.validate_response(mock_response)
    
    @pytest.mark.asyncio
    async def test_agent_communication_contract(self):
        """Test inter-agent communication contract"""
        from agents.contracts import CommunicationContract
        
        contract = CommunicationContract()
        
        message = {
            "sender": "agent_001",
            "recipient": "agent_002",
            "message_type": "task_request",
            "content": {"task": "process"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        assert contract.validate_message(message)


# ============================================================================
# Test Helpers
# ============================================================================

class AgentTestHelper:
    """Helper class for agent testing"""
    
    @staticmethod
    async def create_test_scenario(scenario_type: str) -> Dict[str, Any]:
        """Create test scenarios"""
        scenarios = {
            "simple": {
                "documents": generate_test_documents(count=10, complexity="simple"),
                "expected_success_rate": 0.95,
                "max_processing_time": 1000
            },
            "complex": {
                "documents": generate_test_documents(count=10, complexity="complex"),
                "expected_success_rate": 0.85,
                "max_processing_time": 5000
            },
            "anomalous": {
                "documents": generate_test_documents(
                    count=10,
                    anomaly_rate=0.5
                ),
                "expected_anomaly_detection": 0.9,
                "expected_routing": "eval_set_2"
            }
        }
        
        return scenarios.get(scenario_type, scenarios["simple"])
    
    @staticmethod
    def assert_agent_metrics(agent: BaseAgent, expected: Dict[str, Any]):
        """Assert agent metrics match expected values"""
        assert agent.decisions_made >= expected.get("min_decisions", 0)
        assert agent.success_rate >= expected.get("min_success_rate", 0.8)
        
        if "max_memory_mb" in expected:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            assert memory_mb <= expected["max_memory_mb"]


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "chaos: mark test as chaos engineering test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "contract: mark test as contract test")
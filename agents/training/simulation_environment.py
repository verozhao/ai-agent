"""
Advanced simulation environment for training and testing AI agents
in realistic document processing scenarios.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
import numpy as np
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from agents.core.base_agent import BaseAgent, AgentRole
from agents.training.scenario_generator import ScenarioGenerator
from agents.training.performance_evaluator import PerformanceEvaluator


logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    STRESS_TEST = "stress_test"
    ADVERSARIAL = "adversarial"


@dataclass
class SimulationConfig:
    """Configuration for simulation environment"""
    mode: SimulationMode
    duration_hours: int
    document_rate: float  # documents per minute
    anomaly_rate: float
    complexity_distribution: Dict[str, float]
    agent_failure_rate: float
    network_latency_ms: Tuple[int, int]
    enable_learning: bool
    record_trajectories: bool


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    total_documents: int
    successful_extractions: int
    anomalies_detected: int
    false_positives: int
    false_negatives: int
    avg_processing_time: float
    agent_collaborations: int
    human_interventions: int
    system_failures: int
    learning_improvements: Dict[str, float]


class DocumentProcessingEnv(gym.Env):
    """
    OpenAI Gym environment for training agents in document processing.
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        agent_system: Any
    ):
        super().__init__()
        self.config = config
        self.agent_system = agent_system
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(10)  # 10 possible actions
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(128,), dtype=np.float32
        )
        
        # Environment state
        self.current_document = None
        self.processing_queue = []
        self.metrics = SimulationMetrics(
            total_documents=0,
            successful_extractions=0,
            anomalies_detected=0,
            false_positives=0,
            false_negatives=0,
            avg_processing_time=0,
            agent_collaborations=0,
            human_interventions=0,
            system_failures=0,
            learning_improvements={}
        )
        
        # Scenario generator
        self.scenario_generator = ScenarioGenerator()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Generate initial document
        self.current_document = self.scenario_generator.generate_document(
            self.config.complexity_distribution
        )
        
        # Reset metrics
        self.metrics = SimulationMetrics(
            total_documents=0,
            successful_extractions=0,
            anomalies_detected=0,
            false_positives=0,
            false_negatives=0,
            avg_processing_time=0,
            agent_collaborations=0,
            human_interventions=0,
            system_failures=0,
            learning_improvements={}
        )
        
        # Get initial observation
        obs = self._get_observation()
        info = {"document": self.current_document}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return results"""
        # Map action to agent decision
        agent_action = self._map_action(action)
        
        # Execute action through agent system
        result = asyncio.run(
            self.agent_system.process_action(
                agent_action,
                self.current_document
            )
        )
        
        # Calculate reward
        reward = self._calculate_reward(result)
        
        # Update metrics
        self._update_metrics(result)
        
        # Generate next document
        self.current_document = self.scenario_generator.generate_document(
            self.config.complexity_distribution
        )
        
        # Check if episode is done
        terminated = self.metrics.total_documents >= 1000
        truncated = False
        
        # Get next observation
        obs = self._get_observation()
        info = {
            "result": result,
            "metrics": self.metrics,
            "document": self.current_document
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation"""
        # Extract document features
        doc_features = self.scenario_generator.extract_features(
            self.current_document
        )
        
        # Get system state
        system_state = self.agent_system.get_system_state()
        
        # Combine into observation vector
        obs = np.concatenate([
            doc_features,
            system_state["agent_utilization"],
            system_state["queue_lengths"],
            [self.metrics.success_rate],
            [self.metrics.avg_processing_time / 1000]  # Normalize
        ])
        
        return obs.astype(np.float32)[:128]  # Ensure correct size
    
    def _calculate_reward(self, result: Dict[str, Any]) -> float:
        """Calculate reward based on action result"""
        reward = 0.0
        
        # Success reward
        if result["status"] == "success":
            reward += 1.0
            
            # Bonus for high confidence
            if result["confidence"] > 0.9:
                reward += 0.5
            
            # Bonus for fast processing
            if result["processing_time"] < 1000:  # < 1 second
                reward += 0.3
        
        # Penalty for failures
        elif result["status"] == "failed":
            reward -= 1.0
        
        # Penalty for unnecessary human intervention
        if result.get("human_intervention") and not result.get("intervention_necessary"):
            reward -= 0.5
        
        # Reward for correct anomaly detection
        if result.get("anomaly_detected") == self.current_document.get("has_anomaly"):
            reward += 0.3
        else:
            reward -= 0.3  # Penalty for false positive/negative
        
        # Reward for efficient collaboration
        if result.get("agent_collaboration") and result["collaboration_efficient"]:
            reward += 0.2
        
        return reward


class AgentTrainingSimulator:
    """
    Complete simulation environment for training and testing AI agents.
    """
    
    def __init__(
        self,
        agent_system: Any,
        config: SimulationConfig
    ):
        self.agent_system = agent_system
        self.config = config
        
        # Create environments
        self.envs = self._create_environments()
        
        # Performance evaluator
        self.evaluator = PerformanceEvaluator()
        
        # Training models
        self.models = {}
        
        # Scenario variations
        self.scenarios = {
            "normal_load": self._normal_load_scenario,
            "peak_hours": self._peak_hours_scenario,
            "anomaly_spike": self._anomaly_spike_scenario,
            "system_degradation": self._system_degradation_scenario,
            "adversarial_attack": self._adversarial_attack_scenario
        }
    
    def _create_environments(self) -> List[DocumentProcessingEnv]:
        """Create multiple parallel environments"""
        def make_env():
            return DocumentProcessingEnv(self.config, self.agent_system)
        
        # Create parallel environments for faster training
        num_envs = 8
        if self.config.mode == SimulationMode.TRAINING:
            return SubprocVecEnv([make_env for _ in range(num_envs)])
        else:
            return DummyVecEnv([make_env])
    
    async def run_simulation(
        self,
        scenario_name: str = "normal_load",
        duration_hours: Optional[int] = None
    ) -> SimulationMetrics:
        """Run a complete simulation scenario"""
        duration = duration_hours or self.config.duration_hours
        scenario_func = self.scenarios.get(scenario_name, self._normal_load_scenario)
        
        logger.info(f"Starting {scenario_name} simulation for {duration} hours")
        
        # Initialize simulation
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration)
        
        # Run scenario
        metrics = await scenario_func(start_time, end_time)
        
        # Evaluate performance
        evaluation = await self.evaluator.evaluate(
            metrics,
            self.agent_system.get_agent_logs()
        )
        
        logger.info(f"Simulation completed. Overall score: {evaluation['overall_score']}")
        
        return metrics
    
    async def _normal_load_scenario(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> SimulationMetrics:
        """Simulate normal operating conditions"""
        current_time = start_time
        interval_seconds = 60 / self.config.document_rate
        
        while current_time < end_time:
            # Generate document
            document = self.scenario_generator.generate_document(
                self.config.complexity_distribution,
                anomaly_probability=self.config.anomaly_rate
            )
            
            # Submit to agent system
            result = await self.agent_system.process_document_request(
                document["id"],
                priority=document.get("priority", 5)
            )
            
            # Simulate network latency
            latency = random.uniform(*self.config.network_latency_ms) / 1000
            await asyncio.sleep(latency)
            
            # Update time
            current_time += timedelta(seconds=interval_seconds)
            
            # Periodic system health check
            if random.random() < 0.01:  # 1% chance
                await self._simulate_system_issue()
        
        return self.agent_system.get_simulation_metrics()
    
    async def _peak_hours_scenario(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> SimulationMetrics:
        """Simulate peak load conditions"""
        # Increase document rate by 3x
        original_rate = self.config.document_rate
        self.config.document_rate *= 3
        
        # Run simulation with higher load
        metrics = await self._normal_load_scenario(start_time, end_time)
        
        # Restore original rate
        self.config.document_rate = original_rate
        
        return metrics
    
    async def _anomaly_spike_scenario(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> SimulationMetrics:
        """Simulate sudden spike in anomalous documents"""
        # Increase anomaly rate dramatically
        original_anomaly_rate = self.config.anomaly_rate
        self.config.anomaly_rate = 0.4  # 40% anomalies
        
        # Add more complex documents
        self.config.complexity_distribution["complex"] *= 2
        
        metrics = await self._normal_load_scenario(start_time, end_time)
        
        # Restore original configuration
        self.config.anomaly_rate = original_anomaly_rate
        
        return metrics
    
    async def _adversarial_attack_scenario(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> SimulationMetrics:
        """Simulate adversarial attacks on the system"""
        current_time = start_time
        
        while current_time < end_time:
            # Generate adversarial document
            document = self.scenario_generator.generate_adversarial_document()
            
            # Try to confuse the agents
            tasks = []
            for _ in range(10):  # Submit same document multiple times
                task = self.agent_system.process_document_request(
                    document["id"],
                    priority=random.randint(1, 10)
                )
                tasks.append(task)
            
            # Execute simultaneously
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Simulate DDoS-like pattern
            if random.random() < 0.3:
                burst_tasks = []
                for _ in range(100):
                    doc = self.scenario_generator.generate_document()
                    task = self.agent_system.process_document_request(doc["id"])
                    burst_tasks.append(task)
                
                await asyncio.gather(*burst_tasks, return_exceptions=True)
            
            current_time += timedelta(minutes=5)
        
        return self.agent_system.get_simulation_metrics()
    
    async def train_agents(
        self,
        num_episodes: int = 10000,
        save_interval: int = 1000
    ):
        """Train agents using reinforcement learning"""
        logger.info(f"Starting agent training for {num_episodes} episodes")
        
        # Create RL model
        model = PPO(
            "MlpPolicy",
            self.envs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./logs/agent_training/"
        )
        
        # Training loop
        for episode in range(0, num_episodes, save_interval):
            # Train for interval
            model.learn(
                total_timesteps=save_interval,
                reset_num_timesteps=False
            )
            
            # Evaluate performance
            evaluation = await self._evaluate_trained_model(model)
            logger.info(
                f"Episode {episode}: Average reward = {evaluation['avg_reward']:.2f}"
            )
            
            # Save checkpoint
            model.save(f"models/agents/checkpoint_{episode}")
            
            # Update agent policies
            await self._update_agent_policies(model)
        
        # Final save
        model.save("models/agents/final_model")
        
        logger.info("Agent training completed")
    
    async def _evaluate_trained_model(
        self,
        model: Any,
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate trained model performance"""
        eval_env = DocumentProcessingEnv(self.config, self.agent_system)
        
        total_reward = 0
        total_success = 0
        
        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                if info["result"]["status"] == "success":
                    total_success += 1
            
            total_reward += episode_reward
        
        return {
            "avg_reward": total_reward / num_episodes,
            "success_rate": total_success / (num_episodes * 1000)  # 1000 docs per episode
        }
    
    async def stress_test_system(
        self,
        test_duration_hours: int = 24
    ) -> Dict[str, Any]:
        """Comprehensive stress test of the agent system"""
        logger.info(f"Starting {test_duration_hours}-hour stress test")
        
        stress_tests = [
            ("sustained_high_load", self._test_sustained_load),
            ("memory_pressure", self._test_memory_pressure),
            ("network_failures", self._test_network_failures),
            ("agent_failures", self._test_agent_failures),
            ("cascade_failures", self._test_cascade_failures)
        ]
        
        results = {}
        
        for test_name, test_func in stress_tests:
            logger.info(f"Running stress test: {test_name}")
            result = await test_func(test_duration_hours // len(stress_tests))
            results[test_name] = result
            
            # Check if system is still healthy
            health = await self.agent_system.check_health()
            if health["status"] != "healthy":
                logger.warning(f"System unhealthy after {test_name}: {health}")
                break
        
        # Generate comprehensive report
        report = self._generate_stress_test_report(results)
        
        return report
    
    async def _simulate_system_issue(self):
        """Simulate random system issues"""
        issues = [
            ("agent_crash", 0.02),
            ("network_partition", 0.01),
            ("database_slowdown", 0.03),
            ("memory_leak", 0.01),
            ("cpu_spike", 0.02)
        ]
        
        for issue, probability in issues:
            if random.random() < probability:
                logger.warning(f"Simulating system issue: {issue}")
                
                if issue == "agent_crash":
                    # Simulate agent crash
                    agent_id = random.choice(list(self.agent_system.agents.keys()))
                    await self.agent_system.simulate_agent_failure(agent_id)
                
                elif issue == "network_partition":
                    # Simulate network issues
                    await asyncio.sleep(random.uniform(1, 5))
                
                elif issue == "database_slowdown":
                    # Add artificial delay
                    self.config.network_latency_ms = (500, 2000)
                
                break
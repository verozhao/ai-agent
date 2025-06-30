"""
Entry point for the AI Agent Document Processing System.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from agents.orchestration.system_launcher import AIAgentSystem
from config.agent_config import AgentSystemConfig


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("AI Agent Document Processing System")
    logger.info("=" * 50)
    
    # Load configuration
    config = AgentSystemConfig.from_env()
    
    # Create system
    system = AIAgentSystem(config)
    
    # Setup signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, system.handle_shutdown)
    
    try:
        # Initialize system
        await system.initialize()
        
        # Start system
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {str(e)}", exc_info=True)
    finally:
        # Ensure clean shutdown
        await system.stop()
        logger.info("System shutdown complete")


if __name__ == "__main__":
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Run the system
    asyncio.run(main())
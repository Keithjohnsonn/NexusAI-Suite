import sys
from loguru import logger

def setup_logger(level: str = "INFO"):
    """
    Configures the Loguru logger with custom formatting and levels.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )
    return logger

# Initialize default logger
log = setup_logger()

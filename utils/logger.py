"""
Centralized logging setup for the pipeline.
Writes to both stdout and a rotating file handler.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL, LOGS_DIR

def get_logger(
    name: str,
    level: str = LOG_LEVEL,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Return a logger with console + optional rotating-file handlers.
    
    Args:
        name: Logger name.
        level: Logging level string, e.g. 'INFO', 'DEBUG'.
        log_file: Optional file path; defaults to LOGS_DIR / '<name>.log'.
        
    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers in interactive / notebook environments
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler
    if log_file is None:
        log_file = LOGS_DIR / f"{name.replace('.', '_')}.log"
        
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes = 10 * 1024 * 1024, # 10 MB
        backupCount = 3,
        encoding = "utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger (avoids duplicate log lines).
    logger.propagate = False
    
    return logger
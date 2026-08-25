
#main goal is to setup logging which configures for proper logging.

import logging
import sys

#importing the app base configurations
from app.core.config import Settings

def setup_logging(log_level) -> logging.Logger:
    
    #creating a custom logger
    logger = logging.getLogger("audio_logger")
    
    #checking is there is anything written about the logger.DEBUG in <settings>
    # and/or initializing + setting the logging levels.
    # log_level = logging.INFO
    
    logger.setLevel(level=log_level)
    
    # Prevent log messages from being duplicated if setup_logging is called twice
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create a formatter that includes the Process ID (%(process)d)
    # This is critical for debugging our ProcessPoolExecutor
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | PID:%(process)d | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Attach formatter to handler, and handler to logger
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Silence noisy third-party libraries if necessary
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("passlib").setLevel(logging.WARNING)
    
    return logger

# Expose a pre-configured logger instance
level = logging.INFO
logger = setup_logging(log_level=level)
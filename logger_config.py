import logging
import sys

def setup_logger(logger_name='image_styler_app', level=logging.INFO):
    """
    Sets up and returns a logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create a handler (e.g., StreamHandler to log to console)
    # You can add more handlers like FileHandler here
    stream_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for Streamlit compatibility

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    # Check if handlers are already added to prevent duplication during Streamlit reruns
    if not logger.handlers:
        logger.addHandler(stream_handler)

    return logger

# Example of how to get a configured logger
# logger = setup_logger()
# logger.info("This is an info message from logger_config.py if run directly.")
"""Logger for fasthtml application."""

# Reference : https://github.com/eugeneyan/framework-comparison/blob/main/fasthtml/logger.py
import logging
import sys


def get_logger(name: str = __name__, level: str = "INFO") -> logging.Logger:
    """Initialize a simple logger that outputs to the console.

    Args:
        name (str): The name of the logger. Defaults to the module name.
        level (str): The logging level. Defaults to "INFO".

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level
    level = getattr(logging, level.upper(), logging.INFO)  # type: ignore
    logger.setLevel(level)

    # Create a console handler and set its level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add the formatter to the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger

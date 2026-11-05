# Start scheduler
import json
import logging
import logging.config
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(config_file: str = "logging_config.json") -> None:
    """Loads the logging configuration."""
    logger.info("Logging configuration starting.")
    path = str(Path(__file__).resolve().parent.parent)
    try:
        with open(path + "/resources/" + config_file) as file_handler:
            config = json.load(file_handler)
        logging.config.dictConfig(config)
    except Exception:
        logging.basicConfig(level=logging.INFO)
        logger.exception("Failed to load application logging")
    logger.info("Logging configured and application starting.")

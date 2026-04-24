import logging
import os
import sys
from pythonjsonlogger import jsonlogger


class ContextFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.env = os.getenv("APP_ENV", "dev")
        self.service = os.getenv("SERVICE_NAME", "fraud-detection-api")
        self.model_version = os.getenv("MODEL_VERSION", "unknown")

    def filter(self, record: logging.LogRecord) -> bool:
        record.env = self.env
        record.service = self.service
        record.model_version = self.model_version
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())

    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s %(env)s %(service)s %(model_version)s",
        rename_fields={
            "asctime": "timestamp",
            "name": "logger",
            "levelname": "level",
        },
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
import logging
import os
from datetime import datetime
from pathlib import Path


class CustomLogger:
    def __init__(self, config, logger_name: str = "RBCParser"):
        self.config = config
        self.logger_name = logger_name
        self.logger = self._setup_logger()

    def get_log_filename(self, logs_dir: str) -> str:
        """Generate unique log filename based on start time"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(logs_dir, f'rbc_parser_{timestamp}.log')

    def _setup_logger(self) -> logging.Logger:
        # Create logger
        logger = logging.getLogger(self.logger_name)

        # Clear any existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(self.config.logging.log_level)

        # Create logs directory if it doesn't exist
        log_dir = Path(self.config.paths.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler
        log_file = self.get_log_filename(self.config.paths.log_dir)
        file_handler = logging.FileHandler(log_file, encoding=self.config.output.encoding)
        file_handler.setLevel(self.config.logging.log_level)

        # Create formatter
        formatter = logging.Formatter(
            fmt=self.config.logging.format,
            datefmt=self.config.logging.date_format
        )

        # Add formatter to handler
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        return logger

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def debug(self, message: str):
        self.logger.debug(message)


import logging
import os
from datetime import datetime


def setup_logging(logs_dir: str, file_level: str, console_level: str, log_format: str):
    """Configure logging with file and console handlers"""
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = get_log_filename(logs_dir)

    formatter = logging.Formatter(log_format)

    # File handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(getattr(logging, file_level))
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level))
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, console_level))  # Устанавливаем базовый уровень как console_level

    # Очищаем существующие обработчики
    logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_filename}")

def get_log_filename(logs_dir: str) -> str:
    """Generate unique log filename based on start time"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(logs_dir, f'rbc_parser_{timestamp}.log')
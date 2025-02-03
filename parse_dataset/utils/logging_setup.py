import logging
import os
from datetime import datetime
from .file_utils import create_directory_structure

def setup_logging():
    """Configure logging with file and console handlers"""
    create_directory_structure()
    log_filename = get_log_filename()

    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_filename}")

def get_log_filename() -> str:
    """Generate unique log filename based on start time"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join('logs', f'rbc_parser_{timestamp}.log')
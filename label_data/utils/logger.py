import logging
import os
from datetime import datetime
from pathlib import Path


class CustomLogger:
    """
    Кастомный логгер для модуля маркировки данных.

    Класс предоставляет настраиваемый механизм логирования с поддержкой
    записи в файл и форматированием сообщений. Логи сохраняются в указанную
    директорию с уникальным именем файла, включающим временную метку.

    Attributes:
        config: Конфигурационный объект с настройками логгера
        logger_name (str): Имя логгера, по умолчанию "DataLabeler"
        logger (logging.Logger): Основной объект логгера
    """

    def __init__(self, config, logger_name: str = "DataLabeler") -> None:
        """
        Инициализация кастомного логгера.

        Args:
            config: Объект конфигурации с настройками логгера
            logger_name: Имя логгера, используется для идентификации источника логов
        """
        self.config = config
        self.logger_name = logger_name
        self.logger = self._setup_logger()

    def get_log_filename(self, logs_dir: str) -> str:
        """
        Генерирует уникальное имя файла для логов на основе текущего времени.

        Args:
            logs_dir: Директория для сохранения файлов логов

        Returns:
            str: Полный путь к файлу лога с уникальным именем
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(logs_dir, f'data_labeler_{timestamp}.log')

    def _setup_logger(self) -> logging.Logger:
        """
        Настраивает и инициализирует логгер с заданными параметрами.

        Создает директорию для логов, если она не существует, настраивает
        форматирование и обработчики для записи в файл.

        Returns:
            logging.Logger: Настроенный объект логгера

        Raises:
            OSError: При проблемах с созданием директории или файла логов
            ValueError: При некорректных настройках логирования
        """
        # Создаем логгер
        logger = logging.getLogger(self.logger_name)

        # Очищаем существующие обработчики
        if logger.hasHandlers():
            logger.handlers.clear()

        # Устанавливаем уровень логирования
        logger.setLevel(self.config.logging.log_level)

        # Создаем директорию для логов, если она не существует
        log_dir = Path(self.config.paths.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Создаем обработчик файла
        log_file = self.get_log_filename(self.config.paths.log_dir)
        file_handler = logging.FileHandler(
            log_file,
            encoding=self.config.output.encoding
        )
        file_handler.setLevel(self.config.logging.log_level)

        # Создаем обработчик консоли
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.logging.log_level)

        # Создаем форматтер
        formatter = logging.Formatter(
            fmt=self.config.logging.format,
            datefmt=self.config.logging.date_format
        )

        # Добавляем форматтер к обработчикам
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Добавляем обработчики к логгеру
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def info(self, message: str) -> None:
        """
        Логирует информационное сообщение.

        Args:
            message: Текст сообщения для логирования
        """
        self.logger.info(message)

    def error(self, message: str) -> None:
        """
        Логирует сообщение об ошибке.

        Args:
            message: Текст сообщения об ошибке
        """
        self.logger.error(message)

    def warning(self, message: str) -> None:
        """
        Логирует предупреждающее сообщение.

        Args:
            message: Текст предупреждения
        """
        self.logger.warning(message)

    def debug(self, message: str) -> None:
        """
        Логирует отладочное сообщение.

        Args:
            message: Текст отладочного сообщения
        """
        self.logger.debug(message)
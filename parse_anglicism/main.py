#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import hydra
from omegaconf import DictConfig

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модули для анализа
from utils.parser import parse_anglicisms
from utils.analyzer import analyze_anglicisms, clean_anglicisms, advanced_analysis
from utils.visualizer import visualize_anglicisms
from utils.io_utils import save_anglicisms, setup_directory_structure


@hydra.main(version_base=None, config_path="../configs/parse_anglicism", config_name="main")
def main(cfg: DictConfig):
    """
    Основная функция для анализа англицизмов с использованием конфигурации Hydra.

    Args:
        cfg (DictConfig): Конфигурация Hydra
    """
    # Настройка путей с абсолютными значениями
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = {
        "data_dir": os.path.join(base_dir, "data"),
        "output_dir": os.path.join(base_dir, "data"),
        "logs_dir": os.path.join(base_dir, "logs"),
        "visualization_dir": os.path.join(base_dir, "data/visualization")
    }

    # Создание структуры директорий
    setup_directory_structure(paths)

    # Создаем директорию для логов, если её нет
    if not os.path.exists(paths["logs_dir"]):
        os.makedirs(paths["logs_dir"])

    # Настройка логирования
    log_file = os.path.join(paths["logs_dir"], "parse_anglicism.log")

    # Настраиваем логирование
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Очищаем все обработчики
    while root_logger.handlers:
        root_logger.handlers.pop()

    # Создаем новые обработчики
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    console_handler = logging.StreamHandler()

    # Форматирование
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Добавляем обработчики
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info("Запуск анализа англицизмов")

    # Путь к файлу с англицизмами
    input_file = os.path.join(paths["data_dir"], "input.txt")

    # Проверка наличия входного файла
    if not os.path.exists(input_file):
        logger.error(f"Файл с англицизмами не найден: {input_file}")
        return

    # Парсинг англицизмов
    logger.info(f"Чтение файла: {input_file}")
    anglicisms_dict = parse_anglicisms(input_file, cfg.analysis if hasattr(cfg, "analysis") else None)

    if not anglicisms_dict["all_anglicisms"]:
        logger.error("Не удалось найти англицизмы в файле.")
        return

    # Базовый анализ данных
    logger.info("Выполнение базового анализа")
    df = analyze_anglicisms(anglicisms_dict)

    # Очистка и нормализация данных
    logger.info("Очистка и нормализация данных")
    clean_df = clean_anglicisms(df, cfg.analysis if hasattr(cfg, "analysis") else None)

    # Сохранение обработанных англицизмов
    output_file = os.path.join(paths["data_dir"], "output.txt")
    csv_output = os.path.join(paths["data_dir"], "output.csv")
    save_anglicisms(clean_df, output_file, csv_output)

    # Логгируем информацию о сохранении файлов
    logger.info(f"Сохранен список англицизмов: {output_file}")
    logger.info(f"Сохранен CSV файл с анализом: {csv_output}")

    # Продолжение анализа только если задано в конфигурации
    if hasattr(cfg, "analysis") and hasattr(cfg.analysis, "perform_advanced") and cfg.analysis.perform_advanced:
        # Расширенный анализ данных
        logger.info("Выполнение расширенного анализа")
        advanced_analysis(clean_df, cfg.analysis)

        # Визуализация данных
        logger.info("Создание визуализаций")
        visualize_anglicisms(clean_df, paths["visualization_dir"],
                             cfg.visualization if hasattr(cfg, "visualization") else None)
        logger.info(f"Визуализации сохранены в: {paths['visualization_dir']}")

    logger.info("Анализ англицизмов успешно завершен")
    return clean_df


if __name__ == "__main__":
    # Настраиваем переменные окружения для Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_LOGGING.LEVEL"] = "WARN"  # Уменьшаем вывод логов Hydra

    # Запуск через Hydra
    main()
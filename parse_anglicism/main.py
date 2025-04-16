#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модули для анализа
from utils.parser import parse_anglicisms, clean_wiki_markup
from utils.analyzer import analyze_anglicisms, clean_anglicisms, advanced_analysis, compare_languages, \
    analyze_letter_patterns
from utils.visualizer import visualize_anglicisms
from utils.io_utils import save_anglicisms, setup_directory_structure

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("parse_anglicism.log", encoding="utf-8")
    ]
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/parse_anglicism/analysis", config_name="main")
def main(cfg: DictConfig):
    """
    Основная функция для анализа англицизмов с использованием конфигурации Hydra.

    Args:
        cfg (DictConfig): Конфигурация Hydra
    """
    logger.info("Запуск анализа англицизмов")
    logger.info(f"Конфигурация: {OmegaConf.to_yaml(cfg)}")

    # Настройка путей
    paths = {
        "data_dir": "data",
        "output_dir": "outputs",
        "logs_dir": "logs",
        "visualization_dir": "outputs/visualization"
    }

    # Создание структуры директорий
    setup_directory_structure(paths)

    # Путь к файлу с англицизмами
    input_file = os.path.join(paths["data_dir"], "angl.txt")

    # Проверка наличия входного файла
    if not os.path.exists(input_file):
        logger.error(f"Файл с англицизмами не найден: {input_file}")
        logger.info("Создаем пустой файл для тестирования...")

        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(input_file), exist_ok=True)

        # Пример тестового содержимого
        test_content = """== Из английского языка ==
[[автомобиль]] происходит от англ. automobile
[[компьютер]] происходит от англ. computer
[[телефон]] происходит от англ. telephone

== Из французского языка ==
[[абажур]] происходит от фр. abat-jour
[[балет]] происходит от фр. ballet через англ. ballet
[[вуаль]] происходит от фр. voile
"""
        # Записываем тестовое содержимое
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        logger.info(f"Создан тестовый файл: {input_file}")

    # Парсинг англицизмов
    logger.info("Парсинг англицизмов из файла...")
    anglicisms_dict = parse_anglicisms(input_file, cfg)

    if not anglicisms_dict["all_anglicisms"]:
        logger.error("Не удалось найти англицизмы в файле.")
        return

    # Базовый анализ данных
    logger.info("=== БАЗОВЫЙ АНАЛИЗ ===")
    df = analyze_anglicisms(anglicisms_dict)

    # Очистка и нормализация данных
    logger.info("=== ОЧИСТКА ДАННЫХ ===")
    clean_df = clean_anglicisms(df, cfg)

    # Сохранение обработанных англицизмов
    output_file = os.path.join(paths["output_dir"], "clean_anglicisms.txt")
    excel_output = os.path.join(paths["output_dir"], "anglicisms_analysis.xlsx")
    save_anglicisms(clean_df, output_file, excel_output)

    # Продолжение анализа только если задано в конфигурации
    if cfg.perform_advanced:
        # Расширенный анализ данных
        logger.info("=== РАСШИРЕННЫЙ АНАЛИЗ ===")
        analysis_results = advanced_analysis(clean_df, cfg)

    # Анализ паттернов букв если задано в конфигурации
    if cfg.analyze_patterns:
        logger.info("=== АНАЛИЗ ПАТТЕРНОВ БУКВ ===")
        pattern_results = analyze_letter_patterns(clean_df, cfg)

    # Сравнительный анализ языков если задано в конфигурации
    if cfg.compare_languages:
        logger.info("=== СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЯЗЫКОВ ===")
        comparison_df = compare_languages(clean_df, cfg.top_n_languages, cfg)

    # Визуализация данных
    logger.info("=== СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ ===")

    # Загрузка конфигурации визуализации
    visualization_config_path = "configs/parse_anglicism/visualization/main.yaml"
    if os.path.exists(visualization_config_path):
        try:
            viz_cfg = OmegaConf.load(visualization_config_path)
            logger.info(f"Загружена конфигурация визуализации: {visualization_config_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации визуализации: {e}")
            viz_cfg = None
    else:
        logger.warning(f"Файл конфигурации визуализации не найден: {visualization_config_path}")
        viz_cfg = None

    # Создаем визуализации
    visualization_dir = paths["visualization_dir"]
    visualize_anglicisms(clean_df, visualization_dir, viz_cfg)

    # Вывод примеров англицизмов
    logger.info("\nПримеры англицизмов:")
    for word in clean_df['word'].head(10).tolist():
        logger.info(f"  {word}")

    logger.info("Анализ англицизмов завершен успешно")
    return clean_df


if __name__ == "__main__":
    # Запуск через Hydra
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Главный модуль для запуска процесса разметки англицизмов в текстах.

Модуль использует библиотеку Hydra для управления конфигурацией
и запускает процесс разметки англицизмов на основе предоставленного
списка и предобработанных текстов.
"""

import os
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модули для разметки
from label_data.utils.labeler import AngliсismLabeler


@hydra.main(
    version_base=None,
    config_path="../configs/label_data",
    config_name="main"
)
def main(config: DictConfig) -> None:
    """
    Основная функция для запуска разметки англицизмов.

    Создает экземпляр разметчика AngliсismLabeler с заданной
    конфигурацией и запускает процесс разметки и анализа данных.
    Сохраняет результаты в указанные директории.

    Args:
        config: Конфигурационный объект Hydra, содержащий все
               необходимые настройки для работы разметчика

    Returns:
        None
    """
    # Создаем необходимые директории
    for dir_path in [config.paths.output_dir, config.paths.analysis_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Инициализация разметчика с загруженной конфигурацией
    labeler = AngliсismLabeler(config)

    # Загрузка данных
    labeler.load_data()

    # Выполнение разметки
    binary_results, bio_results, stats = labeler.label_data()

    # Анализ и визуализация результатов
    if config.labeler.visualization.enabled:
        labeler.analyze_results(binary_results, stats, config.paths.analysis_dir)

    # Вывод результатов разметки
    print(f"Разметка завершена.")
    print(f"Всего обработано текстов: {stats['total_texts']}")
    print(f"Всего токенов: {stats['total_tokens']}")
    print(f"Всего англицизмов: {stats['total_anglicisms']}")
    print(f"Процент англицизмов: {stats['anglicism_percentage']:.2f}%")
    print(f"Количество текстов с англицизмами: {stats['texts_with_anglicisms']} ({stats['texts_with_anglicisms_percentage']:.2f}%)")
    print(f"Результаты сохранены в директории: {config.paths.output_dir}")
    print(f"Анализ сохранен в директории: {config.paths.analysis_dir}")


if __name__ == "__main__":
    # Настраиваем переменные окружения для Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_LOGGING.LEVEL"] = "WARN"  # Уменьшаем вывод логов Hydra

    # Запуск через Hydra
    main()
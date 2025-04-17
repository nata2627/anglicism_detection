#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модули для анализа
from utils.parser import parse_anglicisms
from utils.analyzer import analyze_anglicisms, clean_anglicisms, advanced_analysis
from utils.visualizer import visualize_anglicisms
from utils.io_utils import save_anglicisms, setup_directory_structure


def main():
    # Настройка путей с абсолютными значениями
    paths = {
        "data_dir": "assets",
        "output_dir": "assets",
        "logs_dir": "logs",
        "visualization_dir": "assets/visualization"
    }

    # Создание структуры директорий
    setup_directory_structure(paths)

    print("Запуск анализа англицизмов")

    # Путь к файлу с англицизмами
    input_file = os.path.join(paths["data_dir"], "angl.txt")

    # Проверка наличия входного файла
    if not os.path.exists(input_file):
        print(f"Файл с англицизмами не найден: {input_file}")
        return

    # Парсинг англицизмов
    print(f"Чтение файла: {input_file}")
    anglicisms_dict = parse_anglicisms(input_file)

    if not anglicisms_dict["all_anglicisms"]:
        print("Не удалось найти англицизмы в файле.")
        return

    # Базовый анализ данных
    print("Выполнение базового анализа")
    df = analyze_anglicisms(anglicisms_dict)

    # Очистка и нормализация данных
    print("Очистка и нормализация данных")
    clean_df = clean_anglicisms(df)

    # Сохранение обработанных англицизмов
    output_file = os.path.join(paths["data_dir"], "clean_anglicism.txt")
    csv_output = os.path.join(paths["data_dir"], "anglicism_stats.csv")
    save_anglicisms(clean_df, output_file, csv_output)

    # Логгируем информацию о сохранении файлов
    print(f"Сохранен список англицизмов: {output_file}")
    print(f"Сохранен CSV файл с анализом: {csv_output}")

    # Выполняем расширенный анализ данных
    print("Выполнение расширенного анализа")
    advanced_analysis(clean_df)

    # Визуализация данных
    print("Создание визуализаций")
    visualize_anglicisms(clean_df, paths["visualization_dir"])
    print(f"Визуализации сохранены в: {paths['visualization_dir']}")

    print("Анализ англицизмов успешно завершен")
    return clean_df


if __name__ == "__main__":
    main()
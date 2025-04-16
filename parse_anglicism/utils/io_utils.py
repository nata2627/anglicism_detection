#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
from utils.parser import clean_wiki_markup

logger = logging.getLogger(__name__)


def setup_directory_structure(paths_config):
    """
    Создает структуру директорий проекта.

    Args:
        paths_config: Конфигурация путей
    """
    for path_name, path in paths_config.items():
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Создана директория: {path}")

    # Создание директории для изображений в outputs
    images_dir = os.path.join(paths_config["output_dir"], "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        logger.info(f"Создана директория для изображений: {images_dir}")


def save_anglicisms(df, output_file, excel_output=None):
    """
    Сохраняет обработанные англицизмы в файл.

    Args:
        df (DataFrame): DataFrame с англицизмами
        output_file (str): Путь к файлу для сохранения
        excel_output (str, optional): Путь для сохранения полных данных в Excel
    """
    # Проверяем наличие директории для output_file
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Создана директория для выходного файла: {output_dir}")

    # Сохраняем только колонку с самими словами
    df[['word']].to_csv(output_file, index=False, header=False)
    logger.info(f"Англицизмы сохранены в файл: {output_file}")

    # Если указан путь для Excel, сохраняем все данные в Excel
    if excel_output:
        # Проверяем наличие директории для excel_output
        excel_dir = os.path.dirname(excel_output)
        if not os.path.exists(excel_dir):
            os.makedirs(excel_dir)
            logger.info(f"Создана директория для Excel-файла: {excel_dir}")

        # Создаем копию DataFrame с наиболее важными колонками
        export_df = df[['word', 'origin_language', 'word_length']].copy()

        # Добавляем колонку 'through_english' если она существует
        if 'through_english' in df.columns:
            export_df['through_english'] = df['through_english']

        # Переименовываем колонки для удобства
        export_df.columns = ['Англицизм', 'Язык происхождения', 'Длина слова'] + \
                            (['Через английский'] if 'through_english' in df.columns else [])

        # Очищаем данные от вики-разметки в колонке "Язык происхождения"
        export_df['Язык происхождения'] = export_df['Язык происхождения'].apply(clean_wiki_markup)

        # Преобразуем булево значение в текст (если колонка существует)
        if 'Через английский' in export_df.columns:
            export_df['Через английский'] = export_df['Через английский'].map({True: 'Да', False: 'Нет'})

        # Сохраняем в Excel
        try:
            export_df.to_excel(excel_output, index=False, sheet_name='Англицизмы')
            logger.info(f"Расширенные данные сохранены в Excel: {excel_output}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении Excel-файла: {e}")


def load_anglicisms(file_path):
    """
    Загружает список англицизмов из файла.

    Args:
        file_path (str): Путь к файлу с англицизмами

    Returns:
        list: Список англицизмов
    """
    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            anglicisms = [line.strip() for line in file.readlines() if line.strip()]
        logger.info(f"Загружено англицизмов из файла {file_path}: {len(anglicisms)}")
        return anglicisms
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {file_path}: {e}")
        return []


def load_anglicisms_excel(file_path):
    """
    Загружает данные англицизмов из Excel-файла.

    Args:
        file_path (str): Путь к Excel-файлу

    Returns:
        DataFrame: DataFrame с данными об англицизмах
    """
    if not os.path.exists(file_path):
        logger.error(f"Excel-файл не найден: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(file_path)
        logger.info(f"Загружено записей из Excel-файла {file_path}: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Ошибка при чтении Excel-файла {file_path}: {e}")
        return pd.DataFrame()
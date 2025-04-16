#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import logging
from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs/parse_anglicism/analysis",
    config_name="main"
)
def parse_anglicisms_with_config(cfg: DictConfig):
    """
    Парсит англицизмы из файла формата Викисловаря с использованием конфигурации.

    Args:
        cfg (DictConfig): Конфигурация Hydra

    Returns:
        dict: Словарь англицизмов с их происхождением
    """
    # Определяем путь к файлу из конфигурации
    if hasattr(cfg, 'paths') and hasattr(cfg.paths, 'data_dir'):
        file_path = f"{cfg.paths.data_dir}/angl.txt"
    else:
        file_path = 'data/angl.txt'

    logger.info(f"Парсинг англицизмов из файла {file_path} с использованием конфигурации")
    return parse_anglicisms(file_path, cfg)


def parse_anglicisms(file_path, cfg=None):
    """
    Парсит англицизмы из файла формата Викисловаря.

    Args:
        file_path (str): Путь к файлу с англицизмами
        cfg (DictConfig, optional): Конфигурация Hydra

    Returns:
        dict: Словарь англицизмов с их происхождением
    """
    # Загрузка конфигурации, если она не передана
    if cfg is None:
        # Используем значения по умолчанию
        language_section_pattern = r'== Из (.*?) =='
        anglicism_pattern = r'\[\[(.*?)\]\](.*?(?=\[\[|$))'
        through_english_pattern = r'через англ'
    else:
        # Проверяем наличие секции patterns
        if hasattr(cfg, 'patterns'):
            # Загружаем из конфигурации или используем значения по умолчанию
            language_section_pattern = getattr(cfg.patterns, 'language_section', r'== Из (.*?) ==')
            anglicism_pattern = getattr(cfg.patterns, 'anglicism', r'\[\[(.*?)\]\](.*?(?=\[\[|$))')
            through_english_pattern = getattr(cfg.patterns, 'through_english', r'через англ')
        else:
            # Если секции patterns нет, используем значения по умолчанию
            language_section_pattern = r'== Из (.*?) =='
            anglicism_pattern = r'\[\[(.*?)\]\](.*?(?=\[\[|$))'
            through_english_pattern = r'через англ'

    logger.info(f"Чтение файла: {file_path}")

    # Чтение файла
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        logger.error(f"Файл не найден: {file_path}")
        return {"by_language": {}, "all_anglicisms": []}
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        return {"by_language": {}, "all_anglicisms": []}

    # Регулярное выражение для поиска разделов с языками
    language_sections = re.findall(language_section_pattern, content)

    # Словарь для хранения англицизмов по языкам происхождения
    anglicisms_by_language = defaultdict(list)

    # Текущий раздел языка
    current_language = None

    # Обработка строк файла
    for line in content.split('\n'):
        # Проверка, является ли строка заголовком раздела
        language_match = re.search(language_section_pattern, line)
        if language_match:
            current_language = language_match.group(1)
            logger.debug(f"Обнаружен раздел языка: {current_language}")
            continue

        # Если текущий язык определен, ищем англицизмы
        if current_language:
            matches = re.findall(anglicism_pattern, line)
            for match in matches:
                word = match[0]
                description = match[1].strip()

                # Проверяем, содержит ли описание упоминание английского языка
                is_through_english = bool(re.search(through_english_pattern, description))

                # Если слово содержит |, берем только часть до |
                if '|' in word:
                    word = word.split('|')[0]

                # Очистка названия языка от викиразметки
                clean_language = clean_wiki_markup(current_language)

                anglicisms_by_language[clean_language].append({
                    'word': word.strip(),
                    'description': description,
                    'through_english': is_through_english
                })

    # Создаем общий список всех англицизмов
    all_anglicisms = []
    for language, words in anglicisms_by_language.items():
        for word_info in words:
            all_anglicisms.append({
                'word': word_info['word'],
                'origin_language': language,
                'description': word_info['description'],
                'through_english': word_info['through_english']
            })

    logger.info(f"Обнаружено англицизмов: {len(all_anglicisms)}")
    logger.info(f"Обнаружено языков происхождения: {len(anglicisms_by_language)}")

    return {
        'by_language': anglicisms_by_language,
        'all_anglicisms': all_anglicisms
    }


def clean_wiki_markup(text):
    """
    Очищает текст от викиразметки.

    Args:
        text (str): Текст с викиразметкой

    Returns:
        str: Очищенный текст
    """
    # Удаление двойных квадратных скобок и содержимого между вертикальной чертой и закрывающими скобками
    text = re.sub(r'\[\[([^|]+)\|[^\]]+\]\]', r'\1', text)

    # Удаление двойных квадратных скобок без вертикальной черты
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Удаление одиночных квадратных скобок
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)

    # Удаление других возможных элементов разметки
    text = re.sub(r'<[^>]+>', '', text)

    return text


if __name__ == "__main__":
    parse_anglicisms_with_config()
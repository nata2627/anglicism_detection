#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Пакет утилит для анализа англицизмов.

Модули:
    - parser: Парсинг англицизмов из файлов
    - analyzer: Анализ данных об англицизмах
    - visualizer: Визуализация данных об англицизмах
    - io_utils: Утилиты ввода-вывода
"""

from utils.parser import parse_anglicisms
from utils.analyzer import analyze_anglicisms, clean_anglicisms, advanced_analysis, compare_languages, analyze_letter_patterns
from utils.visualizer import visualize_anglicisms
from utils.io_utils import save_anglicisms, load_anglicisms, load_anglicisms_excel, setup_directory_structure

__all__ = [
    'parse_anglicisms',
    'analyze_anglicisms',
    'clean_anglicisms',
    'advanced_analysis',
    'compare_languages',
    'analyze_letter_patterns',
    'visualize_anglicisms',
    'save_anglicisms',
    'load_anglicisms',
    'load_anglicisms_excel',
    'setup_directory_structure'
]
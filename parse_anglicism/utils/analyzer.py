#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
from collections import Counter
import re
import hydra
from omegaconf import DictConfig

from utils.parser import clean_wiki_markup

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs/analysis",
    config_name="main"
)
def analyze_with_config(cfg: DictConfig):
    """
    Запускает анализ с конфигурацией Hydra.
    Эта функция используется при запуске модуля напрямую.

    Args:
        cfg (DictConfig): Конфигурация Hydra
    """
    # Здесь можно добавить код для запуска анализа через Hydra
    pass


def analyze_anglicisms(anglicisms_dict):
    """
    Анализирует словарь англицизмов и выводит статистику.

    Args:
        anglicisms_dict (dict): Словарь англицизмов

    Returns:
        DataFrame: DataFrame с проанализированными данными
    """
    all_anglicisms = anglicisms_dict['all_anglicisms']
    by_language = anglicisms_dict['by_language']

    logger.info(f"Всего найдено англицизмов: {len(all_anglicisms)}")
    logger.info("Распределение по языкам происхождения:")

    for language, words in by_language.items():
        logger.info(f"  {language}: {len(words)} слов")

    # Создаем DataFrame для более удобного анализа
    df = pd.DataFrame(all_anglicisms)

    # Очистка данных от викиразметки в DataFrame
    df['origin_language'] = df['origin_language'].apply(clean_wiki_markup)
    df['word'] = df['word'].apply(clean_wiki_markup)

    # Количество англицизмов, пришедших через английский
    through_english_count = df['through_english'].sum()
    logger.info(
        f"Англицизмов, пришедших через английский: {through_english_count} ({through_english_count / len(df) * 100:.2f}%)")

    # Проверка на дубликаты слов
    duplicates = df[df.duplicated('word', keep=False)]
    if not duplicates.empty:
        logger.info(f"Найдены дубликаты слов ({len(duplicates)}):")
        for word in duplicates['word'].unique():
            logger.debug(f"  {word}")

    # Анализ длины слов
    df['word_length'] = df['word'].apply(len)
    avg_length = df['word_length'].mean()
    logger.info(f"Средняя длина англицизма: {avg_length:.2f} символов")

    # Топ-10 самых длинных слов
    longest_words = df.nlargest(10, 'word_length')
    logger.info("Топ-10 самых длинных англицизмов:")
    for _, row in longest_words.iterrows():
        logger.info(f"  {row['word']} ({row['word_length']} символов)")

    # Анализ по первым буквам
    df['first_letter'] = df['word'].str[0]
    first_letter_counts = df['first_letter'].value_counts().head(10)
    logger.info("Топ-10 самых частых первых букв в англицизмах:")
    for letter, count in first_letter_counts.items():
        logger.info(f"  {letter}: {count} слов")

    # Анализ частотности букв
    all_letters = ''.join(df['word'].str.lower())
    letter_counts = Counter(all_letters)
    logger.info("Топ-10 самых частых букв в англицизмах:")
    for letter, count in letter_counts.most_common(10):
        logger.info(f"  {letter}: {count} вхождений")

    return df


def clean_anglicisms(df, cfg=None):
    """
    Очищает и нормализует список англицизмов.

    Args:
        df (DataFrame): DataFrame с англицизмами
        cfg (DictConfig, optional): Конфигурация для очистки

    Returns:
        DataFrame: Очищенный DataFrame
    """
    # Копия DataFrame для безопасного изменения
    clean_df = df.copy()

    # Настройки по умолчанию
    lowercase = True
    remove_special_chars = True
    remove_duplicates = True

    # Загрузка настроек из конфигурации, если она передана
    if cfg is not None and hasattr(cfg, 'cleaning'):
        lowercase = cfg.cleaning.lowercase
        remove_special_chars = cfg.cleaning.remove_special_chars
        remove_duplicates = cfg.cleaning.remove_duplicates

    # Приведение всех слов к нижнему регистру
    if lowercase:
        clean_df['word'] = clean_df['word'].str.lower()

    # Удаление лишних символов (например, двоеточий, запятых)
    if remove_special_chars:
        clean_df['word'] = clean_df['word'].str.replace(r'[^\w\s]', '', regex=True)

    # Удаление дубликатов
    if remove_duplicates:
        clean_df = clean_df.drop_duplicates('word')

    # Обновление длины слов после очистки
    clean_df['word_length'] = clean_df['word'].apply(len)

    logger.info(f"После очистки осталось англицизмов: {len(clean_df)}")

    return clean_df


def advanced_analysis(df, cfg=None):
    """
    Проводит расширенный анализ англицизмов.

    Args:
        df (DataFrame): DataFrame с англицизмами
        cfg (DictConfig, optional): Конфигурация для анализа

    Returns:
        dict: Словарь с результатами анализа
    """
    # Создаем словарь для хранения результатов анализа
    analysis_results = {}

    # Настройки категорий длины слов
    bins = [0, 4, 8, 12, 100]
    labels = ['Короткие (1-4)', 'Средние (5-8)', 'Длинные (9-12)', 'Очень длинные (>12)']

    # Загрузка настроек из конфигурации, если она передана
    if cfg is not None and hasattr(cfg, 'length_categories'):
        if hasattr(cfg.length_categories, 'bins'):
            bins = cfg.length_categories.bins
        if hasattr(cfg.length_categories, 'labels'):
            labels = cfg.length_categories.labels

    # 1. Анализ длины слов по языкам
    length_by_lang = df.groupby('origin_language')['word_length'].agg(['mean', 'median', 'min', 'max', 'count'])
    length_by_lang = length_by_lang.sort_values('count', ascending=False)
    analysis_results['length_by_language'] = length_by_lang

    logger.info("Средняя длина слов по языкам происхождения (топ-10 по количеству):")
    logger.info(length_by_lang.head(10).to_string())

    # 2. Анализ англицизмов, пришедших через английский, по языкам
    if 'through_english' in df.columns:
        through_eng_by_lang = df.groupby('origin_language')['through_english'].mean() * 100
        through_eng_by_lang = through_eng_by_lang.sort_values(ascending=False)
        analysis_results['through_english_by_language'] = through_eng_by_lang

        logger.info("Доля англицизмов, пришедших через английский, по языкам происхождения (%):")
        for lang, percentage in through_eng_by_lang.head(10).items():
            logger.info(f"  {lang}: {percentage:.2f}%")

    # 3. Анализ частотности букв
    all_letters = ''.join(df['word'].str.lower())
    letter_freq = pd.Series(Counter(all_letters)).sort_values(ascending=False)
    analysis_results['letter_frequency'] = letter_freq

    # 4. Распределение длины описаний
    if 'description' in df.columns:
        df['description_length'] = df['description'].str.len()
        desc_length_stats = df['description_length'].describe()
        analysis_results['description_length_stats'] = desc_length_stats

        logger.info("Статистика длины описаний англицизмов:")
        logger.info(desc_length_stats.to_string())

    # 5. Анализ уникальности слов по языкам
    unique_words_by_lang = df.groupby('origin_language')['word'].nunique()
    total_words_by_lang = df.groupby('origin_language')['word'].count()
    uniqueness_ratio = unique_words_by_lang / total_words_by_lang * 100
    uniqueness_ratio = uniqueness_ratio.sort_values(ascending=False)
    analysis_results['uniqueness_ratio_by_language'] = uniqueness_ratio

    logger.info("Коэффициент уникальности слов по языкам (%):")
    for lang, ratio in uniqueness_ratio.head(10).items():
        logger.info(f"  {lang}: {ratio:.2f}%")

    # 6. Создаем новую колонку для категоризации длины слов
    df['length_category'] = pd.cut(df['word_length'], bins=bins, labels=labels, right=False)

    length_category_counts = df['length_category'].value_counts().sort_index()
    analysis_results['length_category_counts'] = length_category_counts

    logger.info("Распределение англицизмов по категориям длины:")
    for category, count in length_category_counts.items():
        logger.info(f"  {category}: {count} слов ({count / len(df) * 100:.2f}%)")

    return analysis_results


def compare_languages(df, top_n=5, cfg=None):
    """
    Проводит сравнительный анализ англицизмов по языкам происхождения.

    Args:
        df (DataFrame): DataFrame с англицизмами
        top_n (int): Количество языков для сравнения
        cfg (DictConfig, optional): Конфигурация для сравнения

    Returns:
        DataFrame: DataFrame со сравнительными данными
    """
    # Загрузка настроек из конфигурации, если она передана
    if cfg is not None and hasattr(cfg, 'compare_languages'):
        if hasattr(cfg.compare_languages, 'top_n_languages'):
            top_n = cfg.compare_languages.top_n_languages

    # Получаем топ-N языков по количеству англицизмов
    top_languages = df['origin_language'].value_counts().head(top_n).index.tolist()

    # Фильтруем DataFrame по этим языкам
    filtered_df = df[df['origin_language'].isin(top_languages)]

    # Создаем сводную таблицу для сравнения
    comparison_data = []

    for lang in top_languages:
        lang_df = filtered_df[filtered_df['origin_language'] == lang]

        if not lang_df.empty:
            min_idx = lang_df['word_length'].idxmin() if not lang_df.empty else None
            max_idx = lang_df['word_length'].idxmax() if not lang_df.empty else None

            comparison_data.append({
                'Язык': lang,
                'Количество слов': len(lang_df),
                'Средняя длина': lang_df['word_length'].mean(),
                'Медиана длины': lang_df['word_length'].median(),
                'Минимальная длина': lang_df['word_length'].min(),
                'Максимальная длина': lang_df['word_length'].max(),
                'Через английский (%)': lang_df[
                                            'through_english'].mean() * 100 if 'through_english' in lang_df.columns else 0,
                'Самое короткое слово': lang_df.loc[min_idx, 'word'] if min_idx is not None else 'N/A',
                'Самое длинное слово': lang_df.loc[max_idx, 'word'] if max_idx is not None else 'N/A',
                'Самый популярный префикс': lang_df['word'].str[:2].value_counts().index[
                    0] if not lang_df.empty and len(lang_df) > 0 else 'N/A'
            })

    comparison_df = pd.DataFrame(comparison_data)

    logger.info("=== СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЯЗЫКОВ ===")
    logger.info(comparison_df.to_string(index=False))

    return comparison_df


def analyze_letter_patterns(df, cfg=None):
    """
    Анализирует паттерны букв в англицизмах.

    Args:
        df (DataFrame): DataFrame с англицизмами
        cfg (DictConfig, optional): Конфигурация для анализа паттернов

    Returns:
        dict: Словарь с результатами анализа
    """
    pattern_results = {}

    # Настройки по умолчанию
    prefix_length = 2
    suffix_length = 2
    top_n = 10

    # Загрузка настроек из конфигурации, если она передана
    if cfg is not None and hasattr(cfg, 'letter_patterns'):
        if hasattr(cfg.letter_patterns, 'prefix_length'):
            prefix_length = cfg.letter_patterns.prefix_length
        if hasattr(cfg.letter_patterns, 'suffix_length'):
            suffix_length = cfg.letter_patterns.suffix_length
        if hasattr(cfg.letter_patterns, 'top_n'):
            top_n = cfg.letter_patterns.top_n

    # 1. Анализ суффиксов
    df['suffix'] = df['word'].apply(lambda x: x[-suffix_length:] if len(x) >= suffix_length else x)
    suffix_counts = df['suffix'].value_counts().head(top_n)
    pattern_results['top_suffixes'] = suffix_counts

    logger.info(f"Топ-{top_n} самых частых суффиксов (последние {suffix_length} буквы):")
    for suffix, count in suffix_counts.items():
        logger.info(f"  '{suffix}': {count} слов")

    # 2. Анализ префиксов
    df['prefix'] = df['word'].apply(lambda x: x[:prefix_length] if len(x) >= prefix_length else x)
    prefix_counts = df['prefix'].value_counts().head(top_n)
    pattern_results['top_prefixes'] = prefix_counts

    logger.info(f"Топ-{top_n} самых частых префиксов (первые {prefix_length} буквы):")
    for prefix, count in prefix_counts.items():
        logger.info(f"  '{prefix}': {count} слов")

    # 3. Биграммы (пары последовательных букв)
    bigrams = []
    for word in df['word']:
        for i in range(len(word) - 1):
            bigrams.append(word[i:i + 2])

    bigram_counts = pd.Series(Counter(bigrams)).sort_values(ascending=False).head(top_n)
    pattern_results['top_bigrams'] = bigram_counts

    logger.info(f"Топ-{top_n} самых частых биграмм (пар букв):")
    for bigram, count in bigram_counts.items():
        logger.info(f"  '{bigram}': {count} вхождений")

    return pattern_results


if __name__ == "__main__":
    analyze_with_config()
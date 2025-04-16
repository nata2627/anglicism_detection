import pandas as pd
import logging
from collections import Counter

from utils.parser import clean_wiki_markup

logger = logging.getLogger(__name__)


def analyze_anglicisms(anglicisms_dict):
    all_anglicisms = anglicisms_dict['all_anglicisms']

    # Создаем DataFrame для анализа
    df = pd.DataFrame(all_anglicisms)

    # Очистка данных от викиразметки в DataFrame
    df['origin_language'] = df['origin_language'].apply(clean_wiki_markup)
    df['word'] = df['word'].apply(clean_wiki_markup)

    # Анализ длины слов
    df['word_length'] = df['word'].apply(len)

    return df


def clean_anglicisms(df, cfg=None):
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

    # Дополнительная очистка языка происхождения
    clean_df['origin_language'] = clean_df['origin_language'].apply(clean_wiki_markup)

    # Удаление дубликатов
    if remove_duplicates:
        clean_df = clean_df.drop_duplicates('word')

    # Обновление длины слов после очистки
    clean_df['word_length'] = clean_df['word'].apply(len)

    return clean_df


def advanced_analysis(df, cfg=None):
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

    # 2. Анализ англицизмов, пришедших через английский, по языкам
    if 'through_english' in df.columns:
        through_eng_by_lang = df.groupby('origin_language')['through_english'].mean() * 100
        through_eng_by_lang = through_eng_by_lang.sort_values(ascending=False)
        analysis_results['through_english_by_language'] = through_eng_by_lang

    # 3. Анализ частотности букв
    all_letters = ''.join(df['word'].str.lower())
    letter_freq = pd.Series(Counter(all_letters)).sort_values(ascending=False)
    analysis_results['letter_frequency'] = letter_freq

    # 4. Создаем новую колонку для категоризации длины слов
    df['length_category'] = pd.cut(df['word_length'], bins=bins, labels=labels, right=False)
    length_category_counts = df['length_category'].value_counts().sort_index()
    analysis_results['length_category_counts'] = length_category_counts

    return analysis_results
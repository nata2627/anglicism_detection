#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.ticker as mticker
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs/visualization",
    config_name="main"
)
def visualize_with_config(cfg: DictConfig):
    """
    Запускает визуализацию с конфигурацией Hydra.

    Args:
        cfg (DictConfig): Конфигурация Hydra
    """
    # Код для запуска визуализации через Hydra напрямую
    pass


def visualize_anglicisms(df, output_dir="visualization", cfg=None):
    """
    Создает визуализации для анализа англицизмов.

    Args:
        df (DataFrame): DataFrame с англицизмами
        output_dir (str): Директория для сохранения графиков
        cfg (DictConfig, optional): Конфигурация для визуализации
    """
    # Создаем директорию для сохранения визуализаций, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Создана директория для визуализаций: {output_dir}")

    # Настройка стиля графиков
    sns.set(style="whitegrid")

    # Настройки из конфигурации (или значения по умолчанию)
    dpi = 300
    font_family = 'DejaVu Sans'
    save_formats = ['png']

    if cfg is not None and hasattr(cfg, 'visualization'):
        dpi = cfg.visualization.dpi if hasattr(cfg.visualization, 'dpi') else dpi
        font_family = cfg.visualization.font_family if hasattr(cfg.visualization, 'font_family') else font_family
        if hasattr(cfg.visualization, 'save_formats'):
            save_formats = cfg.visualization.save_formats

    # Настраиваем поддержку русского языка в matplotlib
    plt.rcParams['font.family'] = font_family

    # Функция для сохранения графиков в разных форматах
    def save_figure(name):
        for fmt in save_formats:
            file_path = f"{output_dir}/{name}.{fmt}"
            plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
            logger.debug(f"Сохранен файл: {file_path}")

    # 1. Круговая диаграмма распределения по языкам происхождения
    logger.info("Создание круговой диаграммы распределения по языкам происхождения")
    plt.figure(figsize=(12, 8))
    lang_counts = df['origin_language'].value_counts()

    # Выделяем топ-8 языков, остальные группируем как "Другие"
    if len(lang_counts) > 8:
        top_langs = lang_counts.head(8)
        others = pd.Series({'Другие': lang_counts[8:].sum()})
        lang_counts = pd.concat([top_langs, others])

    plt.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%',
            startangle=90, shadow=True, textprops={'fontsize': 12})
    plt.title('Распределение англицизмов по языкам происхождения', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    save_figure("languages_pie_chart")
    plt.close()

    # 2. Гистограмма длин слов
    logger.info("Создание гистограммы длин слов")
    plt.figure(figsize=(12, 8))
    sns.histplot(df['word_length'], kde=True, bins=20)
    plt.title('Распределение длин англицизмов', fontsize=16)
    plt.xlabel('Количество символов', fontsize=14)
    plt.ylabel('Количество слов', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_figure("word_length_histogram")
    plt.close()

    # 3. Средняя длина слов по языкам
    logger.info("Создание графика средней длины слов по языкам")
    plt.figure(figsize=(14, 10))
    avg_length_by_lang = df.groupby('origin_language')['word_length'].mean().sort_values(ascending=False)

    # Выделяем топ-15 языков для лучшей читаемости
    if len(avg_length_by_lang) > 15:
        avg_length_by_lang = avg_length_by_lang.head(15)

    ax = sns.barplot(x=avg_length_by_lang.index, y=avg_length_by_lang.values)
    plt.title('Средняя длина англицизмов по языкам происхождения', fontsize=16)
    plt.xlabel('Язык происхождения', fontsize=14)
    plt.ylabel('Средняя длина (символов)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Добавляем значения над столбцами
    for i, v in enumerate(avg_length_by_lang.values):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=10)

    plt.tight_layout()
    save_figure("avg_length_by_language")
    plt.close()

    # 4. Количество слов по языкам (горизонтальный барплот)
    logger.info("Создание горизонтального барплота количества слов по языкам")
    plt.figure(figsize=(14, 10))
    words_by_lang = df['origin_language'].value_counts().sort_values(ascending=True)

    # Выделяем топ-15 языков для лучшей читаемости
    if len(words_by_lang) > 15:
        words_by_lang = words_by_lang.tail(15)

    ax = sns.barplot(x=words_by_lang.values, y=words_by_lang.index)
    plt.title('Количество англицизмов по языкам происхождения', fontsize=16)
    plt.xlabel('Количество слов', fontsize=14)
    plt.ylabel('Язык происхождения', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Добавляем значения к столбцам
    for i, v in enumerate(words_by_lang.values):
        ax.text(v + 1, i, str(v), va='center', fontsize=10)

    plt.tight_layout()
    save_figure("words_count_by_language")
    plt.close()

    # 5. Визуализация через/не через английский
    if 'through_english' in df.columns:
        logger.info("Создание визуализации англицизмов через/не через английский")
        plt.figure(figsize=(10, 8))
        through_english = df['through_english'].value_counts()
        labels = ['Напрямую', 'Через английский']
        if len(through_english) == 1:  # Если есть только одна категория
            if through_english.index[0]:
                values = [0, through_english.iloc[0]]
            else:
                values = [through_english.iloc[0], 0]
        else:
            values = [through_english.get(False, 0), through_english.get(True, 0)]

        plt.bar(labels, values, color=['#3498db', '#e74c3c'])
        plt.title('Англицизмы: напрямую vs через английский', fontsize=16)
        plt.ylabel('Количество слов', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Добавляем процентные значения над столбцами
        total = sum(values)
        for i, v in enumerate(values):
            plt.text(i, v + 5, f'{v} ({v / total * 100:.1f}%)', ha='center', fontsize=12)

        plt.tight_layout()
        save_figure("through_english")
        plt.close()

    # 6. Частотность первых букв (топ-10)
    logger.info("Создание графика частотности первых букв")
    plt.figure(figsize=(12, 8))
    first_letters = df['word'].str[0].value_counts().sort_values(ascending=False).head(10)
    ax = sns.barplot(x=first_letters.index, y=first_letters.values)
    plt.title('Топ-10 самых частых первых букв в англицизмах', fontsize=16)
    plt.xlabel('Буква', fontsize=14)
    plt.ylabel('Количество слов', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Добавляем значения над столбцами
    for i, v in enumerate(first_letters.values):
        ax.text(i, v + 1, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    save_figure("first_letters_frequency")
    plt.close()

    # 7. Word Cloud всех англицизмов
    logger.info("Создание облака слов англицизмов")
    plt.figure(figsize=(14, 10))
    wordcloud = WordCloud(width=1200, height=800,
                          background_color='white',
                          max_words=200,
                          colormap='viridis',
                          contour_width=1,
                          contour_color='steelblue').generate(' '.join(df['word']))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако слов англицизмов', fontsize=16, pad=20)
    plt.tight_layout()
    save_figure("anglicisms_wordcloud")
    plt.close()

    # 8. Распределение длин слов по языкам (boxplot)
    logger.info("Создание boxplot распределения длин слов по языкам")
    plt.figure(figsize=(16, 10))

    # Выбираем только языки с достаточным количеством слов (>= 5)
    langs_with_enough_words = df['origin_language'].value_counts()[df['origin_language'].value_counts() >= 5].index
    filtered_df = df[df['origin_language'].isin(langs_with_enough_words)]

    sns.boxplot(x='origin_language', y='word_length', data=filtered_df)
    plt.title('Распределение длин слов по языкам происхождения', fontsize=16)
    plt.xlabel('Язык происхождения', fontsize=14)
    plt.ylabel('Длина слова (символов)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_figure("word_length_boxplot_by_language")
    plt.close()

    # 9. Тепловая карта корреляций (если есть числовые данные)
    num_columns = df.select_dtypes(include=[np.number]).columns
    if len(num_columns) > 1:
        logger.info("Создание тепловой карты корреляций")
        plt.figure(figsize=(10, 8))
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Корреляции между числовыми характеристиками', fontsize=16)
        plt.tight_layout()
        save_figure("correlation_heatmap")
        plt.close()

    # 10. Сводный отчет по доле англицизмов из разных языков
    logger.info("Создание сводного отчета по доле англицизмов из разных языков")
    plt.figure(figsize=(10, 8))
    lang_percentages = df['origin_language'].value_counts(normalize=True) * 100

    # Создаем сводный отчет в виде красивой таблицы
    plt.axis('tight')
    plt.axis('off')
    cell_text = [[f"{lang}", f"{pct:.2f}%"] for lang, pct in lang_percentages.items()]
    column_labels = ["Язык происхождения", "Процент англицизмов"]
    table = plt.table(cellText=cell_text, colLabels=column_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Стилизуем таблицу
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Заголовки
            cell.set_text_props(fontproperties=plt.matplotlib.font_manager.FontProperties(weight='bold'))
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    plt.title('Процентное распределение англицизмов по языкам', fontsize=16, pad=20)
    plt.tight_layout()
    save_figure("language_percentage_table")
    plt.close()

    logger.info(f"Визуализации сохранены в директорию: {output_dir}")


if __name__ == "__main__":
    visualize_with_config()
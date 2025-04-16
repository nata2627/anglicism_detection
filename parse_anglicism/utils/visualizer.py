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
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs/parse_anglicism/visualization",
    config_name="main"
)
def visualize_with_config(cfg: DictConfig):
    """
    Запускает визуализацию с конфигурацией Hydra.

    Args:
        cfg (DictConfig): Конфигурация Hydra
    """
    logger.info("Запуск визуализации с использованием конфигурации Hydra")

    # Здесь можно добавить код для загрузки DataFrame и вызова visualize_anglicisms
    # Например, загрузка из Excel-файла:
    try:
        excel_path = "outputs/anglicisms_analysis.xlsx"
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            output_dir = "outputs/visualization_standalone"
            visualize_anglicisms(df, output_dir, cfg)
        else:
            logger.error(f"Файл с данными не найден: {excel_path}")
    except Exception as e:
        logger.error(f"Ошибка при запуске визуализации: {e}")


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

    # Настройки по умолчанию
    dpi = 300
    font_family = 'DejaVu Sans'
    save_formats = ['png']
    style = "whitegrid"

    # Размеры графиков по умолчанию
    figure_sizes = {
        "default": (12, 8),
        "pie_chart": (12, 8),
        "histogram": (12, 8),
        "bar_plot": (14, 10),
        "hbar_plot": (14, 10),
        "word_cloud": (14, 10),
        "boxplot": (16, 10),
        "heatmap": (10, 8),
        "table": (10, 8)
    }

    # Цвета по умолчанию
    colors = {
        "primary": '#3498db',
        "secondary": '#e74c3c',
        "accent": '#2ecc71',
        "pie_cmap": 'viridis',
        "heatmap_cmap": 'coolwarm'
    }

    # Лимиты по умолчанию
    limits = {
        "top_languages_pie": 8,
        "top_languages_bar": 15,
        "min_words_per_language": 5,
        "max_words_wordcloud": 200
    }

    # Настройки таблицы по умолчанию
    table_settings = {
        "scale_x": 1.2,
        "scale_y": 1.5,
        "header_bgcolor": '#4CAF50',
        "header_color": 'white',
        "even_row_bgcolor": '#f0f0f0',
        "odd_row_bgcolor": 'white'
    }

    # Загрузка настроек из конфигурации, если она передана
    if cfg is not None and hasattr(cfg, 'visualization'):
        viz_cfg = cfg.visualization

        # Основные настройки
        if hasattr(viz_cfg, 'dpi'):
            dpi = viz_cfg.dpi
        if hasattr(viz_cfg, 'font_family'):
            font_family = viz_cfg.font_family
        if hasattr(viz_cfg, 'save_formats'):
            save_formats = viz_cfg.save_formats
        if hasattr(viz_cfg, 'style'):
            style = viz_cfg.style

        # Размеры графиков
        if hasattr(viz_cfg, 'figure_sizes'):
            for key, value in viz_cfg.figure_sizes.items():
                figure_sizes[key] = tuple(value)

        # Цвета
        if hasattr(viz_cfg, 'colors'):
            for key, value in viz_cfg.colors.items():
                colors[key] = value

        # Лимиты
        if hasattr(viz_cfg, 'limits'):
            for key, value in viz_cfg.limits.items():
                limits[key] = value

        # Настройки таблицы
        if hasattr(viz_cfg, 'table'):
            for key, value in viz_cfg.table.items():
                table_settings[key] = value

    # Настраиваем поддержку русского языка в matplotlib
    plt.rcParams['font.family'] = font_family

    # Настройка стиля графиков
    sns.set(style=style)

    # Функция для сохранения графиков в разных форматах
    def save_figure(name, tight_layout=True):
        if tight_layout:
            plt.tight_layout()

        for fmt in save_formats:
            file_path = f"{output_dir}/{name}.{fmt}"
            plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
            logger.debug(f"Сохранен файл: {file_path}")

    # Проверяем наличие необходимых колонок в DataFrame
    if 'word' not in df.columns:
        logger.error("В DataFrame отсутствует колонка 'word'")
        return

    # Убедимся, что у нас есть колонка word_length
    if 'word_length' not in df.columns:
        logger.info("Добавляем колонку word_length")
        df['word_length'] = df['word'].apply(len)

    # Убедимся, что у нас есть колонка first_letter
    if 'first_letter' not in df.columns and len(df) > 0:
        logger.info("Добавляем колонку first_letter")
        df['first_letter'] = df['word'].str[0]

    # 1. Круговая диаграмма распределения по языкам происхождения
    if 'origin_language' in df.columns:
        logger.info("Создание круговой диаграммы распределения по языкам происхождения")
        plt.figure(figsize=figure_sizes["pie_chart"])
        lang_counts = df['origin_language'].value_counts()

        # Выделяем топ-N языков, остальные группируем как "Другие"
        top_n = limits["top_languages_pie"]
        if len(lang_counts) > top_n:
            top_langs = lang_counts.head(top_n)
            others = pd.Series({'Другие': lang_counts[top_n:].sum()})
            lang_counts = pd.concat([top_langs, others])

        plt.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%',
                startangle=90, shadow=True, textprops={'fontsize': 12},
                colors=plt.cm.get_cmap(colors["pie_cmap"])(np.linspace(0, 1, len(lang_counts))))
        plt.title('Распределение англицизмов по языкам происхождения', fontsize=16)
        plt.axis('equal')
        save_figure("languages_pie_chart")
        plt.close()

    # 2. Гистограмма длин слов
    logger.info("Создание гистограммы длин слов")
    plt.figure(figsize=figure_sizes["histogram"])
    sns.histplot(df['word_length'], kde=True, bins=20, color=colors["primary"])
    plt.title('Распределение длин англицизмов', fontsize=16)
    plt.xlabel('Количество символов', fontsize=14)
    plt.ylabel('Количество слов', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure("word_length_histogram")
    plt.close()

    # 3. Средняя длина слов по языкам
    if 'origin_language' in df.columns:
        logger.info("Создание графика средней длины слов по языкам")
        plt.figure(figsize=figure_sizes["bar_plot"])
        avg_length_by_lang = df.groupby('origin_language')['word_length'].mean().sort_values(ascending=False)

        # Выделяем топ-N языков для лучшей читаемости
        top_n = limits["top_languages_bar"]
        if len(avg_length_by_lang) > top_n:
            avg_length_by_lang = avg_length_by_lang.head(top_n)

        ax = sns.barplot(x=avg_length_by_lang.index, y=avg_length_by_lang.values, palette='viridis')
        plt.title('Средняя длина англицизмов по языкам происхождения', fontsize=16)
        plt.xlabel('Язык происхождения', fontsize=14)
        plt.ylabel('Средняя длина (символов)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Добавляем значения над столбцами
        for i, v in enumerate(avg_length_by_lang.values):
            ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=10)

        save_figure("avg_length_by_language")
        plt.close()

    # 4. Количество слов по языкам (горизонтальный барплот)
    if 'origin_language' in df.columns:
        logger.info("Создание горизонтального барплота количества слов по языкам")
        plt.figure(figsize=figure_sizes["hbar_plot"])
        words_by_lang = df['origin_language'].value_counts().sort_values(ascending=True)

        # Выделяем топ-N языков для лучшей читаемости
        top_n = limits["top_languages_bar"]
        if len(words_by_lang) > top_n:
            words_by_lang = words_by_lang.tail(top_n)

        ax = sns.barplot(x=words_by_lang.values, y=words_by_lang.index, palette='viridis')
        plt.title('Количество англицизмов по языкам происхождения', fontsize=16)
        plt.xlabel('Количество слов', fontsize=14)
        plt.ylabel('Язык происхождения', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Добавляем значения к столбцам
        for i, v in enumerate(words_by_lang.values):
            ax.text(v + 1, i, str(v), va='center', fontsize=10)

        save_figure("words_count_by_language")
        plt.close()

    # 5. Визуализация через/не через английский
    if 'through_english' in df.columns:
        logger.info("Создание визуализации англицизмов через/не через английский")
        plt.figure(figsize=figure_sizes["default"])
        through_english = df['through_english'].value_counts()
        labels = ['Напрямую', 'Через английский']
        if len(through_english) == 1:  # Если есть только одна категория
            if through_english.index[0]:
                values = [0, through_english.iloc[0]]
            else:
                values = [through_english.iloc[0], 0]
        else:
            values = [through_english.get(False, 0), through_english.get(True, 0)]

        plt.bar(labels, values, color=[colors["primary"], colors["secondary"]])
        plt.title('Англицизмы: напрямую vs через английский', fontsize=16)
        plt.ylabel('Количество слов', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Добавляем процентные значения над столбцами
        total = sum(values)
        for i, v in enumerate(values):
            if total > 0:  # Избегаем деления на ноль
                plt.text(i, v + 5, f'{v} ({v / total * 100:.1f}%)', ha='center', fontsize=12)
            else:
                plt.text(i, v + 5, f'{v} (0.0%)', ha='center', fontsize=12)

        save_figure("through_english")
        plt.close()

    # 6. Частотность первых букв (топ-10)
    logger.info("Создание графика частотности первых букв")
    plt.figure(figsize=figure_sizes["default"])
    first_letters = df['word'].str[0].value_counts().sort_values(ascending=False).head(10)
    ax = sns.barplot(x=first_letters.index, y=first_letters.values, palette='viridis')
    plt.title('Топ-10 самых частых первых букв в англицизмах', fontsize=16)
    plt.xlabel('Буква', fontsize=14)
    plt.ylabel('Количество слов', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Добавляем значения над столбцами
    for i, v in enumerate(first_letters.values):
        ax.text(i, v + 1, str(v), ha='center', fontsize=10)

    save_figure("first_letters_frequency")
    plt.close()

    # 7. Word Cloud всех англицизмов
    logger.info("Создание облака слов англицизмов")
    plt.figure(figsize=figure_sizes["word_cloud"])
    try:
        wordcloud = WordCloud(width=1200, height=800,
                              background_color='white',
                              max_words=limits["max_words_wordcloud"],
                              colormap=colors["pie_cmap"],
                              contour_width=1,
                              contour_color='steelblue').generate(' '.join(df['word']))

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Облако слов англицизмов', fontsize=16, pad=20)
        save_figure("anglicisms_wordcloud", tight_layout=False)
    except Exception as e:
        logger.error(f"Ошибка при создании облака слов: {e}")
    finally:
        plt.close()

    # 8. Распределение длин слов по языкам (boxplot)
    if 'origin_language' in df.columns:
        logger.info("Создание boxplot распределения длин слов по языкам")
        plt.figure(figsize=figure_sizes["boxplot"])

        # Выбираем только языки с достаточным количеством слов
        min_words = limits["min_words_per_language"]
        langs_with_enough_words = df['origin_language'].value_counts()[
            df['origin_language'].value_counts() >= min_words].index
        filtered_df = df[df['origin_language'].isin(langs_with_enough_words)]

        if not filtered_df.empty:
            sns.boxplot(x='origin_language', y='word_length', data=filtered_df, palette='viridis')
            plt.title('Распределение длин слов по языкам происхождения', fontsize=16)
            plt.xlabel('Язык происхождения', fontsize=14)
            plt.ylabel('Длина слова (символов)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            save_figure("word_length_boxplot_by_language")
        else:
            logger.warning(f"Нет языков с более чем {min_words} словами для boxplot")

        plt.close()

    # 9. Тепловая карта корреляций (если есть числовые данные)
    num_columns = df.select_dtypes(include=[np.number]).columns
    if len(num_columns) > 1:
        logger.info("Создание тепловой карты корреляций")
        plt.figure(figsize=figure_sizes["heatmap"])
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_matrix, annot=True, cmap=colors["heatmap_cmap"], linewidths=0.5)
        plt.title('Корреляции между числовыми характеристиками', fontsize=16)
        save_figure("correlation_heatmap")
        plt.close()

    # 10. Сводный отчет по доле англицизмов из разных языков
    if 'origin_language' in df.columns:
        logger.info("Создание сводного отчета по доле англицизмов из разных языков")
        plt.figure(figsize=figure_sizes["table"])
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
        table.scale(table_settings["scale_x"], table_settings["scale_y"])

        # Стилизуем таблицу
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Заголовки
                cell.set_text_props(fontproperties=plt.matplotlib.font_manager.FontProperties(weight='bold'))
                cell.set_facecolor(table_settings["header_bgcolor"])
                cell.set_text_props(color=table_settings["header_color"])
            else:
                cell.set_facecolor(
                    table_settings["even_row_bgcolor"] if i % 2 == 0 else table_settings["odd_row_bgcolor"])

        plt.title('Процентное распределение англицизмов по языкам', fontsize=16, pad=20)
        save_figure("language_percentage_table", tight_layout=False)
        plt.close()

    logger.info(f"Визуализации сохранены в директорию: {output_dir}")


if __name__ == "__main__":
    visualize_with_config()
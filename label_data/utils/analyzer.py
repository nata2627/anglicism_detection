import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import logging

logger = logging.getLogger("AngliсismLabeler.Analyzer")


def analyze_anglicisms(marked_texts, anglicism_stats, output_dir="analysis_results", config=None):
    """
    Анализ размеченных англицизмов в текстах.

    Args:
        marked_texts (list): Размеченные тексты
        anglicism_stats (dict): Статистика по англицизмам
        output_dir (str): Директория для сохранения результатов
        config: Конфигурационный объект с настройками визуализации

    Returns:
        tuple: Кортеж из DataFrame (топ англицизмов, статистика по текстам, частотности)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Создана директория для результатов анализа: {output_dir}")

    # Настройки визуализации по умолчанию
    dpi = 300
    style = "whitegrid"
    save_formats = ["png"]
    figure_sizes = {
        "bar_plot": (12, 8),
        "histogram": (12, 8),
        "distribution": (12, 8)
    }

    # Загрузка настроек из конфигурации, если она передана
    if config is not None and hasattr(config, 'labeler') and hasattr(config.labeler, 'visualization'):
        viz_config = config.labeler.visualization
        if hasattr(viz_config, 'dpi'):
            dpi = viz_config.dpi
        if hasattr(viz_config, 'style'):
            style = viz_config.style
        if hasattr(viz_config, 'save_formats'):
            save_formats = viz_config.save_formats
        if hasattr(viz_config, 'figure_sizes'):
            for key, value in viz_config.figure_sizes.items():
                if key in figure_sizes:
                    figure_sizes[key] = tuple(value)

    # Настройка стиля графиков
    sns.set(style=style)

    # Функция для сохранения графиков в разных форматах
    def save_figure(name, tight_layout=True):
        if tight_layout:
            plt.tight_layout()

        for fmt in save_formats:
            file_path = f"{output_dir}/{name}.{fmt}"
            plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
            logger.debug(f"Сохранен график: {file_path}")

    # 1. Топ-20 наиболее частых англицизмов
    logger.info("Создание графика топ-20 наиболее частых англицизмов")
    plt.figure(figsize=figure_sizes["bar_plot"])
    top_anglicisms = sorted(anglicism_stats['anglicism_frequency'].items(), key=lambda x: x[1], reverse=True)[:20]
    df_top = pd.DataFrame(top_anglicisms, columns=['Англицизм', 'Частота'])

    sns.barplot(x='Частота', y='Англицизм', data=df_top)
    plt.title('Топ-20 наиболее частых англицизмов в корпусе')
    plt.tight_layout()
    save_figure("top20_anglicisms")
    plt.close()

    # 2. Распределение количества англицизмов в текстах
    logger.info("Создание графика распределения количества англицизмов в текстах")
    plt.figure(figsize=figure_sizes["histogram"])
    sns.histplot(anglicism_stats['texts_anglicism_count'], kde=True, bins=20)
    plt.title('Распределение количества англицизмов в текстах')
    plt.xlabel('Количество англицизмов в тексте')
    plt.ylabel('Частота')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure("anglicisms_count_distribution")
    plt.close()

    # 3. Распределение процента англицизмов в текстах
    logger.info("Создание графика распределения процента англицизмов в текстах")
    plt.figure(figsize=figure_sizes["distribution"])
    sns.histplot(anglicism_stats['texts_anglicism_percent'], kde=True, bins=20)
    plt.title('Распределение процента англицизмов в текстах')
    plt.xlabel('Процент англицизмов в тексте')
    plt.ylabel('Частота')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure("anglicisms_percent_distribution")
    plt.close()

    # 4. Сохранение статистики в CSV
    logger.info("Сохранение статистики по текстам в CSV")
    df_stats = pd.DataFrame({
        'Текст': [i+1 for i in range(len(anglicism_stats['texts_anglicism_count']))],
        'Кол-во англицизмов': anglicism_stats['texts_anglicism_count'],
        'Процент англицизмов': anglicism_stats['texts_anglicism_percent']
    })
    df_stats.to_csv(f"{output_dir}/anglicisms_stats.csv", index=False)

    # 5. Сохранение частотности англицизмов
    logger.info("Сохранение частотности англицизмов в CSV")
    df_freq = pd.DataFrame(list(anglicism_stats['anglicism_frequency'].items()),
                           columns=['Англицизм', 'Частота'])
    df_freq = df_freq.sort_values('Частота', ascending=False)
    df_freq.to_csv(f"{output_dir}/anglicisms_frequency.csv", index=False)

    # 6. Сохранение общей статистики в JSON
    logger.info("Сохранение общей статистики в JSON")
    general_stats = {
        'total_tokens': anglicism_stats['total_tokens'],
        'total_anglicisms': anglicism_stats['total_anglicisms'],
        'anglicism_percentage': round(anglicism_stats['total_anglicisms'] / anglicism_stats['total_tokens'] * 100, 2) if anglicism_stats['total_tokens'] > 0 else 0,
        'texts_with_anglicisms': anglicism_stats['texts_with_anglicisms'],
        'total_texts': len(anglicism_stats['texts_anglicism_count']),
        'texts_with_anglicisms_percentage': round(anglicism_stats['texts_with_anglicisms'] / len(anglicism_stats['texts_anglicism_count']) * 100, 2) if len(anglicism_stats['texts_anglicism_count']) > 0 else 0
    }
    with open(f"{output_dir}/general_stats.json", 'w', encoding='utf-8') as f:
        json.dump(general_stats, f, ensure_ascii=False, indent=4)

    logger.info(f"Результаты анализа сохранены в директории: {output_dir}")

    return df_top, df_stats, df_freq
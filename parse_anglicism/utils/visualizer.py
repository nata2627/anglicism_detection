import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def visualize_anglicisms(df, output_dir="visualization"):
    # Создаем директорию для сохранения визуализаций, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Настройки визуализации
    dpi = 300
    font_family = 'DejaVu Sans'
    save_formats = ['png']
    style = "whitegrid"

    # Размеры графиков
    figure_sizes = {
        "default": (12, 8),
        "pie_chart": (12, 8),
        "histogram": (12, 8),
        "bar_plot": (14, 10),
        "word_cloud": (14, 10)
    }

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

    # Проверяем наличие необходимых колонок в DataFrame
    if 'word' not in df.columns:
        print("В DataFrame отсутствует колонка 'word'")
        return

    # Убедимся, что у нас есть колонка word_length
    if 'word_length' not in df.columns:
        df['word_length'] = df['word'].apply(len)

    # 1. Круговая диаграмма распределения по языкам происхождения
    if 'origin_language' in df.columns:
        plt.figure(figsize=figure_sizes["pie_chart"])
        lang_counts = df['origin_language'].value_counts()

        # Выделяем топ-8 языков, остальные группируем как "Другие"
        top_n = 8
        if len(lang_counts) > top_n:
            top_langs = lang_counts.head(top_n)
            others = pd.Series({'Другие': lang_counts[top_n:].sum()})
            lang_counts = pd.concat([top_langs, others])

        plt.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%',
                startangle=90, shadow=True, textprops={'fontsize': 12},
                colors=plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(lang_counts))))
        plt.title('Распределение англицизмов по языкам происхождения', fontsize=16)
        plt.axis('equal')
        save_figure("languages_pie_chart")
        plt.close()

    # 2. Гистограмма длин слов
    plt.figure(figsize=figure_sizes["histogram"])
    sns.histplot(df['word_length'], kde=True, bins=20)
    plt.title('Распределение длин англицизмов', fontsize=16)
    plt.xlabel('Количество символов', fontsize=14)
    plt.ylabel('Количество слов', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure("word_length_histogram")
    plt.close()

    # 3. Средняя длина слов по языкам
    if 'origin_language' in df.columns:
        plt.figure(figsize=figure_sizes["bar_plot"])
        avg_length_by_lang = df.groupby('origin_language')['word_length'].mean().sort_values(ascending=False)

        # Выделяем топ-15 языков для лучшей читаемости
        top_n = 15
        if len(avg_length_by_lang) > top_n:
            avg_length_by_lang = avg_length_by_lang.head(top_n)

        ax = sns.barplot(x=avg_length_by_lang.index, y=avg_length_by_lang.values)
        plt.title('Средняя длина англицизмов по языкам происхождения', fontsize=16)
        plt.xlabel('Язык происхождения', fontsize=14)
        plt.ylabel('Средняя длина (символов)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        save_figure("avg_length_by_language")
        plt.close()

    # 4. Word Cloud всех англицизмов
    plt.figure(figsize=figure_sizes["word_cloud"])
    try:
        wordcloud = WordCloud(width=1200, height=800,
                             background_color='white',
                             max_words=200,
                             contour_width=1,
                             contour_color='steelblue').generate(' '.join(df['word']))

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Облако слов англицизмов', fontsize=16, pad=20)
        save_figure("anglicisms_wordcloud", tight_layout=False)
    except Exception as e:
        print(f"Ошибка при создании облака слов: {e}")
    finally:
        plt.close()

    print(f"Визуализации сохранены в директорию: {output_dir}")
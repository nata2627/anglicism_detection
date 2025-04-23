import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx
from matplotlib import cm
from matplotlib.colors import Normalize


def visualize_anglicisms(df, output_dir="visualization"):
    """
    Создает визуализации для анализа англицизмов.

    Args:
        df: DataFrame с данными англицизмов
        output_dir: Директория для сохранения визуализаций
    """
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
        "histogram": (12, 8),
        "heatmap": (14, 10),
        "network": (16, 16),
        "bar_plot": (14, 10)
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

    # 1. Гистограмма длин слов
    try:
        plt.figure(figsize=figure_sizes["histogram"])
        sns.histplot(df['word_length'], kde=True, bins=20)
        plt.title('Распределение длин англицизмов', fontsize=16)
        plt.xlabel('Количество символов', fontsize=14)
        plt.ylabel('Количество слов', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        save_figure("word_length_histogram")
        plt.close()
        print("Создана гистограмма длин слов")
    except Exception as e:
        print(f"Ошибка при создании гистограммы длин слов: {e}")

    # 2. Тепловая карта букв на определенных позициях в слове
    try:
        plt.figure(figsize=figure_sizes["heatmap"])
        # Анализируем первые две и последние две буквы
        prefix_df = pd.DataFrame()
        # Первая буква
        prefix_df['first'] = df['word'].apply(lambda x: x[0].lower() if len(x) > 0 else '')
        # Вторая буква
        prefix_df['second'] = df['word'].apply(lambda x: x[1].lower() if len(x) > 1 else '')
        # Предпоследняя буква
        prefix_df['second_last'] = df['word'].apply(lambda x: x[-2].lower() if len(x) > 1 else '')
        # Последняя буква
        prefix_df['last'] = df['word'].apply(lambda x: x[-1].lower() if len(x) > 0 else '')

        # Считаем частоту первой и последней буквы
        first_last_counts = pd.crosstab(prefix_df['first'], prefix_df['last'])
        # Отбираем наиболее частые буквы (топ 10)
        top_first = prefix_df['first'].value_counts().nlargest(10).index
        top_last = prefix_df['last'].value_counts().nlargest(10).index

        # Убедимся, что у нас есть данные для всех комбинаций
        filtered_first = [f for f in top_first if f in first_last_counts.index]
        filtered_last = [l for l in top_last if l in first_last_counts.columns]

        if filtered_first and filtered_last:
            heatmap_data = first_last_counts.loc[filtered_first, filtered_last]

            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='viridis')
            plt.title('Частота сочетаний первой и последней буквы в англицизмах', fontsize=16)
            plt.xlabel('Последняя буква', fontsize=14)
            plt.ylabel('Первая буква', fontsize=14)
            save_figure("first_last_letter_heatmap")
            plt.close()
            print("Создана тепловая карта сочетаний букв")
        else:
            print("Недостаточно данных для создания тепловой карты сочетаний букв")
    except Exception as e:
        print(f"Ошибка при создании тепловой карты букв: {e}")

    # 3. Визуализация наиболее частых биграмм
    try:
        # Получаем биграммы
        all_bigrams = []
        for word in df['word']:
            if len(word) >= 2:
                word_bigrams = [word[i:i + 2] for i in range(len(word) - 1)]
                all_bigrams.extend(word_bigrams)

        bigram_counts = Counter(all_bigrams)
        top_n_bigrams = 20

        if bigram_counts:
            top_bigrams = dict(bigram_counts.most_common(top_n_bigrams))

            plt.figure(figsize=figure_sizes["bar_plot"])
            sns.barplot(x=list(top_bigrams.keys()), y=list(top_bigrams.values()))
            plt.title(f'Топ-{top_n_bigrams} биграмм в англицизмах', fontsize=16)
            plt.xlabel('Биграмма', fontsize=14)
            plt.ylabel('Частота', fontsize=14)
            plt.xticks(rotation=45)
            save_figure("top_bigrams")
            plt.close()
            print("Создана визуализация частых биграмм")
        else:
            print("Недостаточно данных для визуализации биграмм")
    except Exception as e:
        print(f"Ошибка при визуализации биграмм: {e}")

    # 4. Визуализация распределения стеммированных слов
    try:
        # Выполняем стемминг, если не выполнен
        if 'stem' not in df.columns:
            try:
                from nltk.stem.snowball import SnowballStemmer
                stemmer = SnowballStemmer("russian")
                df['stem'] = df['word'].apply(lambda x: stemmer.stem(x))
                print("Выполнен стемминг слов")
            except Exception as e:
                print(f"Не удалось выполнить стемминг: {e}")
                df['stem'] = df['word']  # Используем исходные слова

        stem_counts = df['stem'].value_counts()

        if not stem_counts.empty:
            top_stems = stem_counts.head(20)

            plt.figure(figsize=figure_sizes["bar_plot"])
            sns.barplot(x=top_stems.index, y=top_stems.values)
            plt.title('Наиболее частые основы слов (стеммы)', fontsize=16)
            plt.xlabel('Основа слова', fontsize=14)
            plt.ylabel('Количество слов', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            save_figure("top_stems")
            plt.close()
            print("Создана визуализация распределения стемм")
        else:
            print("Нет данных для визуализации стемм")
    except Exception as e:
        print(f"Ошибка при визуализации стемм: {e}")

    # 5. Сетевая визуализация связей между первыми и последними буквами
    try:
        plt.figure(figsize=figure_sizes["network"])
        G = nx.DiGraph()

        # Создаем граф, где узлы - буквы
        # Связи - переходы от первой буквы к последней
        edge_weights = {}
        for _, row in prefix_df.iterrows():
            first = row['first']
            last = row['last']
            if first and last:  # Убедимся, что буквы есть
                if (first, last) in edge_weights:
                    edge_weights[(first, last)] += 1
                else:
                    edge_weights[(first, last)] = 1

        # Проверяем, есть ли у нас связи
        if edge_weights:
            # Добавляем наиболее значимые связи (топ 50 или все, если меньше 50)
            num_edges = min(50, len(edge_weights))
            if num_edges > 0:
                # Сортируем веса ребер
                sorted_weights = sorted(edge_weights.values(), reverse=True)
                threshold = sorted_weights[min(num_edges - 1, len(sorted_weights) - 1)]

                # Добавляем ребра с весами
                for (first, last), weight in edge_weights.items():
                    if weight >= threshold:
                        G.add_edge(first, last, weight=weight)

                # Проверяем, есть ли узлы в графе
                if len(G.nodes()) > 0:
                    # Расположение узлов
                    try:
                        pos = nx.spring_layout(G, seed=42)
                    except Exception as layout_error:
                        print(f"Ошибка при расположении узлов: {layout_error}")
                        pos = {node: (np.random.rand(), np.random.rand()) for node in G.nodes()}

                    # Настройки размеров узлов и рёбер
                    node_sizes = [100 + G.degree(node) * 20 for node in G.nodes()]
                    edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]

                    # Рисуем граф
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.7)
                    nx.draw_networkx_labels(G, pos, font_size=12)

                    # Нормализуем веса для цветовой карты
                    if edge_weights:
                        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
                        cmap = cm.get_cmap('viridis')

                        # Рисуем рёбра с разными цветами в зависимости от веса
                        for i, (u, v) in enumerate(G.edges()):
                            nx.draw_networkx_edges(
                                G, pos, edgelist=[(u, v)], width=edge_weights[i] * 0.2,
                                alpha=0.7, edge_color=[cmap(norm(edge_weights[i]))]
                            )

                    plt.title('Сеть связей между первыми и последними буквами', fontsize=16)
                    plt.axis('off')
                    save_figure("letter_network")
                    print("Создана сетевая визуализация связей между буквами")
                else:
                    print("Недостаточно узлов для создания сетевой визуализации")
            else:
                print("Недостаточно связей для создания сетевой визуализации")
        else:
            print("Не найдены связи для сетевой визуализации")

        plt.close()
    except Exception as e:
        print(f"Не удалось создать сетевую визуализацию: {e}")

    # 6. Распределение окончаний слов (суффиксов)
    try:
        plt.figure(figsize=figure_sizes["bar_plot"])
        suffix_length = 2  # Длина анализируемого суффикса
        suffixes = df['word'].apply(lambda x: x[-suffix_length:] if len(x) >= suffix_length else x)
        suffix_counts = suffixes.value_counts().nlargest(15)

        if not suffix_counts.empty:
            sns.barplot(x=suffix_counts.index, y=suffix_counts.values)
            plt.title(f'Наиболее распространенные {suffix_length}-буквенные окончания', fontsize=16)
            plt.xlabel('Окончание', fontsize=14)
            plt.ylabel('Количество слов', fontsize=14)
            plt.xticks(rotation=0)
            save_figure(f"top_suffixes_{suffix_length}")
            print(f"Создана визуализация распределения {suffix_length}-буквенных окончаний")
        else:
            print(f"Недостаточно данных для визуализации {suffix_length}-буквенных окончаний")

        plt.close()
    except Exception as e:
        print(f"Ошибка при визуализации суффиксов: {e}")

    # 7. Распределение букв по первой позиции в слове
    try:
        plt.figure(figsize=figure_sizes["bar_plot"])
        first_letter_counts = prefix_df['first'].value_counts().nlargest(15)

        if not first_letter_counts.empty:
            sns.barplot(x=first_letter_counts.index, y=first_letter_counts.values)
            plt.title('Наиболее распространенные первые буквы в англицизмах', fontsize=16)
            plt.xlabel('Первая буква', fontsize=14)
            plt.ylabel('Количество слов', fontsize=14)
            save_figure("first_letter_counts")
            print("Создана визуализация распределения первых букв")
        else:
            print("Недостаточно данных для визуализации первых букв")

        plt.close()
    except Exception as e:
        print(f"Ошибка при визуализации первых букв: {e}")

    # 8. Визуализация наиболее частых лемм
    try:
        if 'lemma' in df.columns:
            lemma_counts = df['lemma'].value_counts()

            if not lemma_counts.empty:
                top_lemmas = lemma_counts.head(20)

                plt.figure(figsize=figure_sizes["bar_plot"])
                sns.barplot(x=top_lemmas.index, y=top_lemmas.values)
                plt.title('Наиболее частые леммы', fontsize=16)
                plt.xlabel('Лемма', fontsize=14)
                plt.ylabel('Количество слов', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                save_figure("top_lemms")
                plt.close()
                print("Создана визуализация наиболее частых лемм")
            else:
                print("Недостаточно данных для визуализации наиболее частых лемм")
    except Exception as e:
        print(f"Ошибка при визуализации наиболее частых лемм: {e}")

    # 9. Сравнение стеммов и лемм
    try:
        if 'stem' in df.columns and 'lemma' in df.columns:
            plt.figure(figsize=figure_sizes["bar_plot"])

            # Создаем DataFrame для сравнения количества уникальных стеммов и лемм
            comparison_data = {
                'Тип': ['Оригинальные слова', 'Стеммы', 'Леммы'],
                'Количество': [
                    len(df['word'].unique()),
                    len(df['stem'].unique()),
                    len(df['lemma'].unique())
                ]
            }
            comparison_df = pd.DataFrame(comparison_data)

            sns.barplot(x='Тип', y='Количество', data=comparison_df)
            plt.title('Сравнение количества уникальных слов, стеммов и лемм', fontsize=16)
            plt.ylabel('Количество уникальных значений', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            save_figure("stem_lemma_comparison")
            plt.close()
            print("Создана визуализация сравнения количества стемм и лемм")

            # Диаграмма количества слов с одинаковой леммой
            try:
                plt.figure(figsize=figure_sizes["histogram"])
                lemma_group_sizes = df.groupby('lemma').size()

                if not lemma_group_sizes.empty:
                    sns.histplot(lemma_group_sizes, kde=True, bins=20)
                    plt.title('Распределение количества слов для каждой леммы', fontsize=16)
                    plt.xlabel('Количество слов с одинаковой леммой', fontsize=14)
                    plt.ylabel('Частота', fontsize=14)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    save_figure("lemma_group_sizes")
                    print("Создана визуализация распределения групп лемм")
                else:
                    print("Недостаточно данных для визуализации групп лемм")

                plt.close()
            except Exception as e:
                print(f"Ошибка при создании гистограммы групп лемм: {e}")
    except Exception as e:
        print(f"Не удалось создать визуализации для стеммов и лемм: {e}")

    print(f"Визуализации сохранены в директорию: {output_dir}")
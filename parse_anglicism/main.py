import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from wordcloud import WordCloud
import matplotlib.ticker as mticker
import os

# Настраиваем поддержку русского языка в matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

def parse_anglicisms(file_path):
    """
    Парсит англицизмы из файла формата Викисловаря.

    Args:
        file_path (str): Путь к файлу с англицизмами

    Returns:
        dict: Словарь англицизмов с их происхождением
    """
    # Чтение файла
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Регулярное выражение для поиска разделов с языками
    language_sections = re.findall(r'== Из (.*?) ==', content)

    # Регулярное выражение для извлечения англицизмов
    # Ищем слова в двойных квадратных скобках, за которыми следует описание
    anglicism_pattern = r'\[\[(.*?)\]\](.*?(?=\[\[|$))'

    # Словарь для хранения англицизмов по языкам происхождения
    anglicisms_by_language = defaultdict(list)

    # Текущий раздел языка
    current_language = None

    # Обработка строк файла
    for line in content.split('\n'):
        # Проверка, является ли строка заголовком раздела
        language_match = re.search(r'== Из (.*?) ==', line)
        if language_match:
            current_language = language_match.group(1)
            continue

        # Если текущий язык определен, ищем англицизмы
        if current_language:
            matches = re.findall(anglicism_pattern, line)
            for match in matches:
                word = match[0]
                description = match[1].strip()

                # Проверяем, содержит ли описание упоминание английского языка
                is_through_english = bool(re.search(r'через англ', description))

                # Если слово содержит |, берем только часть до |
                if '|' in word:
                    word = word.split('|')[0]

                anglicisms_by_language[current_language].append({
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

    return {
        'by_language': anglicisms_by_language,
        'all_anglicisms': all_anglicisms
    }

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

    print(f"Всего найдено англицизмов: {len(all_anglicisms)}")
    print("\nРаспределение по языкам происхождения:")

    for language, words in by_language.items():
        print(f"  {language}: {len(words)} слов")

    # Создаем DataFrame для более удобного анализа
    df = pd.DataFrame(all_anglicisms)

    # Количество англицизмов, пришедших через английский
    through_english_count = df['through_english'].sum()
    print(f"\nАнглицизмов, пришедших через английский: {through_english_count} ({through_english_count/len(df)*100:.2f}%)")

    # Проверка на дубликаты слов
    duplicates = df[df.duplicated('word', keep=False)]
    if not duplicates.empty:
        print(f"\nНайдены дубликаты слов ({len(duplicates)}):")
        for word in duplicates['word'].unique():
            print(f"  {word}")

    # Анализ длины слов
    df['word_length'] = df['word'].apply(len)
    avg_length = df['word_length'].mean()
    print(f"\nСредняя длина англицизма: {avg_length:.2f} символов")

    # Топ-10 самых длинных слов
    longest_words = df.nlargest(10, 'word_length')
    print("\nТоп-10 самых длинных англицизмов:")
    for _, row in longest_words.iterrows():
        print(f"  {row['word']} ({row['word_length']} символов)")

    # Анализ по первым буквам
    df['first_letter'] = df['word'].str[0]
    first_letter_counts = df['first_letter'].value_counts().head(10)
    print("\nТоп-10 самых частых первых букв в англицизмах:")
    for letter, count in first_letter_counts.items():
        print(f"  {letter}: {count} слов")

    # Анализ частотности букв
    all_letters = ''.join(df['word'].str.lower())
    letter_counts = Counter(all_letters)
    print("\nТоп-10 самых частых букв в англицизмах:")
    for letter, count in letter_counts.most_common(10):
        print(f"  {letter}: {count} вхождений")

    return df

def clean_anglicisms(df):
    """
    Очищает и нормализует список англицизмов.

    Args:
        df (DataFrame): DataFrame с англицизмами

    Returns:
        DataFrame: Очищенный DataFrame
    """
    # Копия DataFrame для безопасного изменения
    clean_df = df.copy()

    # Приведение всех слов к нижнему регистру
    clean_df['word'] = clean_df['word'].str.lower()

    # Удаление лишних символов (например, двоеточий, запятых)
    clean_df['word'] = clean_df['word'].str.replace(r'[^\w\s]', '', regex=True)

    # Удаление дубликатов
    clean_df = clean_df.drop_duplicates('word')

    print(f"\nПосле очистки осталось англицизмов: {len(clean_df)}")

    return clean_df

def save_anglicisms(df, output_file, excel_output=None):
    """
    Сохраняет обработанные англицизмы в файл.

    Args:
        df (DataFrame): DataFrame с англицизмами
        output_file (str): Путь к файлу для сохранения
        excel_output (str, optional): Путь для сохранения полных данных в Excel
    """
    # Сохраняем только колонку с самими словами
    df[['word']].to_csv(output_file, index=False, header=False)
    print(f"\nАнглицизмы сохранены в файл: {output_file}")

    # Если указан путь для Excel, сохраняем все данные в Excel
    if excel_output:
        # Создаем копию DataFrame с наиболее важными колонками
        export_df = df[['word', 'origin_language', 'word_length', 'through_english']]

        # Переименовываем колонки для удобства
        export_df.columns = ['Англицизм', 'Язык происхождения', 'Длина слова', 'Через английский']

        # Сохраняем в Excel
        export_df.to_excel(excel_output, index=False, sheet_name='Англицизмы')

        print(f"Расширенные данные сохранены в Excel: {excel_output}")

def visualize_anglicisms(df, output_dir="visualization"):
    """
    Создает визуализации для анализа англицизмов.

    Args:
        df (DataFrame): DataFrame с англицизмами
        output_dir (str): Директория для сохранения графиков
    """
    # Создаем директорию для сохранения визуализаций, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Настройка стиля графиков
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # 1. Круговая диаграмма распределения по языкам происхождения
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
    plt.savefig(f"{output_dir}/languages_pie_chart.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Гистограмма длин слов
    plt.figure(figsize=(12, 8))
    sns.histplot(df['word_length'], kde=True, bins=20)
    plt.title('Распределение длин англицизмов', fontsize=16)
    plt.xlabel('Количество символов', fontsize=14)
    plt.ylabel('Количество слов', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/word_length_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Средняя длина слов по языкам
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
    plt.savefig(f"{output_dir}/avg_length_by_language.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Количество слов по языкам (горизонтальный барплот)
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
    plt.savefig(f"{output_dir}/words_count_by_language.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Визуализация через/не через английский
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
        plt.text(i, v + 5, f'{v} ({v/total*100:.1f}%)', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/through_english.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Частотность первых букв (топ-10)
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
    plt.savefig(f"{output_dir}/first_letters_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Word Cloud всех англицизмов
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
    plt.savefig(f"{output_dir}/anglicisms_wordcloud.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Распределение длин слов по языкам (boxplot)
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
    plt.savefig(f"{output_dir}/word_length_boxplot_by_language.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Тепловая карта корреляций (если есть числовые данные)
    if df.select_dtypes(include=[np.number]).shape[1] > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Корреляции между числовыми характеристиками', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 10. Сводный отчет по доле англицизмов из разных языков
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
    plt.savefig(f"{output_dir}/language_percentage_table.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nВизуализации сохранены в директорию: {output_dir}")

def advanced_analysis(df):
    """
    Проводит расширенный анализ англицизмов.

    Args:
        df (DataFrame): DataFrame с англицизмами

    Returns:
        dict: Словарь с результатами анализа
    """
    # Создаем словарь для хранения результатов анализа
    analysis_results = {}

    # 1. Анализ длины слов по языкам
    length_by_lang = df.groupby('origin_language')['word_length'].agg(['mean', 'median', 'min', 'max', 'count'])
    length_by_lang = length_by_lang.sort_values('count', ascending=False)
    analysis_results['length_by_language'] = length_by_lang

    print("\nСредняя длина слов по языкам происхождения (топ-10 по количеству):")
    print(length_by_lang.head(10))

    # 2. Анализ англицизмов, пришедших через английский, по языкам
    through_eng_by_lang = df.groupby('origin_language')['through_english'].mean() * 100
    through_eng_by_lang = through_eng_by_lang.sort_values(ascending=False)
    analysis_results['through_english_by_language'] = through_eng_by_lang

    print("\nДоля англицизмов, пришедших через английский, по языкам происхождения (%):")
    for lang, percentage in through_eng_by_lang.head(10).items():
        print(f"  {lang}: {percentage:.2f}%")

    # 3. Анализ частотности букв
    all_letters = ''.join(df['word'].str.lower())
    letter_freq = pd.Series(Counter(all_letters)).sort_values(ascending=False)
    analysis_results['letter_frequency'] = letter_freq

    # 4. Распределение длины описаний
    df['description_length'] = df['description'].str.len()
    desc_length_stats = df['description_length'].describe()
    analysis_results['description_length_stats'] = desc_length_stats

    print("\nСтатистика длины описаний англицизмов:")
    print(desc_length_stats)

    # 5. Анализ уникальности слов по языкам
    unique_words_by_lang = df.groupby('origin_language')['word'].nunique()
    total_words_by_lang = df.groupby('origin_language')['word'].count()
    uniqueness_ratio = unique_words_by_lang / total_words_by_lang * 100
    uniqueness_ratio = uniqueness_ratio.sort_values(ascending=False)
    analysis_results['uniqueness_ratio_by_language'] = uniqueness_ratio

    print("\nКоэффициент уникальности слов по языкам (%):")
    for lang, ratio in uniqueness_ratio.head(10).items():
        print(f"  {lang}: {ratio:.2f}%")

    # 6. Создаем новую колонку для категоризации длины слов
    bins = [0, 4, 8, 12, 100]
    labels = ['Короткие (1-4)', 'Средние (5-8)', 'Длинные (9-12)', 'Очень длинные (>12)']
    df['length_category'] = pd.cut(df['word_length'], bins=bins, labels=labels, right=False)

    length_category_counts = df['length_category'].value_counts().sort_index()
    analysis_results['length_category_counts'] = length_category_counts

    print("\nРаспределение англицизмов по категориям длины:")
    for category, count in length_category_counts.items():
        print(f"  {category}: {count} слов ({count/len(df)*100:.2f}%)")

    return analysis_results

def compare_languages(df, top_n=5):
    """
    Проводит сравнительный анализ англицизмов по языкам происхождения.

    Args:
        df (DataFrame): DataFrame с англицизмами
        top_n (int): Количество языков для сравнения

    Returns:
        DataFrame: DataFrame со сравнительными данными
    """
    # Получаем топ-N языков по количеству англицизмов
    top_languages = df['origin_language'].value_counts().head(top_n).index.tolist()

    # Фильтруем DataFrame по этим языкам
    filtered_df = df[df['origin_language'].isin(top_languages)]

    # Создаем сводную таблицу для сравнения
    comparison_data = []

    for lang in top_languages:
        lang_df = filtered_df[filtered_df['origin_language'] == lang]

        comparison_data.append({
            'Язык': lang,
            'Количество слов': len(lang_df),
            'Средняя длина': lang_df['word_length'].mean(),
            'Медиана длины': lang_df['word_length'].median(),
            'Минимальная длина': lang_df['word_length'].min(),
            'Максимальная длина': lang_df['word_length'].max(),
            'Через английский (%)': lang_df['through_english'].mean() * 100,
            'Самое короткое слово': lang_df.loc[lang_df['word_length'].idxmin(), 'word'] if not lang_df.empty else 'N/A',
            'Самое длинное слово': lang_df.loc[lang_df['word_length'].idxmax(), 'word'] if not lang_df.empty else 'N/A',
            'Самый популярный префикс': lang_df['word'].str[:2].value_counts().index[0] if not lang_df.empty else 'N/A'
        })

    comparison_df = pd.DataFrame(comparison_data)

    print("\n=== СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЯЗЫКОВ (ТОП-5) ===")
    print(comparison_df)

    # Визуализация сравнительного анализа
    plt.figure(figsize=(14, 8))

    # Создаем столбчатую диаграмму для сравнения средней длины
    x = comparison_df['Язык']
    y1 = comparison_df['Средняя длина']
    y2 = comparison_df['Через английский (%)']

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Первая ось для средней длины
    bars = ax1.bar(x, y1, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Язык происхождения', fontsize=14)
    ax1.set_ylabel('Средняя длина (символов)', fontsize=14, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', color='royalblue', fontweight='bold')

    # Вторая ось для процента через английский
    ax2 = ax1.twinx()
    line = ax2.plot(x, y2, 'o-', color='crimson', linewidth=2, markersize=8)
    ax2.set_ylabel('Через английский (%)', fontsize=14, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')

    # Добавляем значения к линии
    for i, v in enumerate(y2):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', color='crimson', fontweight='bold')

    plt.title('Сравнение языков по средней длине и доле заимствований через английский', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"language_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    return comparison_df

def analyze_letter_patterns(df):
    """
    Анализирует паттерны букв в англицизмах.

    Args:
        df (DataFrame): DataFrame с англицизмами

    Returns:
        dict: Словарь с результатами анализа
    """
    pattern_results = {}

    # 1. Анализ суффиксов (последние 2 буквы)
    df['suffix'] = df['word'].apply(lambda x: x[-2:] if len(x) >= 2 else x)
    suffix_counts = df['suffix'].value_counts().head(10)
    pattern_results['top_suffixes'] = suffix_counts

    print("\nТоп-10 самых частых суффиксов (последние 2 буквы):")
    for suffix, count in suffix_counts.items():
        print(f"  '{suffix}': {count} слов")

    # 2. Анализ префиксов (первые 2 буквы)
    df['prefix'] = df['word'].apply(lambda x: x[:2] if len(x) >= 2 else x)
    prefix_counts = df['prefix'].value_counts().head(10)
    pattern_results['top_prefixes'] = prefix_counts

    print("\nТоп-10 самых частых префиксов (первые 2 буквы):")
    for prefix, count in prefix_counts.items():
        print(f"  '{prefix}': {count} слов")

    # 3. Биграммы (пары последовательных букв)
    bigrams = []
    for word in df['word']:
        for i in range(len(word)-1):
            bigrams.append(word[i:i+2])

    bigram_counts = pd.Series(Counter(bigrams)).sort_values(ascending=False).head(10)
    pattern_results['top_bigrams'] = bigram_counts

    print("\nТоп-10 самых частых биграмм (пар букв):")
    for bigram, count in bigram_counts.items():
        print(f"  '{bigram}': {count} вхождений")

    return pattern_results

# Пример использования
if __name__ == "__main__":
    # Путь к файлу с англицизмами
    file_path = "angl.txt"

    # Директория для сохранения визуализаций
    visualization_dir = "anglicisms_visualization"

    # Парсинг англицизмов
    print("Парсинг англицизмов из файла...")
    anglicisms_dict = parse_anglicisms(file_path)

    # Анализ полученных данных
    print("\n=== БАЗОВЫЙ АНАЛИЗ ===")
    df = analyze_anglicisms(anglicisms_dict)

    # Очистка и нормализация данных
    print("\n=== ОЧИСТКА ДАННЫХ ===")
    clean_df = clean_anglicisms(df)

    # Расширенный анализ данных
    print("\n=== РАСШИРЕННЫЙ АНАЛИЗ ===")
    analysis_results = advanced_analysis(clean_df)

    # Анализ паттернов букв
    print("\n=== АНАЛИЗ ПАТТЕРНОВ БУКВ ===")
    pattern_results = analyze_letter_patterns(clean_df)

    # Сравнительный анализ языков
    comparison_df = compare_languages(clean_df, top_n=5)

    # Визуализация данных
    print("\n=== СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ ===")
    visualize_anglicisms(clean_df, visualization_dir)

    # Сохранение обработанных англицизмов
    save_anglicisms(clean_df, "clean_anglicisms.txt", excel_output="anglicisms_analysis.xlsx")

    # Вывести несколько примеров англицизмов
    print("\nПримеры англицизмов:")
    for word in clean_df['word'].head(20).tolist():
        print(f"  {word}")
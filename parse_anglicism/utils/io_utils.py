import os
import csv


def setup_directory_structure(paths_config):
    """
    Создает структуру директорий согласно конфигурации.

    Args:
        paths_config: Словарь с путями для создания
    """
    for path_name, path in paths_config.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Создана директория: {path}")


def save_anglicisms(df, output_file, csv_output=None):
    """
    Сохраняет англицизмы в текстовый файл и CSV.

    Args:
        df: DataFrame с англицизмами
        output_file: Путь для сохранения списка англицизмов
        csv_output: Путь для сохранения CSV файла с анализом
    """
    # Проверяем наличие директории для output_file
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Сохраняем только колонку с самими словами
    df[['word']].to_csv(output_file, index=False, header=False)
    print(f"Англицизмы сохранены в файл: {output_file}")

    # Если указан путь для CSV, сохраняем данные в CSV
    if csv_output:
        # Проверяем наличие директории для csv_output
        csv_dir = os.path.dirname(csv_output)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        # Создаем копию DataFrame с наиболее важными колонками
        export_columns = ['word', 'word_length']

        # Проверяем наличие дополнительных колонок
        if 'letter' in df.columns:
            export_columns.append('letter')

        if 'stem' in df.columns:
            export_columns.append('stem')

        if 'first_letter' in df.columns:
            export_columns.append('first_letter')

        if 'last_letter' in df.columns:
            export_columns.append('last_letter')

        export_df = df[export_columns].copy()

        # Словарь переименований колонок для удобства
        column_renames = {
            'word': 'Англицизм',
            'word_length': 'Длина слова',
            'letter': 'Буква',
            'stem': 'Основа слова',
            'first_letter': 'Первая буква',
            'last_letter': 'Последняя буква'
        }

        # Переименовываем колонки, которые присутствуют в DataFrame
        rename_dict = {col: column_renames[col] for col in export_columns if col in column_renames}
        export_df.rename(columns=rename_dict, inplace=True)

        # Сохраняем в CSV
        try:
            export_df.to_csv(csv_output, index=False)
            print(f"Данные сохранены в CSV: {csv_output}")
        except Exception as e:
            print(f"Ошибка при сохранении CSV-файла: {e}")


def save_analysis_results(analysis_results, output_file):
    """
    Сохраняет результаты расширенного анализа в текстовый файл.

    Args:
        analysis_results: Словарь с результатами анализа
        output_file: Путь для сохранения результатов
    """
    try:
        # Создаем директорию, если её нет
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("РЕЗУЛЬТАТЫ АНАЛИЗА АНГЛИЦИЗМОВ\n")
            file.write("=" * 50 + "\n\n")

            # Статистика по длине слов
            if 'length_stats' in analysis_results:
                file.write("Статистика по длине слов:\n")
                file.write("-" * 30 + "\n")
                for stat_name, value in analysis_results['length_stats'].items():
                    file.write(f"{stat_name}: {value:.2f}\n")
                file.write("\n")

            # Категории длины слов
            if 'length_category_counts' in analysis_results:
                file.write("Распределение по категориям длины:\n")
                file.write("-" * 30 + "\n")
                for category, count in analysis_results['length_category_counts'].items():
                    file.write(f"{category}: {count}\n")
                file.write("\n")

            # Частотность первых букв
            if 'first_letter_freq' in analysis_results:
                file.write("Частотность первых букв:\n")
                file.write("-" * 30 + "\n")
                for letter, count in analysis_results['first_letter_freq'].head(10).items():
                    file.write(f"{letter}: {count}\n")
                file.write("\n")

            # Частотность последних букв
            if 'last_letter_freq' in analysis_results:
                file.write("Частотность последних букв:\n")
                file.write("-" * 30 + "\n")
                for letter, count in analysis_results['last_letter_freq'].head(10).items():
                    file.write(f"{letter}: {count}\n")
                file.write("\n")

            # Стеммы
            if 'stem_counts' in analysis_results:
                file.write("Наиболее частые основы слов (стеммы):\n")
                file.write("-" * 30 + "\n")
                for stem, count in analysis_results['stem_counts'].head(20).items():
                    file.write(f"{stem}: {count}\n")
                file.write("\n")

            # Биграммы
            if 'bigrams' in analysis_results:
                file.write("Наиболее частые биграммы:\n")
                file.write("-" * 30 + "\n")
                for bigram, count in sorted(analysis_results['bigrams'].items(), key=lambda x: x[1], reverse=True)[:20]:
                    file.write(f"{bigram}: {count}\n")
                file.write("\n")

            # Суффиксы
            if 'suffixes' in analysis_results:
                for length, suffix_counter in analysis_results['suffixes'].items():
                    file.write(f"Наиболее частые суффиксы длины {length}:\n")
                    file.write("-" * 30 + "\n")
                    for suffix, count in suffix_counter.most_common(10):
                        file.write(f"{suffix}: {count}\n")
                    file.write("\n")

            # Леммы
            if 'lemma_counts' in analysis_results:
                file.write("Наиболее частые леммы:\n")
                file.write("-" * 30 + "\n")
                for lemma, count in analysis_results['lemma_counts'].head(20).items():
                    file.write(f"{lemma}: {count}\n")
                file.write("\n")

            # Сравнение лемм и стеммов
            if 'lemma_stem_comparison' in analysis_results:
                file.write("Интересные случаи: разные стеммы для одной леммы:\n")
                file.write("-" * 30 + "\n")
                comparison = analysis_results['lemma_stem_comparison']
                for lemma in comparison['lemma'].unique():
                    file.write(f"Лемма: {lemma}\n")
                    group = comparison[comparison['lemma'] == lemma]
                    for _, row in group.iterrows():
                        file.write(f"  Слово: {row['word']}, Стемм: {row['stem']}\n")
                    file.write("\n")

        print(f"Результаты анализа сохранены в: {output_file}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов анализа: {e}")


def convert_to_lowercase(input_file, output_file):
    """
    Конвертирует слова в нижний регистр.

    Args:
        input_file: Путь к исходному файлу
        output_file: Путь для сохранения результата
    """
    try:
        # Проверяем, существует ли файл
        if not os.path.exists(input_file):
            print(f"Ошибка: Файл {input_file} не существует")
            return False

        # Читаем файл
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            # Если UTF-8 не работает, пробуем cp1251
            with open(input_file, 'r', encoding='cp1251') as f:
                words = [line.strip() for line in f if line.strip()]

        # Переводим все в нижний регистр
        processed_words = [word.lower() for word in words]

        # Записываем результаты в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in processed_words:
                f.write(word + '\n')

        print(f"Обработка завершена. Результат сохранен в {output_file}")
        return True

    except Exception as e:
        print(f"Произошла ошибка при конвертации в нижний регистр: {str(e)}")
        return False


def export_to_csv_by_letter(df, output_dir):
    """
    Экспортирует англицизмы в CSV файлы, разделенные по первым буквам.

    Args:
        df: DataFrame с англицизмами
        output_dir: Директория для сохранения файлов
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Группировка по первой букве
    if 'first_letter' not in df.columns:
        df['first_letter'] = df['word'].apply(lambda x: x[0].lower() if x else "")

    # Сгруппируем данные по первой букве
    grouped = df.groupby('first_letter')

    # Для каждой буквы создадим отдельный CSV файл
    for letter, group in grouped:
        if not letter:  # Пропускаем пустые значения
            continue

        file_path = os.path.join(output_dir, f"anglicisms_{letter}.csv")
        group.to_csv(file_path, index=False)
        print(f"Сохранен файл для буквы '{letter}': {file_path}")

    print(f"Данные экспортированы по буквам в директорию: {output_dir}")
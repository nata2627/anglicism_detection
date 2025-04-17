import os


def setup_directory_structure(paths_config):
    for path_name, path in paths_config.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Создана директория: {path}")


def save_anglicisms(df, output_file, csv_output=None):
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
        export_df = df[['word', 'origin_language', 'word_length']].copy()

        # Переименовываем колонки для удобства
        export_df.columns = ['Англицизм', 'Язык происхождения', 'Длина слова']

        # Сохраняем в CSV
        try:
            export_df.to_csv(csv_output, index=False)
            print(f"Данные сохранены в CSV: {csv_output}")
        except Exception as e:
            print(f"Ошибка при сохранении CSV-файла: {e}")
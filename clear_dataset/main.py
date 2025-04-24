import os
import glob
import pandas as pd
from pathlib import Path


def main() -> None:
    # Определяем пути
    data_dir = 'assets'
    output_dir = 'assets'
    output_file = 'texts.csv'
    file_pattern = 'articles.csv'
    text_column = 'text'

    # Находим входные файлы
    input_files = glob.glob(os.path.join(data_dir, file_pattern))

    if not input_files:
        print(f"Не найдено файлов по шаблону: {os.path.join(data_dir, file_pattern)}")
        return

    print(f"Найдено {len(input_files)} файлов для обработки")

    # Загружаем и объединяем данные из всех файлов
    all_data = []
    for file_path in input_files:
        try:
            print(f"Загрузка данных из файла: {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8')

            # Проверяем наличие необходимой колонки с текстом
            if text_column not in df.columns:
                print(f"В файле {file_path} отсутствует колонка '{text_column}'")
                continue

            # Оставляем только нужную колонку и переименовываем
            df = df[[text_column]].rename(columns={text_column: 'original_text'})
            all_data.append(df)
            print(f"Загружено {len(df)} строк из файла {file_path}")
        except Exception as e:
            print(f"Ошибка при загрузке файла {file_path}: {str(e)}")

    if not all_data:
        print("Не удалось загрузить данные ни из одного файла")
        return

    # Объединяем все данные
    result_df = pd.concat(all_data, ignore_index=True)
    print(f"Всего загружено {len(result_df)} строк из {len(all_data)} файлов")

    # Сохраняем результаты
    try:
        output_path = os.path.join(output_dir, output_file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        result_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Результаты сохранены в файл: {output_path}")
        print(f"Сохранено {len(result_df)} текстов")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {str(e)}")


if __name__ == "__main__":
    main()
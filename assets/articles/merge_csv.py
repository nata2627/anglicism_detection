import os
import pandas as pd
import glob
import chardet


def merge_csv_files(input_directory, output_file):
    """
    Объединяет все CSV файлы из указанной директории в один CSV файл
    и удаляет пустые строки перед сохранением

    Args:
        input_directory (str): Путь к директории с CSV файлами
        output_file (str): Путь к выходному CSV файлу
    """
    # Получаем список всех CSV файлов в директории
    csv_files = glob.glob(os.path.join(input_directory, "*.csv"))

    if not csv_files:
        print(f"В директории {input_directory} не найдено CSV файлов")
        return

    # Создаем пустой список для хранения данных из каждого файла
    all_dataframes = []

    # Читаем каждый файл и добавляем его в список
    for file in csv_files:
        try:
            # Определение кодировки файла
            with open(file, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(10000))
            encoding = result['encoding']

            # Чтение с определенной кодировкой
            df = pd.read_csv(file, encoding=encoding)
            all_dataframes.append(df)
            print(f"Прочитан файл: {os.path.basename(file)} (кодировка: {encoding})")
        except Exception as e:
            print(f"Ошибка при чтении файла {os.path.basename(file)}: {e}")
            # Пробуем запасные варианты кодировок
            for enc in ['utf-8', 'cp1251', 'latin1', 'koi8-r', 'utf-16']:
                try:
                    df = pd.read_csv(file, encoding=enc)
                    all_dataframes.append(df)
                    print(f"Успешно прочитан с кодировкой {enc}: {os.path.basename(file)}")
                    break
                except:
                    continue

    if not all_dataframes:
        print("Не удалось прочитать ни один файл")
        return

    # Объединяем все DataFrame в один
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Удаляем пустые строки
    initial_rows = len(combined_df)
    combined_df = combined_df.dropna(how='all')  # Удаляем строки, где все значения NaN

    # Дополнительно удаляем строки, где все значения - пустые строки
    combined_df = combined_df.loc[~(combined_df.astype(str).applymap(lambda x: x.strip() == '').all(axis=1))]

    removed_rows = initial_rows - len(combined_df)
    if removed_rows > 0:
        print(f"Удалено {removed_rows} пустых строк")

    # Сохраняем объединенный DataFrame в новый CSV файл с кодировкой utf-8
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Все файлы объединены и сохранены в {output_file}")
    print(f"Итоговый файл содержит {len(combined_df)} строк")


if __name__ == "__main__":
    # Укажите путь к директории с CSV файлами
    input_dir = "assets/articles"

    # Укажите путь к выходному файлу
    output_file = "assets/articles.csv"

    merge_csv_files(input_dir, output_file)
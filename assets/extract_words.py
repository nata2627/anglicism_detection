# Создает список уникальных слов из csv файла, сохраняет в txt

import csv
import json
import re


def extract_unique_words(csv_file_path, output_txt_path):
    # Множество для хранения уникальных слов
    unique_words = set()

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            # Читаем CSV файл
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                if len(row) > 1:
                    # Получаем второй столбец
                    words_column = row[1]

                    try:
                        # Парсим JSON-подобную строку со списком слов
                        # Заменяем одинарные кавычки на двойные для правильного парсинга JSON
                        words_column = words_column.replace("'", '"')

                        # Обрабатываем случай, когда в данных могут быть экранированные кавычки
                        words_list = json.loads(words_column)

                        # Добавляем слова в множество уникальных слов
                        for word in words_list:
                            unique_words.add(word)
                    except json.JSONDecodeError:
                        # Если не удалось распарсить через JSON, используем регулярное выражение
                        pattern = r'"([^"]+)"'
                        matches = re.findall(pattern, words_column)
                        for word in matches:
                            unique_words.add(word)

        # Сортируем слова по алфавиту
        sorted_words = sorted(list(unique_words))

        # Записываем уникальные слова в txt файл
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            for word in sorted_words:
                txt_file.write(word + '\n')

        print(f"Успешно извлечено {len(sorted_words)} уникальных слов и сохранено в {output_txt_path}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")


# Пути к файлам
input_file = 'assets/anglicisms_dataset.csv'
output_file = 'assets/unique_words.txt'

# Запускаем функцию
extract_unique_words(input_file, output_file)
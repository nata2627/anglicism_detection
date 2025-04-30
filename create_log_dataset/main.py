import csv
import json
import re
import string
import os
import random
from collections import Counter
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from pymorphy2 import MorphAnalyzer

# Инициализация инструментов
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph = MorphAnalyzer()

# Константы
MISSING_TOKEN = "<NONE>"  # специальный знак для отсутствующих слов
INPUT_FILE = "assets/anglicisms_dataset.csv"
OUTPUT_FILE = "assets/dataset_log.csv"
BALANCED_OUTPUT_FILE = "assets/dataset_balanced.csv"  # новый файл для сбалансированного датасета
MIN_WORD_LENGTH = 3  # минимальная длина слова для обработки

# Требуемые граммемы
REQUIRED_FEATURES = ['Animacy', 'Aspect', 'Case', 'Gender', 'Foreign', 'Number']

# Создаем директорию assets, если её нет
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# Функция для подсчета букв в слове
def count_letters(word):
    # Приводим к нижнему регистру
    word = word.lower()
    # Подсчет отдельных букв
    letter_counts = Counter(word)

    # Подсчет сдвоенных букв
    doubles = []
    for i in range(len(word) - 1):
        if word[i] == word[i + 1]:
            doubles.append(word[i] + word[i + 1])
    double_counts = Counter(doubles)

    return letter_counts, double_counts


# Функция для получения контекста слова
def get_context(tokens, word_index):
    # Получаем слова вокруг целевого слова
    left_left = tokens[word_index - 2] if word_index >= 2 else MISSING_TOKEN
    left = tokens[word_index - 1] if word_index >= 1 else MISSING_TOKEN
    right = tokens[word_index + 1] if word_index < len(tokens) - 1 else MISSING_TOKEN
    right_right = tokens[word_index + 2] if word_index < len(tokens) - 2 else MISSING_TOKEN

    return left_left, left, right, right_right


# Функция для токенизации текста
def tokenize_text(text):
    # Удаляем знаки препинания и разбиваем на слова
    # Используем более точную токенизацию с помощью Natasha
    doc = Doc(text)
    doc.segment(segmenter)
    tokens = [token.text for token in doc.tokens if re.match(r'\w+', token.text)]
    return tokens


# Функция для проверки, является ли слово англицизмом
def is_anglicism(word, anglicisms):
    return word.lower() in [a.lower() for a in anglicisms]


# Функция для анализа слова и получения его характеристик
def analyze_word(word, is_angl=0):
    # Если слово - пустое или специальный токен, возвращаем значения по умолчанию
    if word == MISSING_TOKEN:
        return {
            'text': MISSING_TOKEN,
            'lemma': MISSING_TOKEN,
            'is_anglicism': 0,
            'length': 0,
            'is_capitalized': 0,
            'features': {feature: '' for feature in REQUIRED_FEATURES}
        }

    # Анализ с помощью Natasha
    doc = Doc(word)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    # Если токенов нет, возвращаем значения по умолчанию с текстом слова
    if not doc.tokens:
        return {
            'text': word,
            'lemma': word,
            'is_anglicism': is_angl,
            'length': len(word),
            'is_capitalized': 1 if word[0].isupper() else 0,
            'features': {feature: '' for feature in REQUIRED_FEATURES}
        }

    token = doc.tokens[0]
    token.lemmatize(morph_vocab)

    # Получаем лемму и граммемы
    lemma = token.lemma
    features = token.feats if token.feats else {}

    return {
        'text': word,
        'lemma': lemma,
        'is_anglicism': is_angl,
        'length': len(word),
        'is_capitalized': 1 if word[0].isupper() else 0,
        'features': {feature: features.get(feature, '') for feature in REQUIRED_FEATURES}
    }


# Основная функция обработки всех слов в датасете
def process_dataset(max_samples_per_class=10000):
    """
    Обработка датасета и создание сбалансированного набора данных.

    Args:
        max_samples_per_class (int): Максимальное количество образцов для каждого класса.
                                     Если указано, датасет будет сокращен до указанного
                                     количества примеров каждого класса.
    """
    # Собираем все возможные буквы русского алфавита
    all_letters = set(chr(i) for i in range(ord('а'), ord('я') + 1)) | {'ё'}
    # Создаем множество всех возможных сдвоенных букв
    all_double_letters = {letter + letter for letter in all_letters}

    print(f"Минимальная длина слова для обработки: {MIN_WORD_LENGTH} символа")

    # Собираем данные для анализа
    data = []

    # Счетчики для отслеживания собранных примеров каждого класса
    collected_anglicisms = 0
    collected_non_anglicisms = 0

    # Чтение входного файла с отображением прогресса
    print(f"Начинаем обработку файла. Лимит на класс: {max_samples_per_class} примеров.")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row_index, row in enumerate(reader):
            if not row:
                continue

            # Выводим статус обработки каждые 100 строк
            if row_index % 100 == 0:
                print(
                    f"Обработано строк: {row_index}. Собрано англицизмов: {collected_anglicisms}/{max_samples_per_class}, "
                    f"обычных слов: {collected_non_anglicisms}/{max_samples_per_class}")

            # Если уже собрали нужное количество примеров обоих классов, прерываем обработку
            if collected_anglicisms >= max_samples_per_class and collected_non_anglicisms >= max_samples_per_class:
                print(f"Достигнуто необходимое количество примеров для обоих классов. Прерываем обработку.")
                break

            text = row[0]
            try:
                # Парсим JSON в строке для получения англицизмов
                anglicisms_json = row[1]
                anglicisms = json.loads(anglicisms_json)

                # Токенизация текста
                tokens = tokenize_text(text)

                # Фильтруем слова (минимум 3 символа)
                valid_tokens = [word for word in tokens if len(word) >= 3]

                # Найдем все англицизмы и не-англицизмы в тексте
                text_anglicisms = []
                text_non_anglicisms = []

                for word in valid_tokens:
                    if is_anglicism(word, anglicisms):
                        text_anglicisms.append(word)
                    else:
                        text_non_anglicisms.append(word)

                # Для баланса выбираем случайные не-англицизмы в том же количестве, что и англицизмы
                anglicisms_count = len(text_anglicisms)
                selected_non_anglicisms = []

                if anglicisms_count > 0 and len(text_non_anglicisms) > 0:
                    # Выбираем не больше, чем есть не-англицизмов
                    count_to_select = min(anglicisms_count, len(text_non_anglicisms))
                    selected_non_anglicisms = random.sample(text_non_anglicisms, count_to_select)

                # Обрабатываем все англицизмы и выбранные не-англицизмы
                words_to_process = text_anglicisms + selected_non_anglicisms

                # Чтобы обеспечить уникальность, создаем индексы для обработанных слов
                word_indices = {}
                for i, word in enumerate(tokens):
                    if word in words_to_process and word not in word_indices:
                        word_indices[word] = i

                # Обработка выбранных слов в тексте
                for word, word_index in word_indices.items():
                    # Пропускаем слишком короткие слова и числа
                    if len(word) < 3 or word.isdigit():
                        continue

                    # Проверяем, является ли слово англицизмом
                    is_angl = 1 if is_anglicism(word, anglicisms) else 0

                    # Если уже достаточно примеров данного класса, пропускаем
                    if (is_angl == 1 and collected_anglicisms >= max_samples_per_class) or \
                            (is_angl == 0 and collected_non_anglicisms >= max_samples_per_class):
                        continue

                    # Обновляем счетчики собранных примеров
                    if is_angl == 1:
                        collected_anglicisms += 1
                    else:
                        collected_non_anglicisms += 1

                    # Получаем контекст

                    # Получаем контекст
                    left_left_text, left_text, right_text, right_right_text = get_context(tokens, word_index)

                    # Анализируем текущее слово
                    current_word_info = analyze_word(word, is_angl)

                    # Анализируем слова контекста
                    left_left_info = analyze_word(left_left_text, 1 if is_anglicism(left_left_text, anglicisms) else 0)
                    left_info = analyze_word(left_text, 1 if is_anglicism(left_text, anglicisms) else 0)
                    right_info = analyze_word(right_text, 1 if is_anglicism(right_text, anglicisms) else 0)
                    right_right_info = analyze_word(right_right_text,
                                                    1 if is_anglicism(right_right_text, anglicisms) else 0)

                    # Подсчет букв для текущего слова
                    letter_counts, double_letter_counts = count_letters(word)

                    # Сохраняем данные основного слова
                    entry = {
                        'word': current_word_info['text'],
                        'lemma': current_word_info['lemma'],
                        'is_anglicism': current_word_info['is_anglicism'],
                        'length': current_word_info['length'],
                        'is_capitalized': current_word_info['is_capitalized'],
                        'features': current_word_info['features'],

                        # Информация о контексте
                        'left_left': left_left_info['text'],
                        'left_left_length': left_left_info['length'],
                        'left_left_is_anglicism': left_left_info['is_anglicism'],
                        'left_left_is_capitalized': left_left_info['is_capitalized'],
                        'left_left_features': left_left_info['features'],

                        'left': left_info['text'],
                        'left_length': left_info['length'],
                        'left_is_anglicism': left_info['is_anglicism'],
                        'left_is_capitalized': left_info['is_capitalized'],
                        'left_features': left_info['features'],

                        'right': right_info['text'],
                        'right_length': right_info['length'],
                        'right_is_anglicism': right_info['is_anglicism'],
                        'right_is_capitalized': right_info['is_capitalized'],
                        'right_features': right_info['features'],

                        'right_right': right_right_info['text'],
                        'right_right_length': right_right_info['length'],
                        'right_right_is_anglicism': right_right_info['is_anglicism'],
                        'right_right_is_capitalized': right_right_info['is_capitalized'],
                        'right_right_features': right_right_info['features'],

                        # Информация о буквах
                        'letter_counts': letter_counts,
                        'double_letter_counts': double_letter_counts
                    }
                    data.append(entry)

            except (json.JSONDecodeError, IndexError) as e:
                print(f"Ошибка при обработке строки: {row}, ошибка: {e}")
                continue

    # Статистика собранных данных
    print(f"Обработка завершена. Собрано {len(data)} записей:")
    print(f"- Англицизмов: {collected_anglicisms}")
    print(f"- Обычных слов: {collected_non_anglicisms}")

    # Создаем заголовки CSV
    headers = ['word', 'lemma', 'is_anglicism', 'length', 'is_capitalized',
               'left_left', 'left', 'right', 'right_right']

    # Добавляем граммемы для основного слова
    for feature in REQUIRED_FEATURES:
        headers.append(feature)

    # Добавляем информацию о контексте и их граммемы
    context_positions = ['left_left', 'left', 'right', 'right_right']
    for pos in context_positions:
        headers.extend([
            f'{pos}_length',
            f'{pos}_is_anglicism',
            f'{pos}_is_capitalized'
        ])
        for feature in REQUIRED_FEATURES:
            headers.append(f'{pos}_{feature}')

    # Добавляем счетчики букв
    for letter in sorted(all_letters):
        headers.append(f'count_{letter}')

    # Добавляем счетчики сдвоенных букв
    for double in sorted(all_double_letters):
        headers.append(f'count_{double}')

    # Подготовка данных для записи
    rows_to_write = []
    for entry in data:
        row_dict = {
            'word': entry['word'],
            'lemma': entry['lemma'],
            'is_anglicism': entry['is_anglicism'],
            'length': entry['length'],
            'is_capitalized': entry['is_capitalized'],
            'left_left': entry['left_left'],
            'left': entry['left'],
            'right': entry['right'],
            'right_right': entry['right_right']
        }

        # Заполняем граммемы для основного слова
        for feature in REQUIRED_FEATURES:
            row_dict[feature] = entry['features'].get(feature, '')

        # Заполняем информацию о контексте
        for pos in context_positions:
            row_dict[f'{pos}_length'] = entry[f'{pos}_length']
            row_dict[f'{pos}_is_anglicism'] = entry[f'{pos}_is_anglicism']
            row_dict[f'{pos}_is_capitalized'] = entry[f'{pos}_is_capitalized']

            # Заполняем граммемы для слов контекста
            for feature in REQUIRED_FEATURES:
                row_dict[f'{pos}_{feature}'] = entry[f'{pos}_features'].get(feature, '')

        # Заполняем счетчики букв
        for letter in sorted(all_letters):
            row_dict[f'count_{letter}'] = entry['letter_counts'].get(letter, 0)

        # Заполняем счетчики сдвоенных букв
        for double in sorted(all_double_letters):
            row_dict[f'count_{double}'] = entry['double_letter_counts'].get(double, 0)

        rows_to_write.append(row_dict)

    # Запись в выходной файл (несбалансированный)
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"Несбалансированный датасет сохранен в {OUTPUT_FILE}")

    # Балансировка классов и учет max_samples_per_class
    anglicisms = [row for row in rows_to_write if row['is_anglicism'] == 1]
    non_anglicisms = [row for row in rows_to_write if row['is_anglicism'] == 0]

    print(f"Найдено англицизмов: {len(anglicisms)}, обычных слов: {len(non_anglicisms)}")

    # Ограничиваем размер каждого класса до max_samples_per_class
    if len(anglicisms) > max_samples_per_class:
        print(f"Сокращаем количество англицизмов с {len(anglicisms)} до {max_samples_per_class}")
        anglicisms = random.sample(anglicisms, max_samples_per_class)

    # Определяем количество не-англицизмов для баланса (равно количеству англицизмов, но не больше max_samples_per_class)
    target_non_anglicisms = min(len(anglicisms), max_samples_per_class)

    if len(non_anglicisms) > target_non_anglicisms:
        print(f"Сокращаем количество обычных слов с {len(non_anglicisms)} до {target_non_anglicisms}")
        balanced_non_anglicisms = random.sample(non_anglicisms, target_non_anglicisms)
    else:
        print(f"Внимание: обычных слов ({len(non_anglicisms)}) меньше целевого количества ({target_non_anglicisms}).")
        balanced_non_anglicisms = non_anglicisms

    balanced_dataset = anglicisms + balanced_non_anglicisms

    # Перемешиваем сбалансированный датасет
    random.shuffle(balanced_dataset)

    # Запись сбалансированного датасета
    with open(BALANCED_OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(balanced_dataset)

    print(f"Сбалансированный датасет сохранен в {BALANCED_OUTPUT_FILE}")
    print(f"Размер сбалансированного датасета: {len(balanced_dataset)} записей "
          f"(по {len(anglicisms)} примеров каждого класса)")


if __name__ == "__main__":
    # Вызываем функцию с ограничением в 10000 примеров на класс
    process_dataset(max_samples_per_class=10000)
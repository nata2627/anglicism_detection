import pandas as pd
import re
import os
import json
from nltk.stem.snowball import SnowballStemmer
import nltk


def main():
    # Пути к файлам
    anglicisms_file = "assets/clean_anglicism.txt"
    stopwords_file = "assets/stopwords.txt"
    texts_file = "assets/texts.csv"
    output_file = "assets/anglicisms_dataset.csv"

    print("Настройка NLTK для обработки русского текста...")
    try:
        nltk.download('punkt', quiet=True)
        stemmer = SnowballStemmer("russian")
    except Exception as e:
        print(f"Ошибка при инициализации стеммера: {e}")
        return

    # Загрузка стоп-слов
    print(f"Проверка наличия файла стоп-слов {stopwords_file}...")
    stopwords = []
    if os.path.exists(stopwords_file):
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = [line.strip().lower() for line in f]
            print(f"Загружено {len(stopwords)} стоп-слов")
        except Exception as e:
            print(f"Ошибка при загрузке стоп-слов: {e}")
            print("Продолжаем без стоп-слов")
    else:
        print(f"Файл стоп-слов не найден. Будем использовать стандартный список русских стоп-слов NLTK.")
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords as nltk_stopwords
            stopwords = nltk_stopwords.words('russian')
            print(f"Загружено {len(stopwords)} стоп-слов из NLTK")
        except Exception as e:
            print(f"Ошибка при загрузке стоп-слов из NLTK: {e}")
            print("Продолжаем без стоп-слов")

    # Создаем множество из стемов стоп-слов для быстрой проверки
    stopwords_stems = set(stemmer.stem(word) for word in stopwords)

    print(f"Загрузка англицизмов из {anglicisms_file}...")
    # Загрузка списка англицизмов
    try:
        with open(anglicisms_file, 'r', encoding='utf-8') as f:
            # Загружаем англицизмы и применяем стемминг
            all_anglicisms = [line.strip().lower() for line in f]

            # Фильтруем англицизмы, убирая те, которые есть в стоп-словах
            filtered_anglicisms = []
            for word in all_anglicisms:
                word_stem = stemmer.stem(word)
                if word_stem not in stopwords_stems:
                    filtered_anglicisms.append(word_stem)

            print(f"Загружено {len(all_anglicisms)} англицизмов")
            print(f"После фильтрации стоп-слов осталось {len(filtered_anglicisms)} англицизмов")

            # Сохраняем отфильтрованные англицизмы в множество для быстрого поиска
            anglicisms_base_forms = set(filtered_anglicisms)

        # Выводим примеры для проверки
        if len(filtered_anglicisms) > 5:
            print("Примеры англицизмов после фильтрации (в форме основы):")
            for i, anglicism in enumerate(list(filtered_anglicisms)[:5]):
                print(f"  {i + 1}. {anglicism}")
    except Exception as e:
        print(f"Ошибка при загрузке файла с англицизмами: {e}")
        return

    print(f"Загрузка текстов из {texts_file}...")
    # Загрузка текстов
    try:
        df = pd.read_csv(texts_file)
        print(f"Загружено {len(df)} текстов")
    except Exception as e:
        print(f"Ошибка при загрузке файла с текстами: {e}")
        print("Пробуем альтернативный способ загрузки...")
        try:
            df = pd.read_csv(texts_file, encoding='utf-8-sig')
            print(f"Загружено {len(df)} текстов (с кодировкой utf-8-sig)")
        except Exception as e2:
            print(f"Ошибка при альтернативной загрузке: {e2}")
            return

    # Проверка наличия нужного столбца
    if 'original_text' not in df.columns:
        print(f"Ошибка: в файле {texts_file} отсутствует столбец 'original_text'")
        print(f"Доступные столбцы: {df.columns.tolist()}")
        return

    # Функция для извлечения текста в кавычках
    def extract_quoted_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return set()

        # Ищем текст в различных типах кавычек
        # «» - типографские кавычки, "" - прямые кавычки, '' - одинарные кавычки
        quote_patterns = [
            r'«([^»]+)»',  # «текст»
            r'"([^"]+)"',  # "текст"
            r'\'([^\']+)\'',  # 'текст'
        ]

        quoted_words = set()

        for pattern in quote_patterns:
            try:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Разбиваем текст в кавычках на отдельные слова, включая слова через дефис
                    words_in_quotes = re.findall(r'\b[а-яА-ЯёЁ]+-?[а-яА-ЯёЁ]*\b', match)
                    quoted_words.update(words_in_quotes)
            except Exception as e:
                print(f"Ошибка при обработке шаблона '{pattern}': {e}")

        return quoted_words

    # Функция для определения, есть ли перед словом точка
    def has_dot_before(text, word_pos):
        # Если слово в начале текста, то перед ним точки нет
        if word_pos == 0:
            return False

        # Проверяем символы перед началом слова
        for i in range(word_pos - 1, -1, -1):
            if text[i].isspace():
                continue
            return text[i] == '.' or text[i] == '!' or text[i] == '?'

        return False

    # Функция для поиска англицизмов в тексте
    def find_anglicisms(text):
        if pd.isna(text) or not isinstance(text, str):
            return []

        # Извлекаем слова, находящиеся в кавычках
        quoted_words = extract_quoted_text(text)
        # Создаем множество стемов слов в кавычках для быстрой проверки
        quoted_stems = set(stemmer.stem(word.lower()) for word in quoted_words)

        # Разбиваем текст на слова, включая слова через дефис, но без цифр
        # Используем регулярное выражение для поиска русских слов без цифр
        word_matches = re.finditer(r'\b[а-яА-ЯёЁ]+-?[а-яА-ЯёЁ]+\b', text)

        found_anglicisms = []
        word_stems = {}  # Кэш для стеммированных слов

        for match in word_matches:
            try:
                word = match.group(0)
                word_pos = match.start()

                # Пропускаем слова короче 3 символов для избежания ложных срабатываний
                if len(word) < 3:
                    continue

                # Проверяем условие: если слово с большой буквы и перед ним нет точки, то оно не англицизм
                if word[0].isupper() and not has_dot_before(text, word_pos):
                    continue

                # Проверяем, содержит ли слово цифры (хотя регулярка выше должна это исключать)
                if any(char.isdigit() for char in word):
                    continue

                # Приводим к нижнему регистру для дальнейшей обработки
                word_lower = word.lower()

                # Применяем стемминг
                if word_lower not in word_stems:
                    word_stems[word_lower] = stemmer.stem(word_lower)
                word_stem = word_stems[word_lower]

                # Проверяем, что:
                # 1. Основа слова есть в списке основ англицизмов
                # 2. Основа слова НЕ находится в списке стоп-слов
                # 3. Слово НЕ находится внутри кавычек (не является частью имени собственного)
                if (word_stem in anglicisms_base_forms and
                        word_stem not in quoted_stems):

                    # Дополнительная проверка, не является ли слово частью текста в кавычках
                    is_in_quotes = False
                    for quoted_word in quoted_words:
                        if word.lower() == quoted_word.lower():
                            is_in_quotes = True
                            break

                    # Если слово не в кавычках и ещё не добавлено, добавляем его
                    if not is_in_quotes and word not in found_anglicisms:
                        found_anglicisms.append(word)
            except Exception as e:
                print(f"Ошибка при обработке слова '{word}': {e}")

        return found_anglicisms

    print("Поиск англицизмов в текстах...")

    # Проверяем наличие tqdm для прогресс-бара
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Библиотека tqdm не установлена. Будет отображаться простой прогресс.")

    # Применяем функцию к каждому тексту
    results = []

    if use_tqdm:
        # С прогресс-баром
        for index, row in tqdm(df.iterrows(), total=len(df)):
            text = row['original_text']
            anglicisms = find_anglicisms(text)
            results.append({
                'original_text': text,
                'anglicisms': anglicisms
            })
    else:
        # Без прогресс-бара
        total = len(df)
        for index, row in df.iterrows():
            if index % 100 == 0 or index == total - 1:
                print(f"Обработано {index + 1}/{total} текстов ({((index + 1) / total * 100):.1f}%)")
            text = row['original_text']
            anglicisms = find_anglicisms(text)
            results.append({
                'original_text': text,
                'anglicisms': anglicisms
            })

    # Создаем датасет
    print("Создание итогового датасета...")
    output_df = pd.DataFrame(results)

    # Сохраняем результат
    print(f"Сохранение результата в {output_file}...")

    # Преобразуем списки англицизмов в строки JSON для сохранения в CSV
    output_df['anglicisms'] = output_df['anglicisms'].apply(json.dumps, ensure_ascii=False)
    output_df.to_csv(output_file, index=False, encoding='utf-8')

    # Выводим статистику
    texts_with_anglicisms = sum(1 for anglicisms_json in output_df['anglicisms'] if anglicisms_json != '[]')
    print(f"Обработка завершена. Найдены англицизмы в {texts_with_anglicisms} из {len(output_df)} текстов.")

    # Выводим примеры для проверки
    print("\nПримеры найденных англицизмов:")
    samples = output_df[output_df['anglicisms'] != '[]'].head(5)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        anglicisms = json.loads(row['anglicisms'])
        text = row['original_text']
        if len(text) > 100:
            text = text[:100] + "..."
        print(f"Пример {i}:")
        print(f"  Текст: {text}")
        print(f"  Англицизмы: {', '.join(anglicisms)}")
        print()


if __name__ == "__main__":
    main()
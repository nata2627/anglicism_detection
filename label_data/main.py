import pandas as pd
import re
import os
import json
import nltk
from natasha import Segmenter, MorphVocab, Doc, NewsEmbedding, NewsMorphTagger


def main():
    # Пути к файлам
    anglicisms_file = "assets/clean_anglicism_2.txt"
    stopwords_file = "assets/stopwords.txt"
    exceptions_file = "assets/exceptions_lemma.txt"  # Файл с исключениями в лемматизированной форме
    texts_file = "assets/texts.csv"
    output_file = "assets/anglicisms_dataset.csv"

    print("Настройка natasha для лемматизации русского текста...")
    try:
        # Инициализируем компоненты natasha
        segmenter = Segmenter()
        morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)

        # Загружаем также nltk для tokenize
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Ошибка при инициализации компонентов natasha: {e}")
        return

    # Функция для лемматизации слова с помощью natasha
    def lemmatize_word(word):
        doc = Doc(word)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        # Возвращаем лемму первого токена (для одного слова будет только один токен)
        if doc.tokens:
            return doc.tokens[0].lemma
        return word  # Если не удалось лемматизировать, возвращаем исходное слово

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

    # Создаем множество из лемм стоп-слов для быстрой проверки
    stopwords_lemmas = set(lemmatize_word(word) for word in stopwords)

    # Загрузка исключений (они уже в лемматизированной форме)
    print(f"Проверка наличия файла исключений {exceptions_file}...")
    exceptions_lemmas = set()
    if os.path.exists(exceptions_file):
        try:
            with open(exceptions_file, 'r', encoding='utf-8') as f:
                # ИЗМЕНЕНО: Теперь слова уже в лемматизированной форме, применять lemmatize_word не нужно
                exceptions_lemmas = set(line.strip().lower() for line in f)
            print(f"Загружено {len(exceptions_lemmas)} лемматизированных слов-исключений")
        except Exception as e:
            print(f"Ошибка при загрузке исключений: {e}")
            print("Продолжаем без исключений")
    else:
        print(f"Файл исключений не найден. Продолжаем без списка исключений.")

    print(f"Загрузка англицизмов из {anglicisms_file}...")
    # Загрузка списка англицизмов
    try:
        with open(anglicisms_file, 'r', encoding='utf-8') as f:
            # Загружаем англицизмы и применяем лемматизацию
            all_anglicisms = [line.strip().lower() for line in f]

            # Фильтруем англицизмы, убирая те, которые есть в стоп-словах
            filtered_anglicisms = []
            for word in all_anglicisms:
                word_lemma = lemmatize_word(word)
                if word_lemma not in stopwords_lemmas:
                    filtered_anglicisms.append(word_lemma)

            print(f"Загружено {len(all_anglicisms)} англицизмов")
            print(f"После фильтрации стоп-слов осталось {len(filtered_anglicisms)} англицизмов")

            # Сохраняем отфильтрованные англицизмы в множество для быстрого поиска
            anglicisms_base_forms = set(filtered_anglicisms)

        # Выводим примеры для проверки
        if len(filtered_anglicisms) > 5:
            print("Примеры англицизмов после фильтрации (в форме леммы):")
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

    # Функция для разделения текста на предложения
    def split_text_into_sentences(text):
        if pd.isna(text) or not isinstance(text, str):
            return []

        # Используем natasha для разделения на предложения
        doc = Doc(text)
        doc.segment(segmenter)
        sentences = [sent.text for sent in doc.sents]

        # Удаляем пустые предложения и предложения только из пробелов
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

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
        # Создаем множество лемм слов в кавычках для быстрой проверки
        quoted_lemmas = set(lemmatize_word(word.lower()) for word in quoted_words)

        # Разбиваем текст на слова, включая слова через дефис
        # \b - граница слова, далее русские буквы, затем опционально дефис и снова русские буквы
        word_matches = re.finditer(r'\b[а-яА-ЯёЁ]+(?:-[а-яА-ЯёЁ]+)*\b', text)

        found_anglicisms = []
        word_lemmas = {}  # Кэш для лемматизированных слов

        for match in word_matches:
            try:
                word = match.group(0)
                word_pos = match.start()

                # Пропускаем слова короче 3 символов для избежания ложных срабатываний
                if len(word) < 3:
                    continue

                # Пропускаем словосочетания (проверяем наличие пробелов)
                if ' ' in word:
                    continue

                # Проверяем условие: если слово с большой буквы и перед ним нет точки, то оно не англицизм
                if word[0].isupper() and not has_dot_before(text, word_pos):
                    continue

                # Проверяем, содержит ли слово цифры
                if any(char.isdigit() for char in word):
                    continue

                # Приводим к нижнему регистру для дальнейшей обработки
                word_lower = word.lower()

                # Применяем лемматизацию
                if word_lower not in word_lemmas:
                    word_lemmas[word_lower] = lemmatize_word(word_lower)
                word_lemma = word_lemmas[word_lower]

                # Проверяем, что:
                # 1. Лемма слова есть в списке лемм англицизмов
                # 2. Лемма слова НЕ находится в списке стоп-слов
                # 3. Лемма слова НЕ находится в списке исключений (теперь исключения уже в лемматизированной форме)
                # 4. Слово НЕ находится внутри кавычек (не является частью имени собственного)
                if (word_lemma in anglicisms_base_forms and
                        word_lemma not in stopwords_lemmas and
                        word_lemma not in exceptions_lemmas and
                        word_lemma not in quoted_lemmas):

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

    print("Разделение текстов на предложения и поиск англицизмов...")

    # Проверяем наличие tqdm для прогресс-бара
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Библиотека tqdm не установлена. Будет отображаться простой прогресс.")

    # Создаем множество для отслеживания уникальных предложений
    unique_sentences = set()

    # Применяем функцию к каждому тексту, разделяя на предложения
    results = []
    total_sentences = 0
    duplicate_sentences = 0

    if use_tqdm:
        # С прогресс-баром
        for index, row in tqdm(df.iterrows(), total=len(df)):
            text = row['original_text']
            sentences = split_text_into_sentences(text)
            total_sentences += len(sentences)

            for sentence in sentences:
                # Пропускаем дубликаты предложений
                if sentence in unique_sentences:
                    duplicate_sentences += 1
                    continue

                # Добавляем предложение в множество уникальных
                unique_sentences.add(sentence)

                # Находим англицизмы в предложении
                anglicisms = find_anglicisms(sentence)

                # Добавляем результат
                results.append({
                    'sentence': sentence,
                    'anglicisms': anglicisms
                })
    else:
        # Без прогресс-бара
        total = len(df)
        for index, row in df.iterrows():
            if index % 100 == 0 or index == total - 1:
                print(f"Обработано {index + 1}/{total} текстов ({((index + 1) / total * 100):.1f}%)")

            text = row['original_text']
            sentences = split_text_into_sentences(text)
            total_sentences += len(sentences)

            for sentence in sentences:
                # Пропускаем дубликаты предложений
                if sentence in unique_sentences:
                    duplicate_sentences += 1
                    continue

                # Добавляем предложение в множество уникальных
                unique_sentences.add(sentence)

                # Находим англицизмы в предложении
                anglicisms = find_anglicisms(sentence)

                # Добавляем результат
                results.append({
                    'sentence': sentence,
                    'anglicisms': anglicisms
                })

    # Создаем датасет
    print("Создание итогового датасета...")
    output_df = pd.DataFrame(results)

    # Фильтрация предложений без англицизмов
    print(f"Всего предложений до фильтрации: {len(output_df)}")
    print(f"Всего дубликатов предложений пропущено: {duplicate_sentences}")

    # Закомментируем удаление предложений без англицизмов для тестирования
    output_df = output_df[output_df['anglicisms'].apply(len) > 0]

    print(f"Всего предложений после фильтрации: {len(output_df)}")

    # Сохраняем результат
    print(f"Сохранение результата в {output_file}...")

    # Преобразуем списки англицизмов в строки JSON для сохранения в CSV
    output_df['anglicisms'] = output_df['anglicisms'].apply(json.dumps, ensure_ascii=False)
    output_df.to_csv(output_file, index=False, encoding='utf-8', header=True)

    # Проверяем первую строку файла и удаляем её, если она содержит "sentence,anglicisms"
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Проверяем, содержит ли первая строка "sentence,anglicisms"
        if lines and "sentence,anglicisms" in lines[0]:
            print("Удаляем дублирующий заголовок из первой строки...")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(lines[1:])  # Записываем все строки, кроме первой
    except Exception as e:
        print(f"Ошибка при обработке файла после сохранения: {e}")

    # Выводим статистику
    sentences_with_anglicisms = sum(1 for anglicisms_json in output_df['anglicisms'] if anglicisms_json != '[]')
    print(f"Обработка завершена. Найдены англицизмы в {sentences_with_anglicisms} из {len(output_df)} предложений.")

    print(f"Всего извлечено {total_sentences} предложений из {len(df)} текстов.")
    print(f"Обнаружено и пропущено {duplicate_sentences} дубликатов предложений.")

    # Выводим информацию об использовании исключений
    if exceptions_lemmas:
        print(f"Во время обработки использовался список исключений ({len(exceptions_lemmas)} слов).")

    # Выводим примеры для проверки
    print("\nПримеры найденных англицизмов:")
    samples = output_df[output_df['anglicisms'] != '[]'].head(5)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        anglicisms = json.loads(row['anglicisms'])
        sentence = row['sentence']
        if len(sentence) > 100:
            sentence = sentence[:100] + "..."
        print(f"Пример {i}:")
        print(f"  Предложение: {sentence}")
        print(f"  Англицизмы: {', '.join(anglicisms)}")
        print()


if __name__ == "__main__":
    main()
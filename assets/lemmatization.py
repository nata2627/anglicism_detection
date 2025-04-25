# На вход txt файл со словами, на выходе txt файл с лемматизированными словами

import os
from natasha import Segmenter, MorphVocab, Doc, NewsEmbedding, NewsMorphTagger


def lemmatize_word(word, segmenter, morph_tagger, morph_vocab):
    """
    Лемматизирует отдельное слово с помощью natasha.

    Args:
        word (str): Исходное слово
        segmenter, morph_tagger, morph_vocab: Компоненты natasha

    Returns:
        str: Лемматизированное слово или исходное слово в случае ошибки
    """
    try:
        doc = Doc(word)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)

        # Возвращаем лемму первого токена (для одного слова будет только один токен)
        if doc.tokens:
            return doc.tokens[0].lemma.lower()  # Приводим к нижнему регистру
        return word.lower()  # Если не удалось лемматизировать, возвращаем исходное слово в нижнем регистре
    except Exception as e:
        print(f"Ошибка при лемматизации слова '{word}': {e}")
        return word.lower()  # В случае ошибки возвращаем исходное слово в нижнем регистре


def main():
    """
    Основная функция скрипта.
    """
    # Жестко заданные пути к файлам
    input_file = "assets/exceptions.txt"  # Путь к входному файлу
    output_file = "assets/exceptions_lemma.txt"  # Путь к выходному файлу

    print(f"Используется входной файл: {input_file}")
    print(f"Результат будет сохранен в: {output_file}")

    print("Инициализация компонентов Natasha для лемматизации...")

    # Инициализируем компоненты natasha
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    print(f"Чтение файла: {input_file}")

    # Проверяем существование входного файла
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден")
        return

    try:
        # Чтение всех слов из входного файла
        with open(input_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]

        print(f"Загружено {len(words)} слов из файла")

        # Лемматизация каждого слова
        print("Выполняется лемматизация...")
        lemmatized_words = set()  # Используем множество для хранения уникальных лемм

        for word in words:
            lemma = lemmatize_word(word, segmenter, morph_tagger, morph_vocab)
            if lemma:  # Добавляем только непустые леммы
                lemmatized_words.add(lemma)

        # Сортировка лемматизированных слов
        sorted_lemmas = sorted(lemmatized_words)

        print(f"Получено {len(sorted_lemmas)} уникальных лемматизированных слов")

        # Сохранение результата в выходной файл
        with open(output_file, 'w', encoding='utf-8') as f:
            for lemma in sorted_lemmas:
                f.write(f"{lemma}\n")

        print(f"Результат сохранен в файл: {output_file}")

    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")


if __name__ == "__main__":
    main()


import pandas as pd
from collections import Counter
import re
import nltk
from nltk.stem.snowball import SnowballStemmer

# Проверяем доступность библиотек для лемматизации
LEMMATIZER_LIB = None

try:
    # Пробуем импортировать pymystem3
    from pymystem3 import Mystem

    LEMMATIZER_LIB = "pymystem3"
    print("Используем pymystem3 для лемматизации")
except ImportError:
    try:
        # Если pymystem3 не установлен, пробуем natasha
        from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger

        LEMMATIZER_LIB = "natasha"
        print("Используем natasha для лемматизации")
    except ImportError:
        try:
            # Если natasha не установлена, пробуем pymorphy2
            import pymorphy2

            LEMMATIZER_LIB = "pymorphy2"
            print("Используем pymorphy2 для лемматизации")
        except ImportError:
            print(
                "Предупреждение: Ни одна из библиотек лемматизации (pymystem3, natasha, pymorphy2) не установлена. Лемматизация будет недоступна.")
            LEMMATIZER_LIB = None


def analyze_anglicisms(anglicisms_dict):
    """
    Проводит базовый анализ англицизмов.

    Args:
        anglicisms_dict: Словарь с данными англицизмов

    Returns:
        DataFrame: DataFrame с данными для анализа
    """
    all_anglicisms = anglicisms_dict['all_anglicisms']

    # Создаем DataFrame для анализа
    df = pd.DataFrame(all_anglicisms)

    # Анализ длины слов
    df['word_length'] = df['word'].apply(len)

    # Анализ начальных и конечных букв
    df['first_letter'] = df['word'].apply(lambda x: x[0].lower() if x else "")
    df['last_letter'] = df['word'].apply(lambda x: x[-1].lower() if x else "")

    # Анализ слоговой структуры (примерный подсчёт гласных)
    vowels = 'аеёиоуыэюя'
    df['vowel_count'] = df['word'].apply(lambda x: sum(1 for char in x.lower() if char in vowels))

    return df


def clean_anglicisms(df):
    """
    Очищает и нормализует данные англицизмов.

    Args:
        df: DataFrame с англицизмами

    Returns:
        DataFrame: Очищенный DataFrame
    """
    # Копия DataFrame для безопасного изменения
    clean_df = df.copy()

    # Настройки обработки
    lowercase = True
    remove_special_chars = True
    remove_duplicates = True

    # Приведение всех слов к нижнему регистру
    if lowercase:
        clean_df['word'] = clean_df['word'].str.lower()

    # Удаление лишних символов (например, двоеточий, запятых)
    if remove_special_chars:
        clean_df['word'] = clean_df['word'].str.replace(r'[^\w\s]', '', regex=True)

    # Удаление дубликатов
    if remove_duplicates:
        clean_df = clean_df.drop_duplicates('word')

    # Обновление длины слов после очистки
    clean_df['word_length'] = clean_df['word'].apply(len)

    return clean_df


def perform_stemming(df):
    """
    Выполняет стемминг слов для анализа корневых основ.

    Args:
        df: DataFrame с англицизмами

    Returns:
        DataFrame: DataFrame с добавленными стеммами
    """
    try:
        # Убедимся, что необходимые пакеты NLTK загружены
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    except Exception as e:
        print(f"Не удалось загрузить ресурсы NLTK: {e}")

    stemmer = SnowballStemmer("russian")

    # Добавляем стеммы слов
    df['stem'] = df['word'].apply(lambda x: stemmer.stem(x))

    return df


def perform_lemmatization(df):
    """
    Выполняет лемматизацию слов для получения нормальной формы
    с использованием доступной библиотеки.

    Args:
        df: DataFrame с англицизмами

    Returns:
        DataFrame: DataFrame с добавленными леммами
    """
    # Копируем DataFrame для безопасности
    lemmatized_df = df.copy()

    if LEMMATIZER_LIB == "pymystem3":
        # Используем pymystem3 (Яндекс.Mystem)
        try:
            mystem = Mystem()

            def get_lemma_mystem(word):
                try:
                    lemmas = mystem.lemmatize(word.strip())
                    # Mystem возвращает список, включая пробелы, берем первый непустой элемент
                    return ''.join([lemma for lemma in lemmas if lemma.strip()])
                except Exception as e:
                    print(f"Ошибка при лемматизации слова '{word}': {e}")
                    return word

            lemmatized_df['lemma'] = lemmatized_df['word'].apply(get_lemma_mystem)
            print("Лемматизация выполнена с использованием pymystem3")

        except Exception as e:
            print(f"Ошибка при инициализации pymystem3: {e}")
            lemmatized_df['lemma'] = lemmatized_df['word']

    elif LEMMATIZER_LIB == "natasha":
        # Используем natasha
        try:
            # Инициализация компонентов natasha
            segmenter = Segmenter()
            morph_vocab = MorphVocab()
            emb = NewsEmbedding()
            morph_tagger = NewsMorphTagger(emb)

            def get_lemma_natasha(word):
                try:
                    # В natasha нужно сначала сегментировать текст, затем разметить морфологию
                    doc = segmenter.segment(word)
                    doc.segment(segmenter)  # Сегментация на токены
                    doc.tag_morph(morph_tagger)  # Морфологическая разметка

                    # Лемматизация
                    for token in doc.tokens:
                        token.lemmatize(morph_vocab)

                    # Возвращаем лемму первого токена (у нас обычно только одно слово)
                    if doc.tokens:
                        return doc.tokens[0].lemma if doc.tokens[0].lemma else word
                    return word
                except Exception as e:
                    print(f"Ошибка при лемматизации слова '{word}' с natasha: {e}")
                    return word

            lemmatized_df['lemma'] = lemmatized_df['word'].apply(get_lemma_natasha)
            print("Лемматизация выполнена с использованием natasha")

        except Exception as e:
            print(f"Ошибка при инициализации natasha: {e}")
            lemmatized_df['lemma'] = lemmatized_df['word']

    elif LEMMATIZER_LIB == "pymorphy2":
        # Пытаемся использовать pymorphy2, обрабатывая возможные ошибки
        try:
            import inspect
            # Проверяем, доступна ли функция getargspec в модуле inspect
            if not hasattr(inspect, 'getargspec'):
                # Если функция недоступна, подменяем её на getfullargspec
                inspect.getargspec = inspect.getfullargspec
                print("Внимание: применена совместимость для pymorphy2 на Python 3.11+")

            morph = pymorphy2.MorphAnalyzer()

            def get_lemma(word):
                try:
                    return morph.parse(word)[0].normal_form
                except Exception:
                    return word  # В случае ошибки возвращаем исходное слово

            lemmatized_df['lemma'] = lemmatized_df['word'].apply(get_lemma)
            print("Лемматизация выполнена с использованием pymorphy2")

        except Exception as e:
            print(f"Не удалось использовать pymorphy2 для лемматизации: {e}")
            lemmatized_df['lemma'] = lemmatized_df['word']
    else:
        # Если ни одна библиотека не доступна
        print("Лемматизация недоступна: используем исходные слова как леммы")
        lemmatized_df['lemma'] = lemmatized_df['word']

    return lemmatized_df


def extract_ngrams(df, n=2):
    """
    Извлекает n-граммы из слов.

    Args:
        df: DataFrame с англицизмами
        n: размер n-граммы (по умолчанию биграммы)

    Returns:
        dict: Словарь частотности n-грамм
    """
    ngrams = []

    for word in df['word']:
        if len(word) >= n:
            word_ngrams = [word[i:i + n] for i in range(len(word) - n + 1)]
            ngrams.extend(word_ngrams)

    return Counter(ngrams)


def extract_suffixes(df, min_length=2, max_length=4):
    """
    Извлекает потенциальные суффиксы из слов.

    Args:
        df: DataFrame с англицизмами
        min_length: минимальная длина суффикса
        max_length: максимальная длина суффикса

    Returns:
        dict: Словарь с частотностью суффиксов разной длины
    """
    suffixes = {length: Counter() for length in range(min_length, max_length + 1)}

    for word in df['word']:
        for length in range(min_length, max_length + 1):
            if len(word) > length:
                suffix = word[-length:]
                suffixes[length][suffix] += 1

    return suffixes


def advanced_analysis(df):
    """
    Выполняет расширенный анализ данных.

    Args:
        df: DataFrame с англицизмами

    Returns:
        dict: Словарь с результатами анализа
    """
    # Создаем словарь для хранения результатов анализа
    analysis_results = {}

    # Настройки категорий длины слов
    bins = [0, 4, 8, 12, 100]
    labels = ['Короткие (1-4)', 'Средние (5-8)', 'Длинные (9-12)', 'Очень длинные (>12)']

    # 1. Анализ длины слов
    length_stats = df['word_length'].describe()
    analysis_results['length_stats'] = length_stats

    # 2. Анализ частотности букв
    all_letters = ''.join(df['word'].str.lower())
    letter_freq = pd.Series(Counter(all_letters)).sort_values(ascending=False)
    analysis_results['letter_frequency'] = letter_freq

    # 3. Создаем новую колонку для категоризации длины слов
    df['length_category'] = pd.cut(df['word_length'], bins=bins, labels=labels, right=False)
    length_category_counts = df['length_category'].value_counts().sort_index()
    analysis_results['length_category_counts'] = length_category_counts

    # 4. Анализ частотности начальных и конечных букв
    first_letter_freq = df['first_letter'].value_counts().sort_values(ascending=False)
    last_letter_freq = df['last_letter'].value_counts().sort_values(ascending=False)
    analysis_results['first_letter_freq'] = first_letter_freq
    analysis_results['last_letter_freq'] = last_letter_freq

    # 5. Выполняем стемминг и анализируем стеммы
    stemmed_df = perform_stemming(df)
    stem_counts = stemmed_df['stem'].value_counts()
    analysis_results['stem_counts'] = stem_counts

    # 6. Выполняем лемматизацию и анализируем леммы
    lemmatized_df = perform_lemmatization(stemmed_df.copy())
    lemma_counts = lemmatized_df['lemma'].value_counts()
    analysis_results['lemma_counts'] = lemma_counts

    # 7. Сравнение результатов стемминга и лемматизации
    # Находим слова, которые имеют одинаковую лемму, но разные стеммы
    if 'lemma' in lemmatized_df.columns and 'stem' in lemmatized_df.columns:
        lemma_stem_groups = lemmatized_df.groupby('lemma')['stem'].nunique()
        multiple_stems = lemma_stem_groups[lemma_stem_groups > 1].index.tolist()

        if multiple_stems:
            interesting_cases = lemmatized_df[lemmatized_df['lemma'].isin(multiple_stems)].sort_values('lemma')
            analysis_results['lemma_stem_comparison'] = interesting_cases

    # 8. Биграммы
    bigrams = extract_ngrams(df, n=2)
    analysis_results['bigrams'] = bigrams

    # 9. Суффиксы
    suffixes = extract_suffixes(df)
    analysis_results['suffixes'] = suffixes

    return analysis_results
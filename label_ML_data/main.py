import pandas as pd
import re
import os
import json
import nltk
import numpy as np
from tqdm import tqdm
import torch
import pickle
import xgboost as xgb
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OneHotEncoder
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)


def detect_anglicism(sentence, segmenter, morph_vocab=None, morph_tagger=None):
    MISSING_TOKEN = "<NONE>"  # специальный знак для отсутствующих слов
    REQUIRED_FEATURES = ['Animacy', 'Aspect', 'Case', 'Gender', 'Foreign', 'Number']  # Требуемые граммемы

    # Кэш для анализа слов
    word_analysis_cache = {}
    # Кэш для подсчета букв
    letter_count_cache = {}

    def tokenize_text(text):
        doc = Doc(text)
        doc.segment(segmenter)
        tokens = [token.text for token in doc.tokens if re.match(r'\w+', token.text)]
        return tokens

    def get_context(tokens, word_index):  # Получаем слова вокруг целевого слова
        left_left = tokens[word_index - 2] if word_index >= 2 else MISSING_TOKEN
        left = tokens[word_index - 1] if word_index >= 1 else MISSING_TOKEN
        right = tokens[word_index + 1] if word_index < len(tokens) - 1 else MISSING_TOKEN
        right_right = tokens[word_index + 2] if word_index < len(tokens) - 2 else MISSING_TOKEN
        return left_left, left, right, right_right

    def analyze_word(word):
        # Проверяем, анализировали ли мы это слово раньше
        if word in word_analysis_cache:
            return word_analysis_cache[word]

        # Инициализация необходимых компонентов, если они не переданы в функцию
        nonlocal morph_vocab, morph_tagger
        if morph_vocab is None:
            morph_vocab = MorphVocab()
        if morph_tagger is None:
            emb = NewsEmbedding()
            morph_tagger = NewsMorphTagger(emb)

        # Если слово - пустое или специальный токен, возвращаем значения по умолчанию
        if word == MISSING_TOKEN:
            result = {
                'text': MISSING_TOKEN,
                'lemma': MISSING_TOKEN,
                'length': 0,
                'is_anglicism': 0,
                'is_capitalized': 0,
                'features': {feature: '' for feature in REQUIRED_FEATURES}
            }
            word_analysis_cache[word] = result
            return result

        # Анализ с помощью Natasha
        doc = Doc(word)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        # Если токенов нет, возвращаем значения по умолчанию с текстом слова
        if not doc.tokens:
            result = {
                'text': word,
                'lemma': word,
                'length': len(word),
                'is_anglicism': 0,
                'is_capitalized': 1 if word[0].isupper() else 0,
                'features': {feature: '' for feature in REQUIRED_FEATURES}
            }
            word_analysis_cache[word] = result
            return result

        token = doc.tokens[0]
        token.lemmatize(morph_vocab)

        # Получаем лемму и граммемы
        lemma = token.lemma
        features = token.feats if token.feats else {}

        result = {
            'text': word,
            'lemma': lemma,
            'length': len(word),
            'is_anglicism': 0,
            'is_capitalized': 1 if word[0].isupper() else 0,
            'features': {feature: features.get(feature, '') for feature in REQUIRED_FEATURES}
        }

        # Сохраняем результат в кэш
        word_analysis_cache[word] = result
        return result

    def count_letters(word):
        # Проверяем, есть ли слово в кэше
        if word in letter_count_cache:
            return letter_count_cache[word]

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

        result = (letter_counts, double_counts)
        letter_count_cache[word] = result
        return result

    # Собираем данные для анализа
    data = []
    tokens = tokenize_text(sentence)
    words_to_process = [word for word in tokens if len(word) >= 3]  # Фильтруем слова (минимум 3 символа)
    # Чтобы обеспечить уникальность, создаем индексы для обработанных слов
    word_indices = {}
    for i, word in enumerate(tokens):
        if word in words_to_process and word not in word_indices:
            word_indices[word] = i

    # Предварительно создаем все буквы и сдвоенные буквы
    all_letters = set(chr(i) for i in range(ord('а'), ord('я') + 1)) | {'ё'}
    all_double_letters = {letter + letter for letter in all_letters}

    # Обработка выбранных слов в тексте
    for word, word_index in word_indices.items():
        # Пропускаем слишком короткие слова и числа
        if len(word) < 3 or word.isdigit():
            continue

        # Получаем контекст
        left_left_text, left_text, right_text, right_right_text = get_context(tokens, word_index)
        # Анализируем текущее слово
        current_word_info = analyze_word(word)
        # Анализируем слова контекста
        left_left_info = analyze_word(left_left_text)
        left_info = analyze_word(left_text)
        right_info = analyze_word(right_text)
        right_right_info = analyze_word(right_right_text)

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

    return pd.DataFrame(rows_to_write)


def modify_table(df, tokenizer, model):
    def get_bert_embeddings(texts, model, tokenizer, max_length=128):
        """
        Получает эмбеддинги BERT для списка текстов

        Args:
            texts: список текстов
            model: модель BERT
            tokenizer: токенизатор BERT
            max_length: максимальная длина последовательности

        Returns:
            numpy array с эмбеддингами размерности (len(texts), embedding_dim)
        """
        embeddings = []

        # Обрабатываем тексты батчами
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Заменяем None значения на пустую строку
            batch_texts = [text if text is not None else "" for text in batch_texts]

            # Токенизация
            encoded_input = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            # Перемещаем на GPU, если доступен
            if torch.cuda.is_available():
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                model = model.cuda()

            # Получаем выходы модели без градиентов
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Берем эмбеддинг [CLS] токена из последнего слоя
            # для получения представления всего предложения
            batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

        # Объединяем все батчи
        return np.vstack(embeddings)

    # Словарь с предопределенными категориями для грамматических признаков
    predefined_categories = {
        'Animacy': ['Anim', 'Inan', '_NAN_'],
        'Aspect': ['Imp', 'Perf', '_NAN_'],
        'Case': ['Acc', 'Dat', 'Gen', 'Ins', 'Loc', 'Nom', 'Par', '_NAN_'],
        'Gender': ['Fem', 'Masc', 'Neut', '_NAN_'],
        'Foreign': ['Yes', '_NAN_'],
        'Number': ['Plur', 'Sing', '_NAN_']
    }

    text_columns = ['word', 'lemma', 'left_left', 'left', 'right', 'right_right']
    categorical_columns = [col for col in df.select_dtypes(include=['object']).columns
                           if col not in text_columns]

    # Создаем копию DataFrame для результатов
    df_transformed = df.copy()

    # Сохраняем список столбцов OHE для исключения из проверки мультиколлинеарности
    ohe_columns = []

    # Применяем OneHotEncoder к каждому категориальному столбцу
    for col in categorical_columns:
        # Обрабатываем пропущенные значения как отдельную категорию
        # Заменяем NaN на специальную строку "_NAN_" для обозначения пропущенных значений
        df_col = df[col].fillna("_NAN_")

        # Проверяем, является ли столбец одним из грамматических признаков
        is_grammar_feature = any(feature_name in col for feature_name in predefined_categories.keys())

        # Если это грамматический признак, используем предопределенные категории
        if is_grammar_feature:
            # Определяем, какой именно это грамматический признак
            feature_name = next((feature for feature in predefined_categories.keys() if feature in col), None)

            if feature_name:
                # Создаем OneHotEncoder с предопределенными категориями
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                # Предобучаем на всех возможных значениях этого признака
                encoder.fit(np.array(predefined_categories[feature_name]).reshape(-1, 1))

                # Применяем encoder к данным
                encoded_data = encoder.transform(df_col.values.reshape(-1, 1))
                encoded_cols = [f"{col}_ohe_{cat}" for cat in encoder.categories_[0]]
            else:
                # Обычный OneHotEncoder для других категориальных признаков
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(df_col.values.reshape(-1, 1))
                encoded_data = encoder.transform(df_col.values.reshape(-1, 1))
                encoded_cols = [f"{col}_ohe_{cat}" for cat in encoder.categories_[0]]
        else:
            # Обычный OneHotEncoder для других категориальных признаков
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(df_col.values.reshape(-1, 1))
            encoded_data = encoder.transform(df_col.values.reshape(-1, 1))
            encoded_cols = [f"{col}_ohe_{cat}" for cat in encoder.categories_[0]]

        # Создаем DataFrame с закодированными данными
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)
        ohe_columns.extend(encoded_cols)

        # Удаляем оригинальный столбец и добавляем закодированные
        df_transformed = df_transformed.drop(col, axis=1)
        df_transformed = pd.concat([df_transformed.reset_index(drop=True),
                                    encoded_df.reset_index(drop=True)], axis=1)

    # Сохраняем список столбцов BERT для последующего анализа мультиколлинеарности
    bert_columns = []

    # Используем кэш для эмбеддингов - уникальные тексты получают эмбеддинги только один раз
    text_embedding_cache = {}

    # Получаем эмбеддинги для каждого текстового столбца
    for col in text_columns:
        # Собираем все уникальные тексты в столбце
        unique_texts = df[col].dropna().unique()

        # Создаем словарь текст -> эмбеддинг для всех уникальных текстов
        unique_texts_list = unique_texts.tolist()
        uncached_texts = [text for text in unique_texts_list if text not in text_embedding_cache]

        # Получаем эмбеддинги для текстов, которых еще нет в кэше
        if uncached_texts:
            new_embeddings = get_bert_embeddings(uncached_texts, model, tokenizer)
            for i, text in enumerate(uncached_texts):
                text_embedding_cache[text] = new_embeddings[i]

        # Создаем матрицу эмбеддингов для всех строк в столбце
        embeddings = np.zeros((len(df), 768))  # 768 - размерность BERT эмбеддингов
        for i, text in enumerate(df[col]):
            if pd.isna(text):
                text = ""
            embeddings[i] = text_embedding_cache.get(text, np.zeros(768))

        # Создаем столбцы для эмбеддингов
        embedding_cols = [f"{col}_bert_{i}" for i in range(embeddings.shape[1])]
        bert_columns.extend(embedding_cols)
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)

        # Удаляем оригинальный столбец и добавляем эмбеддинги
        df_transformed = df_transformed.drop(col, axis=1)
        df_transformed = pd.concat([df_transformed.reset_index(drop=True),
                                    embedding_df.reset_index(drop=True)], axis=1)

    return df_transformed.drop(columns=["length", "is_anglicism"])


def evaluate_anglicism(clear_table, model_ML):
    """
    Оценивает вероятность того, что слово является англицизмом

    Args:
        clear_table: DataFrame с обработанными данными
        model_ML: Модель XGBoost для предсказания

    Returns:
        Список англицизмов с вероятностями
    """
    try:
        # Получаем предсказания модели
        predictions = model_ML.predict_proba(clear_table)

        # Второй столбец - вероятность того, что слово является англицизмом (класс 1)
        anglicism_probabilities = predictions[:, 1]

        # Формируем результаты: список англицизмов с вероятностями
        # Будем считать англицизмом слово с вероятностью > 0.5
        anglicisms = []

        # Предполагаем, что в DataFrame был индекс, соответствующий исходным словам
        # и у нас есть доступ к исходным словам перед преобразованием
        original_words = clear_table.index.tolist()

        for i, prob in enumerate(anglicism_probabilities):
            if prob > 0.5:  # Порог можно настраивать
                anglicisms.append({
                    "word_index": i,
                    "probability": float(prob)
                })

        return anglicisms
    except Exception as e:
        print(f"Ошибка при оценке англицизмов: {e}")
        return []


def main():
    input_file = "assets/anglicisms_dataset_test.csv"
    output_file = "assets/anglicisms_dataset_ML_test.csv"

    # Загрузка модели и токенизатора BERT
    print("Загрузка модели BERT...")
    model_name = "DeepPavlov/rubert-base-cased"  # Модель BERT для русского языка
    tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
    model_bert = AutoModel.from_pretrained(model_name)

    print("Загрузка модели ML...")
    # Загрузка модели XGBoost из файла
    with open("assets/ML_model.pkl", "rb") as f:
        model_ML = pickle.load(f)

    # Инициализация компонентов Natasha один раз
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    df = pd.read_csv(input_file)  # Загрузка текстов (в виде предложений)
    sentences = df[df.columns[0]]
    results = []
    for sentence in tqdm(sentences):
        table = detect_anglicism(sentence, segmenter, morph_vocab, morph_tagger)
        # Сохраняем оригинальные слова перед трансформацией
        original_words = table['word'].tolist()
        # Создаем словарь для сопоставления индексов с оригинальными словами
        word_index_map = {i: word for i, word in enumerate(original_words)}

        clear_table = modify_table(table, tokenizer_bert, model_bert)
        anglicisms_indices = evaluate_anglicism(clear_table, model_ML)

        # Преобразуем индексы в слова для финального результата
        anglicisms_words = []
        for anglicism in anglicisms_indices:
            word_idx = anglicism["word_index"]
            if word_idx < len(original_words):
                anglicisms_words.append({
                    "word": original_words[word_idx],
                    "probability": anglicism["probability"]
                })

        results.append({'sentence': sentence, 'anglicisms': anglicisms_words})
        print({'sentence': sentence, 'anglicisms': anglicisms_words})

    output_df = pd.DataFrame(results)
    output_df['anglicisms'] = output_df['anglicisms'].apply(json.dumps, ensure_ascii=False)
    output_df.to_csv(output_file, index=False, encoding='utf-8', header=True)


if __name__ == "__main__":
    main()
import os
import csv
import json
import re
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from natasha import Segmenter, MorphVocab, Doc, NewsEmbedding, NewsMorphTagger


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(file_path):
    """Load the dataset from CSV file."""
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                # Очищаем текст от лишних кавычек в начале и конце, если они есть
                text = row[0]
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]

                try:
                    # Обрабатываем список англицизмов
                    anglicisms_str = row[1]
                    anglicisms = json.loads(anglicisms_str)

                    # Проверяем, что это действительно список
                    if isinstance(anglicisms, list):
                        dataset.append((text, anglicisms))
                    else:
                        print(f"Warning: Anglicisms not in list format: {anglicisms_str}")
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse anglicisms in row: {row}")

    print(f"Successfully loaded {len(dataset)} rows from dataset")
    return dataset


def load_anglicisms_set(file_path, segmenter, morph_vocab, morph_tagger):
    """Загрузка англицизмов из файла и преобразование их в множество лемм."""
    anglicisms_lemmas = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                anglicism = line.strip().lower()
                # Лемматизируем каждый англицизм и добавляем в множество
                anglicism_lemma = lemmatize_word(anglicism, segmenter, morph_vocab, morph_tagger)
                anglicisms_lemmas.add(anglicism_lemma)

        print(f"Loaded {len(anglicisms_lemmas)} anglicisms lemmas from {file_path}")
    except Exception as e:
        print(f"Error loading anglicisms file: {e}")

    return anglicisms_lemmas


def is_anglicism(word, anglicisms_set, segmenter, morph_vocab, morph_tagger):
    """Проверяет, является ли слово англицизмом, учитывая исходную форму и части слов через дефис."""
    word = word.lower().strip()

    # Проверка исходной формы слова
    if word in anglicisms_set:
        return True

    # Проверка лемматизированной формы слова
    word_lemma = lemmatize_word(word, segmenter, morph_vocab, morph_tagger)
    if word_lemma in anglicisms_set:
        return True

    # Если в слове есть дефис, проверяем отдельные части
    if '-' in word:
        parts = word.split('-')
        for part in parts:
            if part.strip() in anglicisms_set:
                return True

            # Проверяем лемматизированные части
            part_lemma = lemmatize_word(part.strip(), segmenter, morph_vocab, morph_tagger)
            if part_lemma in anglicisms_set:
                return True

    return False


def lemmatize_word(word, segmenter, morph_vocab, morph_tagger):
    """Лемматизирует слово с помощью natasha."""
    doc = Doc(word)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    # Возвращаем лемму первого токена (для одного слова будет только один токен)
    if doc.tokens:
        return doc.tokens[0].lemma
    return word  # Если не удалось лемматизировать, возвращаем исходное слово


def generate_synonyms(anglicism, model, tokenizer, device, segmenter, morph_vocab, morph_tagger, anglicisms_set,
                      previous_synonyms=None, num_synonyms=7):
    """Этап 1: Генерация семи базовых русских синонимов для англицизма с проверками."""

    # Приводим англицизм к начальной форме
    anglicism_lemma = lemmatize_word(anglicism.lower(), segmenter, morph_vocab, morph_tagger)

    # Формируем промпт в зависимости от того, есть ли у нас уже отклоненные синонимы
    system_prompt = f"""Ты эксперт по русскому языку. Твоя задача - предложить {num_synonyms} лучших русских эквивалента для замены указанного англицизма. 

Важно: верни только {num_synonyms} слов или коротких фраз в начальной форме, каждое на новой строке, без нумерации и без дополнительных пояснений. 
ВАЖНО: СЛОВА НЕ ДОЛЖНЫ ПОВТОРЯТЬСЯ.
ВАЖНО: Слова не должны иметь единую составную часть, ПРИМЕР КАК НЕ НАДО ДЕЛАТЬ: (Электронная связь и Электронный мир).
ИНОГДА СЛОВА ИМЕЮТ ПЕРЕД НА ДРУГОЙ ЯЗЫК, НАПРИМЕР ДЛЯ 'блэкаут' подходит 'выключение света'.
ПИСАТЬ НУЖНО СТРОГО НА РУССКОМ ЯЗЫКЕ. НУЖНО ПРЕДЛАГАТЬ СТРОГО ТОЛЬКО ИСКОННО РУССКИЕ СЛОВА.
САМОЕ ВАЖНОЕ: ВЕРНИ ТОЛЬКО {num_synonyms} СЛОВ ИЛИ КОРОТКИХ ФРАЗ В НАЧАЛЬНОЙ ФОРМЕ, КАЖДОЕ НА НОВОЙ СТРОКЕ, БЕЗ НУМЕРАЦИИ И БЕЗ ДОПОЛНИТЕЛЬНЫХ ПОЯСНЕНИЙ."""

    user_prompt = f"Предложи {num_synonyms} русских эквивалента для англицизма: '{anglicism}'. НИ В КОЕМ СЛУЧАЕ НЕ ПИШИ ПОХОЖИЕ НА '{anglicism}' СЛОВА. НЕ НУЖНО ЗАМЕНЯТЬ СЛОВО 'БЛОГ' НА 'БЛОГЕР', 'ВЛОГЕР' ИЛИ 'БЛОГЕРША'."

    # Если у нас есть предыдущие синонимы, которые нужно исключить
    if previous_synonyms:
        user_prompt = f"Предложи {num_synonyms} русских эквивалента для англицизма: '{anglicism}'. Не предлагай следующие варианты, так как они не подходят: {', '.join(previous_synonyms)}"

    # Формирование сообщений согласно формату
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Применение шаблона чата
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Токенизация входных данных
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # Генерация ответа
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,  # Увеличиваем, так как нам нужно больше синонимов
            temperature=0.9,
            top_p=0.9,
            do_sample=True
        )

    # Выделение только сгенерированной части
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Декодирование ответа
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Обработка ответа - разделение на отдельные синонимы
    raw_synonyms = [line.strip().lower() for line in response.split('\n') if line.strip()]  # Приводим к нижнему регистру здесь
    clean_synonyms = []

    for syn in raw_synonyms:
        # Удаляем цифры и знаки пунктуации в начале строки
        cleaned = syn.lstrip('0123456789. -)')
        cleaned = cleaned.strip()
        if cleaned:
            clean_synonyms.append(cleaned)

    # Проверка каждого синонима на соответствие требованиям
    valid_synonyms = []
    invalid_synonyms = []

    for synonym in clean_synonyms:
        # Проверяем синоним новой функцией is_anglicism
        if is_anglicism(synonym, anglicisms_set, segmenter, morph_vocab, morph_tagger):
            invalid_synonyms.append(synonym)
            print(f"Отклонён синоним '{synonym}' для '{anglicism}': является англицизмом")
        # Проверяем, не является ли синоним тем же самым словом
        elif lemmatize_word(synonym.lower(), segmenter, morph_vocab, morph_tagger) == anglicism_lemma or synonym.lower() == anglicism.lower():
            invalid_synonyms.append(synonym)
            print(f"Отклонён синоним '{synonym}' для '{anglicism}': совпадает с англицизмом")
        else:
            valid_synonyms.append(synonym)

    # Если у нас достаточно валидных синонимов
    if len(valid_synonyms) >= 5:
        return valid_synonyms

    # Если у нас недостаточно валидных синонимов и мы еще не делали рекурсивный вызов
    if previous_synonyms is None or len(previous_synonyms) < 5:  # Ограничиваем глубину рекурсии
        # Объединяем уже отклоненные синонимы с новыми отклоненными
        all_rejected = (previous_synonyms or []) + invalid_synonyms

        # Рекурсивно вызываем функцию еще раз с указанием отклоненных вариантов
        print(f"Недостаточно валидных синонимов для '{anglicism}', генерирую новые...")
        new_synonyms = generate_synonyms(anglicism, model, tokenizer, device, segmenter, morph_vocab, morph_tagger,
                                         anglicisms_set, all_rejected, num_synonyms)

        # Если у нас есть несколько валидных синонимов и несколько новых, объединяем их
        if valid_synonyms:
            combined = valid_synonyms + new_synonyms
            return combined[:num_synonyms]

        return new_synonyms

    # Если после нескольких попыток все равно не хватает синонимов, дополняем список существующими
    print(f"После нескольких попыток не удалось получить достаточно валидных синонимов для '{anglicism}'")
    while len(valid_synonyms) < num_synonyms:
        if valid_synonyms:
            valid_synonyms.append(valid_synonyms[0])
        else:
            # В крайнем случае, если вообще нет валидных синонимов, используем сам англицизм
            print(f"Не удалось найти ни одного валидного синонима для '{anglicism}', используем сам англицизм")
            valid_synonyms.append(anglicism)

    return valid_synonyms[:num_synonyms]


def simple_replace_in_text(text, anglicism, synonym):
    """Простая замена англицизма на синоним в тексте без учета грамматики."""
    # Находим все вхождения англицизма в тексте (с учетом регистра)
    pattern = re.compile(re.escape(anglicism), re.IGNORECASE)

    # Заменяем на синоним с сохранением регистра первой буквы
    def replace_match(match):
        matched = match.group(0)
        if matched[0].isupper():
            return synonym[0].upper() + synonym[1:]
        return synonym

    return pattern.sub(replace_match, text)


def generate_combinations_and_replace(text, anglicisms, synonyms_map):
    """Генерирует все возможные комбинации замен и возвращает топ-3."""
    # Создаем список всех возможных комбинаций синонимов
    synonyms_lists = [synonyms_map[anglicism] for anglicism in anglicisms]
    combinations = list(itertools.product(*synonyms_lists))

    replaced_texts = []

    for combination in combinations:
        current_text = text
        combo_details = {}

        # Применяем каждую замену в комбинации
        for i, anglicism in enumerate(anglicisms):
            synonym = combination[i]
            current_text = simple_replace_in_text(current_text, anglicism, synonym)
            combo_details[anglicism] = synonym

        replaced_texts.append((current_text, combo_details))

    return replaced_texts


def check_text_for_anglicisms(text, segmenter, morph_vocab, morph_tagger, anglicisms_set, exceptions_lemmas=None,
                              stopwords_lemmas=None):
    """Проверяет текст на наличие англицизмов и возвращает список найденных англицизмов.
    Реализация основана на функции find_anglicisms из второго файла."""
    found_anglicisms = []

    # Если множества не переданы, создаем пустые
    if exceptions_lemmas is None:
        exceptions_lemmas = set()
    if stopwords_lemmas is None:
        stopwords_lemmas = set()

    if pd.isna(text) or not isinstance(text, str):
        return []

    # Извлекаем слова, находящиеся в кавычках
    quoted_words = extract_quoted_text(text)
    # Создаем множество лемм слов в кавычках для быстрой проверки
    quoted_lemmas = set(lemmatize_word(word.lower(), segmenter, morph_vocab, morph_tagger) for word in quoted_words)

    # Разбиваем текст на слова, включая слова через дефис
    # \b - граница слова, далее русские буквы, затем опционально дефис и снова русские буквы
    word_matches = re.finditer(r'\b[а-яА-ЯёЁ]+(?:-[а-яА-ЯёЁ]+)*\b', text)

    word_lemmas = {}  # Кэш для лемматизированных слов

    for match in word_matches:
        try:
            word = match.group(0)
            word_pos = match.start()

            # Пропускаем слова короче 4 символов для избежания ложных срабатываний
            if len(word) <= 3:
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
                word_lemmas[word_lower] = lemmatize_word(word_lower, segmenter, morph_vocab, morph_tagger)
            word_lemma = word_lemmas[word_lower]

            # Проверяем, что:
            # 1. Лемма слова есть в списке лемм англицизмов
            # 2. Лемма слова НЕ находится в списке стоп-слов
            # 3. Лемма слова НЕ находится в списке исключений
            # 4. Слово НЕ находится внутри кавычек (не является частью имени собственного)
            if (word_lemma in anglicisms_set and
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


def transform_sentence_with_synonym(anglicism, replaced_text, model, tokenizer, device):
    """Этап 2: Трансформация предложения с учетом грамматики на основе примера замены."""

    system_prompt = f"""Ты эксперт по русскому языку. В предложении заменили неуместные слова, но окончания слов могут быть выбраны неправильно.
    Нужно минимально менять предложение, вот пример:
    ДО: "Отметил мэр Игорь Терехов прямой эфир украинского телеканала LIGA."
    ТВОЙ ОТВЕТ: "Отметил мэр Игорь Терехов в прямом эфире украинскому телеканалу LIGA."
    ВАЖНО: Верни только изменённое предложение, без дополнительных объяснений.
    НИ В КОЕМ СЛУЧАЕ НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ СЛОВО "{anglicism}" И ЕГО ПРОИЗВОДНЫЕ.
    Не забывай, что иногда приходится менять соседние слова, чтобы предложение было согласованным."""
    user_prompt = f""" Вот предложение без учета правильных грамматических форм.:
{replaced_text}. НИ В КОЕМ СЛУЧАЕ НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ СЛОВО "{anglicism}" или его производные."""

    # Формирование сообщений согласно формату
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Применение шаблона чата
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Токенизация входных данных
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # Генерация ответа
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,  # Увеличиваем длину, так как нам нужно целое предложение
            temperature=0.3,  # Уменьшаем температуру для более точного соответствия грамматике
            top_p=0.95,
            do_sample=True
        )
    # Выделение только сгенерированной части
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Декодирование ответа
    transformed_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"ОТЛАДКА. ПРОМПТ: {messages}")
    print(f"ОТЛАДКА. Ответ: {transformed_text}")

    return transformed_text


def calculate_semantic_similarity(original_text, replaced_text, semantic_model):
    """Calculate semantic similarity between two texts."""
    # Вычисляем эмбеддинги для обоих текстов
    embedding1 = semantic_model.encode(original_text, convert_to_tensor=True)
    embedding2 = semantic_model.encode(replaced_text, convert_to_tensor=True)

    # Вычисляем косинусное сходство
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    return similarity


def replace_anglicisms(text, anglicisms, model, tokenizer, semantic_model, device, segmenter, morph_vocab, morph_tagger,
                       anglicisms_set, exceptions_lemmas=None, stopwords_lemmas=None):
    """Replace anglicisms in the text using the two-stage approach."""
    original_text = text
    replacement_details = {}

    # Добавляем специальные токены в словарь токенизатора
    special_tokens = {"additional_special_tokens": ["<anglicism>", "</anglicism>", "<synonym>", "</synonym>"]}
    tokenizer.add_special_tokens(special_tokens)

    # Изменяем размер эмбеддингов модели для новых токенов
    model.resize_token_embeddings(len(tokenizer))

    # Step 1: Generate synonyms for each anglicism
    synonyms_map = {}
    for anglicism in anglicisms:
        synonyms = generate_synonyms(anglicism, model, tokenizer, device, segmenter, morph_vocab, morph_tagger,
                                     anglicisms_set, num_synonyms=10)
        print(f"Сгенерированные синонимы для '{anglicism}': {synonyms}")
        synonyms_map[anglicism] = synonyms

    # Step 2: Generate all possible combinations of replacements
    all_replacements = generate_combinations_and_replace(text, anglicisms, synonyms_map)

    # Step 3: Calculate semantic similarity for each raw replacement
    replacement_similarities = []
    for replaced_text, combo_details in all_replacements:
        similarity = calculate_semantic_similarity(original_text, replaced_text, semantic_model)
        replacement_similarities.append((replaced_text, combo_details, similarity))

    # Step 4: Sort by similarity and take top
    replacement_similarities.sort(key=lambda x: x[2], reverse=True)
    top_replacements = replacement_similarities[:3] # КОЛИЧЕСТВО ПРЕДЛОЖЕНИЙ

    print(f"Selected top replacements with similarities: {[sim for _, _, sim in top_replacements]}")

    # Step 5: Apply grammatical transformation to each of the top
    transformed_texts = []
    for replaced_text, combo_details, similarity in top_replacements:
        # Transform with proper grammar
        transformed_text = transform_sentence_with_synonym(anglicism, replaced_text, model, tokenizer, device)

        # Проверяем трансформированный текст на наличие англицизмов
        found_anglicisms = check_text_for_anglicisms(transformed_text, segmenter, morph_vocab, morph_tagger,
                                                     anglicisms_set, exceptions_lemmas, stopwords_lemmas)

        if found_anglicisms:
            print(f"Отклоняем трансформированный текст: {transformed_text}")
            print(f"Найденные англицизмы: {found_anglicisms}")
            continue  # Пропускаем этот вариант, если найдены англицизмы

        final_similarity = calculate_semantic_similarity(original_text, transformed_text, semantic_model)
        transformed_texts.append((transformed_text, combo_details, final_similarity))
        print(f"Трансформированное предложение: {transformed_text}")
        print(f"Финальное семантическое сходство: {final_similarity:.4f}")

    # Step 6: Choose the best transformed text
    if not transformed_texts:
        # Если все варианты были отклонены из-за наличия англицизмов
        print(f"Все трансформированные варианты содержат англицизмы. Пример будет пропущен.")
        return None, None  # Возвращаем None, чтобы пропустить этот пример

    transformed_texts.sort(key=lambda x: x[2], reverse=True)
    best_transformed = transformed_texts[0]

    # Сохраняем детали замены
    for anglicism in anglicisms:
        replacement_details[anglicism] = {
            "chosen_synonym": best_transformed[1][anglicism],
            "all_synonyms": synonyms_map[anglicism],
            "similarity": best_transformed[2]
        }

    # Возвращаем финальный текст с заменами и детали замен
    return best_transformed[0], replacement_details


def process_dataset(dataset, model, tokenizer, semantic_model, device, segmenter, morph_vocab, morph_tagger,
                    anglicisms_set, exceptions_lemmas=None, stopwords_lemmas=None):
    """Process the entire dataset."""
    processed_data = []

    # Инициализация tqdm с общим количеством элементов
    progress_bar = tqdm(
        total=len(dataset),
        desc="Processing dataset",
        unit="example",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for i, (text, anglicisms) in enumerate(dataset):
        try:
            # Замена англицизмов с выбором наилучшего варианта
            replaced_text, replacement_details = replace_anglicisms(
                text, anglicisms, model, tokenizer, semantic_model, device, segmenter, morph_vocab, morph_tagger,
                anglicisms_set, exceptions_lemmas, stopwords_lemmas
            )

            # Проверяем, был ли пример отклонен (если все варианты содержали англицизмы)
            if replaced_text is None or replacement_details is None:
                print(f"Пример {i + 1} был пропущен из-за англицизмов во всех вариантах")
                # НЕ добавляем этот пример в processed_data
                progress_bar.update(1)
                continue

            # Сохраняем результаты
            processed_data.append((
                text,
                json.dumps(anglicisms, ensure_ascii=False),
                replaced_text,
                json.dumps(replacement_details, ensure_ascii=False)
            ))

        except Exception as e:
            print(f"Error processing item {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            # Не добавляем неудачные примеры в итоговый файл

        # Обновляем прогресс-бар
        progress_bar.update(1)

    # Закрываем прогресс-бар
    progress_bar.close()
    print(f"Processing completed for {len(processed_data)} items.")
    return processed_data


def save_dataset(data, output_path):
    """Save the processed dataset to CSV file."""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        # Записываем заголовок
        writer.writerow(["Original Text", "Anglicisms", "Replaced Text", "Replacement Details"])

        # Записываем данные
        for row in data:
            writer.writerow(row)

    print(f"Dataset saved to {output_path} with {len(data)} rows")

    # Проверка сохраненного файла
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size} bytes")

        # Проверка содержимого файла
        with open(output_path, 'r', encoding='utf-8') as f:
            first_lines = [next(f) for _ in range(min(5, len(data) + 1))]

        print("First few lines of the saved file:")
        for line in first_lines:
            print(line.strip())


def main():
    # Paths
    input_path = "assets/anglicisms_dataset.csv"
    output_path = "assets/y_dataset_with_synonyms.csv"
    anglicisms_file = "assets/clean_anglicism_2.txt"  # Путь к файлу с англицизмами

    # Ensure directories exist
    ensure_dir('assets')

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Инициализация компонентов Natasha для лемматизации
    print("Initializing Natasha components for lemmatization...")
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    # Загрузка списка англицизмов
    print(f"Loading anglicisms from {anglicisms_file}...")
    anglicisms_set = load_anglicisms_set(anglicisms_file, segmenter, morph_vocab, morph_tagger)

    # Load models and tokenizer
    print("Loading models and tokenizer...")

    # Загрузка основной модели для генерации замен
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    # Загрузка модели для оценки семантического сходства
    print("Loading semantic model...")
    semantic_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    semantic_model = SentenceTransformer(semantic_model_name, device=device)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(input_path)
    print(f"Loaded {len(dataset)} examples.")

    # Process dataset
    print("Processing dataset...")
    processed_data = process_dataset(dataset, model, tokenizer, semantic_model, device, segmenter, morph_vocab,
                                     morph_tagger, anglicisms_set)

    # Save results
    print("Saving results...")
    save_dataset(processed_data, output_path)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
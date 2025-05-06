import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# Пути к файлам
input_csv = "assets/dataset_balanced.csv"
output_csv = "assets/dataset_transformed.csv"

# Определение текстовых и категориальных столбцов
text_columns = ['word', 'lemma', 'left_left', 'left', 'right', 'right_right']


# Функция для получения эмбеддингов BERT
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
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT embedding"):
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


def main():
    print(f"Чтение датасета из {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Загружены данные: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # Определение категориальных столбцов (все object столбцы, кроме текстовых)
    categorical_columns = [col for col in df.select_dtypes(include=['object']).columns
                           if col not in text_columns]

    print(f"Текстовые столбцы: {text_columns}")
    print(f"Категориальные столбцы: {categorical_columns}")

    # Обработка категориальных данных с помощью OneHotEncoder
    print("Применение One-Hot Encoding к категориальным данным...")
    # Создаем копию DataFrame для результатов
    df_transformed = df.copy()

    # Применяем OneHotEncoder к каждому категориальному столбцу
    for col in categorical_columns:
        # Обрабатываем пропущенные значения как отдельную категорию
        # Заменяем NaN на специальную строку "_NAN_" для обозначения пропущенных значений
        df_col = df[col].fillna("_NAN_")

        # Создаем и обучаем encoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df_col.values.reshape(-1, 1))

        # Создаем новые столбцы с закодированными данными
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=[f"{col}_ohe_{cat}" for cat in encoder.categories_[0]]
        )

        # Удаляем оригинальный столбец и добавляем закодированные
        df_transformed = df_transformed.drop(col, axis=1)
        df_transformed = pd.concat([df_transformed.reset_index(drop=True),
                                    encoded_df.reset_index(drop=True)], axis=1)

    # Загрузка модели и токенизатора BERT
    print("Загрузка модели BERT...")
    model_name = "DeepPavlov/rubert-base-cased"  # Модель BERT для русского языка
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Обработка текстовых данных с помощью BERT
    print("Получение эмбеддингов BERT для текстовых данных...")

    # Получаем эмбеддинги для каждого текстового столбца
    for col in text_columns:
        print(f"Обрабатываем столбец: {col}")

        # Получаем эмбеддинги
        embeddings = get_bert_embeddings(df[col].values.tolist(), model, tokenizer)

        # Создаем столбцы для эмбеддингов
        embedding_cols = [f"{col}_bert_{i}" for i in range(embeddings.shape[1])]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)

        # Удаляем оригинальный столбец и добавляем эмбеддинги
        df_transformed = df_transformed.drop(col, axis=1)
        df_transformed = pd.concat([df_transformed.reset_index(drop=True),
                                    embedding_df.reset_index(drop=True)], axis=1)

    # Сохранение результатов
    print(f"Сохранение трансформированного датасета в {output_csv}")
    print(f"Финальный размер: {df_transformed.shape[0]} строк, {df_transformed.shape[1]} столбцов")
    df_transformed.to_csv(output_csv, index=False)
    print("Готово!")


if __name__ == "__main__":
    main()
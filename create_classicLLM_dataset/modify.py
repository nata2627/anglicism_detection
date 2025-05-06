import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Пути к файлам
input_csv = "assets/dataset_balanced.csv"
output_csv = "assets/dataset_transformed.csv"
final_features_csv = "assets/final_features.csv"

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


def calculate_correlation_matrix(df, target_col='is_anglicism'):
    """
    Вычисляет матрицу корреляции между признаками и целевой переменной

    Args:
        df: DataFrame с данными
        target_col: имя столбца с целевой переменной

    Returns:
        DataFrame с матрицей корреляции
    """
    correlation_matrix = df.corr(method='pearson')

    # Сохраняем корреляцию с целевой переменной, если она присутствует
    target_correlation = None
    if target_col in correlation_matrix:
        target_correlation = correlation_matrix[target_col].sort_values(ascending=False)

    return correlation_matrix, target_correlation


def calculate_vif(df, exclude_columns=None):
    """
    Вычисляет VIF (Variance Inflation Factor) для признаков

    Args:
        df: DataFrame с данными
        exclude_columns: список столбцов, которые следует исключить из расчета

    Returns:
        DataFrame с значениями VIF для каждого признака
    """
    if exclude_columns is None:
        exclude_columns = []

    # Отбираем только числовые столбцы, не в списке исключений
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    numerical_df = numerical_df.drop(columns=[col for col in exclude_columns if col in numerical_df.columns])

    # Проверяем, что остались столбцы для вычисления VIF
    if numerical_df.shape[1] == 0:
        return pd.DataFrame(columns=['feature', 'VIF'])

    # Проверка на отсутствие NaN значений
    if numerical_df.isna().any().any():
        print("Предупреждение: обнаружены пропущенные значения, заполняем их нулями")
        numerical_df = numerical_df.fillna(0)

    # Вычисляем VIF для каждого признака
    vif_data = []
    for i, feature in enumerate(numerical_df.columns):
        try:
            vif = variance_inflation_factor(numerical_df.values, i)
            vif_data.append({'feature': feature, 'VIF': vif})
        except Exception as e:
            print(f"Ошибка при вычислении VIF для признака {feature}: {e}")
            vif_data.append({'feature': feature, 'VIF': np.nan})

    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


def plot_correlation_heatmap(correlation_matrix, size=(10, 8)):
    """
    Строит тепловую карту корреляции

    Args:
        correlation_matrix: матрица корреляции
        size: размеры изображения (ширина, высота)
    """
    plt.figure(figsize=size)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Матрица корреляции признаков')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()


def main():
    print(f"Чтение датасета из {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Загружены данные: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # Определяем выборку из 1000 строк для анализа мультиколлинеарности
    sample_size = min(1000, df.shape[0])
    print(f"Используем выборку из {sample_size} строк для анализа мультиколлинеарности")
    df_sample = df.sample(n=sample_size, random_state=42)

    # Определение категориальных столбцов (все object столбцы, кроме текстовых)
    categorical_columns = [col for col in df.select_dtypes(include=['object']).columns
                           if col not in text_columns]

    print(f"Текстовые столбцы: {text_columns}")
    print(f"Категориальные столбцы: {categorical_columns}")

    # Обработка категориальных данных с помощью OneHotEncoder
    print("Применение One-Hot Encoding к категориальным данным...")
    # Создаем копию DataFrame для результатов
    df_transformed = df.copy()
    df_sample_transformed = df_sample.copy()

    # Сохраняем список столбцов OHE для исключения из проверки мультиколлинеарности
    ohe_columns = []

    # Применяем OneHotEncoder к каждому категориальному столбцу
    for col in categorical_columns:
        # Обрабатываем пропущенные значения как отдельную категорию
        # Заменяем NaN на специальную строку "_NAN_" для обозначения пропущенных значений
        df_col = df[col].fillna("_NAN_")
        df_sample_col = df_sample[col].fillna("_NAN_")

        # Создаем и обучаем encoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(df_col.values.reshape(-1, 1))

        # Применяем encoder к полному датасету
        encoded_data = encoder.transform(df_col.values.reshape(-1, 1))
        encoded_cols = [f"{col}_ohe_{cat}" for cat in encoder.categories_[0]]
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)
        ohe_columns.extend(encoded_cols)

        # Применяем encoder к выборке
        encoded_sample_data = encoder.transform(df_sample_col.values.reshape(-1, 1))
        encoded_sample_df = pd.DataFrame(encoded_sample_data, columns=encoded_cols)

        # Удаляем оригинальный столбец и добавляем закодированные
        df_transformed = df_transformed.drop(col, axis=1)
        df_transformed = pd.concat([df_transformed.reset_index(drop=True),
                                    encoded_df.reset_index(drop=True)], axis=1)

        df_sample_transformed = df_sample_transformed.drop(col, axis=1)
        df_sample_transformed = pd.concat([df_sample_transformed.reset_index(drop=True),
                                           encoded_sample_df.reset_index(drop=True)], axis=1)

    # Загрузка модели и токенизатора BERT
    print("Загрузка модели BERT...")
    model_name = "DeepPavlov/rubert-base-cased"  # Модель BERT для русского языка
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Обработка текстовых данных с помощью BERT
    print("Получение эмбеддингов BERT для текстовых данных...")

    # Сохраняем список столбцов BERT для последующего анализа мультиколлинеарности
    bert_columns = []

    # Получаем эмбеддинги для каждого текстового столбца
    for col in text_columns:
        print(f"Обрабатываем столбец: {col}")

        # Получаем эмбеддинги только для полного датасета
        embeddings = get_bert_embeddings(df[col].values.tolist(), model, tokenizer)
        embedding_cols = [f"{col}_bert_{i}" for i in range(embeddings.shape[1])]
        bert_columns.extend(embedding_cols)
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)

        # Удаляем оригинальный столбец и добавляем эмбеддинги в полный датасет
        df_transformed = df_transformed.drop(col, axis=1)
        df_transformed = pd.concat([df_transformed.reset_index(drop=True),
                                    embedding_df.reset_index(drop=True)], axis=1)

        # Для выборки берем соответствующие строки из уже вычисленных эмбеддингов
        # Находим индексы строк из выборки в полном датасете
        sample_indices = df_sample.index
        sample_embedding_df = embedding_df.iloc[sample_indices].reset_index(drop=True)

        # Удаляем оригинальный столбец и добавляем эмбеддинги в выборку
        df_sample_transformed = df_sample_transformed.drop(col, axis=1)
        df_sample_transformed = pd.concat([df_sample_transformed.reset_index(drop=True),
                                           sample_embedding_df.reset_index(drop=True)], axis=1)

    # Проверка мультиколлинеарности
    print("\n" + "=" * 50)
    print("АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ")
    print("=" * 50)

    # Выделяем признаки, для которых нужно проверить мультиколлинеарность (не OHE)
    non_ohe_columns = [col for col in df_sample_transformed.columns if col not in ohe_columns]

    # Проверка, является ли 'is_anglicism' целевой переменной
    target_col = 'is_anglicism'
    if target_col in df_sample_transformed.columns:
        non_ohe_columns = [col for col in non_ohe_columns if col != target_col]
        print(f"\nЦелевая переменная: {target_col}")
    else:
        print("Предупреждение: столбец is_anglicism не найден в данных")
        target_col = None

    print(f"\nАнализ мультиколлинеарности для {len(non_ohe_columns)} признаков (без OHE-признаков)")

    # Рассчитываем матрицу корреляции для выборки
    sample_numeric_df = df_sample_transformed[non_ohe_columns].select_dtypes(include=['float64', 'int64'])

    # Добавляем целевую переменную, если она числовая
    if target_col and target_col in df_sample_transformed.columns:
        if pd.api.types.is_numeric_dtype(df_sample_transformed[target_col]):
            sample_numeric_df = pd.concat([sample_numeric_df, df_sample_transformed[[target_col]]], axis=1)

    print(f"Размерность числовой выборки: {sample_numeric_df.shape}")

    # Вычисляем матрицу корреляции
    try:
        correlation_matrix, target_correlation = calculate_correlation_matrix(sample_numeric_df, target_col)
        print("\nМатрица корреляции успешно вычислена")

        # Строим тепловую карту корреляции
        plot_correlation_heatmap(correlation_matrix)
        print("Тепловая карта корреляции сохранена в файл correlation_heatmap.png")

        # Находим сильно коррелирующие признаки (|r| > 0.8)
        high_correlation = (correlation_matrix.abs() > 0.8) & (correlation_matrix.abs() < 1.0)
        pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if high_correlation.iloc[i, j]:
                    pairs.append(
                        (correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

        if pairs:
            print("\nОбнаружены сильно коррелирующие признаки (|r| > 0.8):")
            for pair in pairs:
                print(f"  {pair[0]} и {pair[1]}: r = {pair[2]:.4f}")
        else:
            print("\nНе обнаружено сильно коррелирующих признаков (|r| > 0.8)")

        # Выводим топ-10 признаков по корреляции с целевой переменной
        if target_correlation is not None:
            print(f"\nТоп-10 признаков по корреляции с {target_col}:")
            for feature, corr in target_correlation[:10].items():
                print(f"  {feature}: {corr:.4f}")

    except Exception as e:
        print(f"Ошибка при вычислении матрицы корреляции: {e}")

    # Вычисляем VIF (Variance Inflation Factor)
    try:
        # Исключаем столбцы с нулевой дисперсией (константы)
        constant_cols = [col for col in sample_numeric_df.columns
                         if col != target_col and sample_numeric_df[col].std() == 0]
        if constant_cols:
            print(f"\nИсключено {len(constant_cols)} столбцов с нулевой дисперсией")
            sample_numeric_df = sample_numeric_df.drop(columns=constant_cols)

        # Исключаем целевую переменную из расчета VIF
        vif_exclude = [target_col] if target_col in sample_numeric_df.columns else []
        vif_data = calculate_vif(sample_numeric_df, vif_exclude)

        if not vif_data.empty:
            print("\nЗначения VIF (Variance Inflation Factor):")
            print("  VIF > 10 указывает на возможную мультиколлинеарность")
            print("  VIF > 30 указывает на серьезную мультиколлинеарность")

            # Выводим топ-20 признаков с высоким VIF
            top_vif = vif_data.head(20)
            for _, row in top_vif.iterrows():
                print(f"  {row['feature']}: {row['VIF']:.2f}")

            # Выявляем признаки с высоким VIF
            high_vif_features = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
            if high_vif_features:
                print(f"\nОбнаружено {len(high_vif_features)} признаков с VIF > 10")
            else:
                print("\nНе обнаружено признаков с высоким VIF")
        else:
            print("\nНе удалось вычислить VIF для данного набора признаков")
    except Exception as e:
        print(f"Ошибка при вычислении VIF: {e}")

    # Формируем список финальных признаков, исключая признаки с высоким VIF
    final_features = []
    try:
        print("\nФормирование списка финальных признаков...")

        # Признаки с высоким VIF (если было вычислено)
        high_vif_features = []
        if 'vif_data' in locals() and not vif_data.empty:
            high_vif_features = vif_data[vif_data['VIF'] > 30]['feature'].tolist()

        # Исключаем признаки с высоким VIF из финального списка
        final_features = [col for col in df_transformed.columns
                          if col != target_col and col not in high_vif_features]

        print(f"Итоговое количество признаков: {len(final_features)}")

        # Сохраняем список финальных признаков
        final_features_df = pd.DataFrame({'feature': final_features})
        final_features_df.to_csv(final_features_csv, index=False)
        print(f"Список финальных признаков сохранен в файл {final_features_csv}")

        # Сохраняем трансформированный датасет только с финальными признаками
        df_final = df_transformed[final_features + ([target_col] if target_col in df_transformed.columns else [])]
        df_final.to_csv(output_csv, index=False)
        print(f"Финальный датасет сохранен в файл {output_csv}")
        print(f"Размер финального датасета: {df_final.shape[0]} строк, {df_final.shape[1]} столбцов")

    except Exception as e:
        print(f"Ошибка при формировании списка финальных признаков: {e}")

        # В случае ошибки сохраняем исходный трансформированный датасет
        df_transformed.to_csv(output_csv, index=False)
        print(
            f"Сохранен исходный трансформированный датасет: {df_transformed.shape[0]} строк, {df_transformed.shape[1]} столбцов")

    print("\nГотово!")


if __name__ == "__main__":
    main()
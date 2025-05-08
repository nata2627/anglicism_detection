import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Пути к файлам
input_csv = "assets/dataset_transformed.csv"
output_dir = "assets/models/logistic_regression"
os.makedirs(output_dir, exist_ok=True)

# Имя модели с временной меткой
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"{output_dir}/logistic_regression_model_{timestamp}.pkl"
metrics_filename = f"{output_dir}/model_metrics_{timestamp}.txt"
plots_dir = f"{output_dir}/plots_{timestamp}"
os.makedirs(plots_dir, exist_ok=True)


def main():
    print("Загрузка трансформированного датасета...")
    try:
        df = pd.read_csv(input_csv)
        print(f"Загружен датасет: {df.shape[0]} строк, {df.shape[1]} столбцов")
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return

    # Проверка наличия целевой переменной
    if 'is_anglicism' not in df.columns:
        print("Ошибка: в датасете отсутствует целевая переменная 'is_anglicism'")
        return

    # Разделение на признаки и целевую переменную
    X = df.drop('is_anglicism', axis=1)
    y = df['is_anglicism']

    print(f"Распределение классов: 0 (не англицизм): {(y == 0).sum()}, 1 (англицизм): {(y == 1).sum()}")

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"Размер обучающей выборки: {X_train.shape[0]} примеров")
    print(f"Размер тестовой выборки: {X_test.shape[0]} примеров")

    # Сохраняем информацию о признаках
    with open(f"{output_dir}/feature_names_{timestamp}.txt", "w") as f:
        for feature in X.columns:
            f.write(f"{feature}\n")

    # Создаем модель логистической регрессии с нормализацией данных
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(
            C=1.0,  # Параметр регуляризации
            max_iter=1000,  # Максимальное число итераций
            random_state=42,
            verbose=1,  # Включаем вывод информации о процессе обучения
            n_jobs=-1  # Используем все доступные ядра процессора
        ))
    ])

    # Обучаем модель с выводом прогресса (verbose=1 включает базовый вывод)
    print("Начинаем обучение логистической регрессии...")
    # Выведем время начала обучения
    start_time = datetime.now()
    print(f"Начало обучения: {start_time.strftime('%H:%M:%S')}")

    # Обучаем модель
    model.fit(X_train, y_train)

    # Выведем время окончания обучения
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Окончание обучения: {end_time.strftime('%H:%M:%S')}")
    print(f"Время обучения: {training_time.total_seconds():.2f} секунд")

    # Получаем коэффициенты модели
    logreg = model.named_steps['logreg']
    coefficients = logreg.coef_[0]
    intercept = logreg.intercept_[0]

    # Создаем DataFrame с коэффициентами для анализа важности признаков
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values(by='Abs_Coefficient', ascending=False)

    # Оценка модели на тестовой выборке
    print("Оцениваем модель на тестовых данных...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Рассчитываем метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"Точность (Precision): {precision:.4f}")
    print(f"Полнота (Recall): {recall:.4f}")
    print(f"F1-мера: {f1:.4f}")

    # Сохраняем отчет о метриках
    with open(metrics_filename, "w") as f:
        f.write(f"Модель Логистическая регрессия для классификации англицизмов\n")
        f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Размер датасета: {df.shape[0]} примеров, {df.shape[1]} признаков\n")
        f.write(f"Обучающая выборка: {X_train.shape[0]} примеров\n")
        f.write(f"Тестовая выборка: {X_test.shape[0]} примеров\n\n")
        f.write(f"Время обучения: {training_time.total_seconds():.2f} секунд\n\n")
        f.write("\nМетрики на тестовой выборке:\n")
        f.write(f"  Точность (Accuracy): {accuracy:.4f}\n")
        f.write(f"  Точность (Precision): {precision:.4f}\n")
        f.write(f"  Полнота (Recall): {recall:.4f}\n")
        f.write(f"  F1-мера: {f1:.4f}\n\n")
        f.write("Отчет по классификации:\n")
        f.write(classification_report(y_test, y_pred))

        # Добавляем топ-20 наиболее важных признаков по абсолютной величине коэффициентов
        f.write("\nТоп-20 важных признаков по абсолютной величине коэффициентов:\n")
        for idx, row in coef_df.head(20).iterrows():
            f.write(f"  {row['Feature']}: {row['Coefficient']:.6f}\n")

    # Визуализация матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Не англицизм', 'Англицизм'],
                yticklabels=['Не англицизм', 'Англицизм'])
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.title('Матрица ошибок')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/confusion_matrix.png")

    # Визуализация важности признаков (топ-20 по абсолютной величине)
    plt.figure(figsize=(12, 10))
    top_features = coef_df.head(20)
    # Сортируем признаки по значению коэффициента (не по абсолютной величине) для наглядности
    top_features = top_features.sort_values(by='Coefficient')

    colors = ['red' if c < 0 else 'green' for c in top_features['Coefficient']]
    plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
    plt.xlabel('Коэффициент')
    plt.ylabel('Признак')
    plt.title('Важность признаков (топ-20 по абсолютной величине)')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/feature_importance.png")

    # Построение ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.savefig(f"{plots_dir}/roc_curve.png")

    # Сохранение модели
    joblib.dump(model, model_filename)
    print(f"Модель сохранена в {model_filename}")
    print(f"Метрики сохранены в {metrics_filename}")
    print(f"Графики сохранены в {plots_dir}")

    return model, accuracy, precision, recall, f1


if __name__ == "__main__":
    main()
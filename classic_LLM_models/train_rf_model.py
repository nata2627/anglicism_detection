import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc
import seaborn as sns
import joblib
from datetime import datetime

# Пути к файлам
input_csv = "assets/dataset_transformed.csv"
output_dir = "assets/models/random_forest"
os.makedirs(output_dir, exist_ok=True)

# Имя модели с временной меткой
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"{output_dir}/random_forest_model_{timestamp}.pkl"
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

    # Создаем и обучаем модель случайного леса
    print("Обучение модели случайного леса...")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    rf_model.fit(X_train, y_train)

    # Оценка модели на тестовой выборке
    print("Оцениваем модель на тестовых данных...")
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

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
        f.write(f"Модель Random Forest для классификации англицизмов\n")
        f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Размер датасета: {df.shape[0]} примеров, {df.shape[1]} признаков\n")
        f.write(f"Обучающая выборка: {X_train.shape[0]} примеров\n")
        f.write(f"Тестовая выборка: {X_test.shape[0]} примеров\n\n")
        f.write("Параметры модели:\n")
        for param, value in rf_model.get_params().items():
            f.write(f"  {param}: {value}\n")
        f.write("\nМетрики на тестовой выборке:\n")
        f.write(f"  Точность (Accuracy): {accuracy:.4f}\n")
        f.write(f"  Точность (Precision): {precision:.4f}\n")
        f.write(f"  Полнота (Recall): {recall:.4f}\n")
        f.write(f"  F1-мера: {f1:.4f}\n\n")
        f.write("Отчет по классификации:\n")
        f.write(classification_report(y_test, y_pred))

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

    # Анализ важности признаков
    plt.figure(figsize=(10, 8))
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_20_features = feature_importances.nlargest(20)
    sns.barplot(x=top_20_features.values, y=top_20_features.index)
    plt.title('Важность признаков (топ-20)')
    plt.xlabel('Важность')
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

    # Оценка зависимости точности от количества деревьев
    estimators = np.arange(10, 550, 50)
    accuracy_scores = []

    for n_estimators in estimators:
        print(f"Тестирование модели с {n_estimators} деревьями...")
        temp_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        temp_model.fit(X_train, y_train)
        temp_pred = temp_model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, temp_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(estimators, accuracy_scores, 'b-', linewidth=2)
    plt.xlabel('Количество деревьев')
    plt.ylabel('Точность (Accuracy)')
    plt.title('Зависимость точности от количества деревьев')
    plt.grid(True)
    plt.savefig(f"{plots_dir}/n_estimators_accuracy.png")

    # Сохранение модели
    joblib.dump(rf_model, model_filename)
    print(f"Модель сохранена в {model_filename}")
    print(f"Метрики сохранены в {metrics_filename}")
    print(f"Графики сохранены в {plots_dir}")

    return rf_model, accuracy, precision, recall, f1


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import seaborn as sns
import joblib
from datetime import datetime

# Пути к файлам
input_csv = "assets/dataset_transformed.csv"
output_dir = "assets/models/final"
os.makedirs(output_dir, exist_ok=True)

# Имя модели с временной меткой
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"{output_dir}/xgboost_model_{timestamp}.pkl"
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

    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Базовая модель для поиска параметров
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        verbosity=1,
        random_state=42,
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=10
    )

    # Отслеживаем процесс обучения
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )

    # Сохраняем результаты обучения
    results = xgb_model.evals_result()

    # Построение графика ошибки в процессе обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['validation_0']['logloss'], label='Train')
    plt.plot(results['validation_1']['logloss'], label='Test')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results['validation_0']['error'], label='Train')
    plt.plot(results['validation_1']['error'], label='Test')
    plt.xlabel('Iterations')
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/learning_curves.png")

    # Оценка модели на тестовой выборке
    print("Оцениваем модель на тестовых данных...")
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

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
        f.write(f"Модель XGBoost для классификации англицизмов\n")
        f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Размер датасета: {df.shape[0]} примеров, {df.shape[1]} признаков\n")
        f.write(f"Обучающая выборка: {X_train.shape[0]} примеров\n")
        f.write(f"Тестовая выборка: {X_test.shape[0]} примеров\n\n")
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
    xgb.plot_importance(xgb_model, max_num_features=20, importance_type='gain')
    plt.title('Важность признаков (топ-20)')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/feature_importance.png")

    # Сохранение модели
    joblib.dump(xgb_model, model_filename)
    print(f"Модель сохранена в {model_filename}")
    print(f"Метрики сохранены в {metrics_filename}")
    print(f"Графики сохранены в {plots_dir}")

    # Дополнительно - построение ROC-кривой
    from sklearn.metrics import roc_curve, auc
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

    return xgb_model, accuracy, precision, recall, f1


if __name__ == "__main__":
    main()
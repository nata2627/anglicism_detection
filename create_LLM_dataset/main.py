import os
import pandas as pd
import json
import re
from pathlib import Path


class DatasetProcessor:
    def __init__(self, input_path="assets/", output_path="assets/llm_datasets"):
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def _mark_all_anglicisms(self, text, anglicisms):
        # Сортируем англицизмы по длине (от длинных к коротким)
        sorted_anglicisms = sorted(anglicisms, key=len, reverse=True)

        # Создаем копию текста для разметки
        marked_text = text

        # Для каждого англицизма создаем регулярное выражение,
        # которое найдет точное совпадение слова с учетом границ
        for anglicism in sorted_anglicisms:
            # Создаем регулярное выражение с границами слова
            # Оно ищет англицизм, который может быть окружен границами слов,
            # пунктуацией или пробелами
            pattern = re.compile(r'(\b|(?<=\s)|(?<=\W))' + re.escape(anglicism) + r'(\b|(?=\s)|(?=\W))')

            # Заменяем все вхождения этого англицизма на размеченную версию
            marked_text = pattern.sub(r'<англицизм>\g<0></англицизм>', marked_text)

        return marked_text

    def create_pair_dataset(self, file_name="etalons.csv"):
        input_file = os.path.join(self.input_path, file_name)
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Файл {input_file} не найден")

        # Считываем исходный датасет
        df = pd.read_csv(input_file)

        # Создаем базовые колонки для выходного датасета
        result_df = pd.DataFrame()

        # Обрабатываем каждую строку
        input_texts = []
        target_texts = []

        for _, row in df.iterrows():
            original_text = row['Original Text']
            anglicisms = eval(row['Anglicisms'])  # Преобразуем строку в список
            replaced_text = row['Replaced Text']

            # Размечаем все англицизмы в тексте
            marked_text = self._mark_all_anglicisms(original_text, anglicisms)

            input_texts.append(marked_text)
            target_texts.append(replaced_text)

        # Формируем датасет в формате 'pair'
        result_df['original'] = input_texts
        result_df['replaced'] = target_texts
        return result_df

    def split_and_save_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        # Создаем датасет в формате пар
        full_df = self.create_pair_dataset()

        # Сначала сохраняем полный датасет
        full_output_file = os.path.join(self.output_path, "pair_dataset.csv")
        full_df.to_csv(full_output_file, index=False)
        print(f"Полный датасет сохранен в {full_output_file} ({len(full_df)} примеров)")

        # Перемешиваем датасет для случайного выбора
        shuffled_df = full_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Вычисляем размеры выборок, обеспечивая точно равные размеры для val и test
        n = len(shuffled_df)
        # Делаем val и test выборки точно равными
        val_size = int(n * val_ratio)
        test_size = val_size  # Теперь они будут иметь одинаковый размер
        train_size = n - (val_size + test_size)

        # Разделяем датасет
        train_df = shuffled_df[:train_size]
        val_df = shuffled_df[train_size:train_size + val_size]
        test_df = shuffled_df[train_size + val_size:]

        # Сохраняем разделенные выборки
        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            output_file = os.path.join(self.output_path, f"{name}_dataset.csv")
            split_df.to_csv(output_file, index=False, header=True, encoding='utf-8')
            print(f"{name.capitalize()} датасет сохранен в {output_file} ({len(split_df)} примеров)")

        # Чтение сохраненного файла для проверки
        test_read = pd.read_csv("assets/llm_datasets/train_dataset.csv", encoding='utf-8')
        print("Заголовки после чтения:", test_read.columns.tolist())

        return full_df, train_df, val_df, test_df


if __name__ == "__main__":
    # Пример использования: сохраняем полный датасет в формате 'pair' и его разделения
    processor = DatasetProcessor()
    full_df, train_df, val_df, test_df = processor.split_and_save_dataset()

    print(f"Всего создано {len(full_df)} примеров в полном датасете")
    print(f"Всего создано {len(train_df)} примеров для обучения")
    print(f"Всего создано {len(val_df)} примеров для валидации")
    print(f"Всего создано {len(test_df)} примеров для тестирования")
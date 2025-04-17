import os
import json
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import re
from rouge import Rouge
from tqdm.auto import tqdm

# Константы для специальных токенов
ANGLICISM_START = "<англ>"
ANGLICISM_END = "</англ>"


def prepare_dataset(csv_path):
    """
    Подготовка датасета из CSV файла.
    Функция парсит оригинальный текст и список англицизмов,
    и добавляет специальные токены вокруг англицизмов в тексте.
    """
    # Чтение CSV файла
    df = pd.read_csv(csv_path)

    # Преобразуем строку со списком англицизмов в реальный список
    df['anglicisms'] = df['anglicisms'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Создаем новые столбцы для обработанных текстов
    df['tagged_text'] = None
    df['expected_output'] = None

    processed_data = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Обработка датасета"):
        # Получаем оригинальный текст и список англицизмов
        original_text = row['original_text']
        anglicisms = row['anglicisms']

        # Создаем текст с тегами (для обучения)
        tagged_text = original_text

        # Словарь для замены англицизмов (здесь нужно будет добавить словарь замен)
        # В реальной системе это должен быть словарь англицизм -> русский эквивалент
        # Для демонстрации добавлены примеры замен
        replacements = {
            "лидер": "руководитель",
            "соцсети": "социальные сети",
            "пресс-конференции": "пресс-встречи",
            "путем": "способом",
            "драйвером": "движущей силой",
            "доминиона": "владения",
            "лишь": "только",
            "конфликта": "противостояния",
            "партия": "политическое объединение",
            "лишился": "потерял",
            "поста": "должности",
            "парламенте": "законодательном органе",
            "партию": "политическое объединение",
            "база": "основа",
            "парламента": "законодательного органа",
            "партии": "политические объединения",
            "баланс": "равновесие",
            "группе": "объединении",
            "Респондентов": "Опрошенных",
            "мере": "степени",
            "иные": "другие",
            "респондентов": "опрошенных",
            "массе": "большинстве",
            "весь": "полный",
            "респонденты": "опрошенные",
            "импорт": "ввоз",
            "иск": "заявление",
            "чемпионов": "победителей"
        }

        # Готовим ожидаемый выходной текст (с заменой англицизмов)
        expected_output = original_text

        # Отмечаем англицизмы тегами и подготавливаем ожидаемый выход
        for anglicism in anglicisms:
            # Ищем все вхождения англицизма в тексте
            pattern = r'\b' + re.escape(anglicism) + r'\b'

            # Заменяем англицизмы специальными токенами в исходном тексте
            tagged_text = re.sub(
                pattern,
                f"{ANGLICISM_START}{anglicism}{ANGLICISM_END}",
                tagged_text
            )

            # Заменяем англицизм на русский эквивалент в ожидаемом выходе
            if anglicism in replacements:
                expected_output = re.sub(
                    pattern,
                    replacements[anglicism],
                    expected_output
                )

        processed_data.append({
            "original_text": original_text,
            "tagged_text": tagged_text,
            "expected_output": expected_output,
            "anglicisms": anglicisms
        })

    return pd.DataFrame(processed_data)


class AnglicismDataset(Dataset):
    """Датасет для обучения модели замены англицизмов"""

    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Подготовка входных данных (текст с отмеченными англицизмами)
        input_text = f"Замени англицизмы на русские эквиваленты: {row['tagged_text']}"

        # Ожидаемый выход (текст с заменёнными англицизмами)
        target_text = row['expected_output']

        # Токенизация входного текста
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Токенизация ожидаемого выхода
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Создаём labels из target_ids, заменяя padding tokens на -100
        labels = target_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


def calculate_rouge_score(predictions, references):
    """Расчёт метрики ROUGE для оценки похожести текстов"""
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores


def custom_loss_function(logits, labels, model, tokenizer, original_texts, anglicisms_list):
    """
    Кастомная функция потерь:
    1. Базовая потеря CrossEntropy
    2. Штраф за неизмененные англицизмы
    3. Бонус за сохранение смысла (ROUGE)
    """
    # Стандартная потеря для языковой модели
    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Получаем предсказания модели
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Генерируем тексты из логитов для сравнения
    batch_size = logits.size(0)
    total_anglicism_penalty = 0.0
    total_rouge_bonus = 0.0

    for i in range(batch_size):
        # Извлекаем предсказанный текст
        pred_ids = torch.argmax(logits[i], dim=-1)
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

        # Оригинальный текст
        orig_text = original_texts[i]

        # Штрафы за неизмененные англицизмы
        anglicism_penalty = 0.0
        for anglicism in anglicisms_list[i]:
            if re.search(r'\b' + re.escape(anglicism) + r'\b', pred_text):
                # Англицизм присутствует в результате - применяем штраф
                anglicism_penalty += 0.1

        # Бонус за сохранение смысла (ROUGE)
        rouge_scores = calculate_rouge_score([pred_text], [orig_text])
        rouge_bonus = rouge_scores['rouge-l']['f'] * 0.1

        total_anglicism_penalty += anglicism_penalty
        total_rouge_bonus += rouge_bonus

    total_anglicism_penalty = total_anglicism_penalty / batch_size
    total_rouge_bonus = total_rouge_bonus / batch_size

    # Финальная функция потерь
    final_loss = ce_loss + total_anglicism_penalty - total_rouge_bonus

    return final_loss


def print_examples(model, tokenizer, example_df, device, num_examples=3):
    """Печатает примеры 'до и после' для визуальной проверки работы модели"""
    model.eval()

    # Выбираем случайные примеры
    example_indices = np.random.choice(len(example_df), min(num_examples, len(example_df)), replace=False)

    print("\n" + "=" * 80)
    print("ПРИМЕРЫ РАБОТЫ МОДЕЛИ:")
    print("=" * 80)

    for idx in example_indices:
        row = example_df.iloc[idx]

        # Входной текст (с отмеченными англицизмами)
        input_text = f"Замени англицизмы на русские эквиваленты: {row['tagged_text']}"

        # Токенизация входного текста
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Генерация ответа
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=1024,
                num_return_sequences=1,
                do_sample=False
            )

        # Декодирование результата
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Выводим исходный текст (без тегов для читаемости)
        print("\nИСХОДНЫЙ ТЕКСТ:")
        clean_input = row['original_text']
        # Выделим англицизмы жирным шрифтом для удобства чтения в консоли
        for anglicism in row['anglicisms']:
            pattern = r'\b' + re.escape(anglicism) + r'\b'
            clean_input = re.sub(pattern, f"\033[1m{anglicism}\033[0m", clean_input)
        print(clean_input[:500] + "..." if len(clean_input) > 500 else clean_input)

        # Выводим сгенерированный текст
        print("\nСГЕНЕРИРОВАННЫЙ ТЕКСТ:")
        print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)

        # Выводим оригинальный ожидаемый выход
        print("\nОЖИДАЕМЫЙ ТЕКСТ:")
        print(row['expected_output'][:500] + "..." if len(row['expected_output']) > 500 else row['expected_output'])

        print("\n" + "-" * 80)

    model.train()


def train_model(processed_df, output_dir="./anglicism_model"):
    """Функция для обучения модели"""
    # Инициализация токенизатора
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # Добавляем специальные токены
    special_tokens = {
        "additional_special_tokens": [ANGLICISM_START, ANGLICISM_END]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Инициализация модели
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                 device_map="auto",
                                                 torch_dtype=torch.float16)

    # Изменяем размер словаря токенов
    model.resize_token_embeddings(len(tokenizer))

    # Замораживаем все слои кроме последнего
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только последний слой трансформера
    # Получаем доступ к последнему слою
    last_layer = model.transformer.h[-1]
    for param in last_layer.parameters():
        param.requires_grad = True

    # Также размораживаем выходной слой (lm_head)
    for param in model.lm_head.parameters():
        param.requires_grad = True

    print(f"Общее количество параметров: {sum(p.numel() for p in model.parameters())}")
    print(f"Количество обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Создаем датасет и загрузчик данных
    dataset = AnglicismDataset(processed_df, tokenizer)

    # Разбиваем на тренировочную и валидационную части
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Преобразуем в HuggingFace Dataset для использования с Trainer
    def convert_to_hf_dataset(pytorch_dataset):
        data_dict = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for i in range(len(pytorch_dataset)):
            example = pytorch_dataset[i]
            data_dict["input_ids"].append(example["input_ids"].numpy())
            data_dict["attention_mask"].append(example["attention_mask"].numpy())
            data_dict["labels"].append(example["labels"].numpy())

        return HFDataset.from_dict(data_dict)

    hf_train_dataset = convert_to_hf_dataset(train_dataset)
    hf_val_dataset = convert_to_hf_dataset(val_dataset)

    # Настраиваем параметры обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # Немного уменьшаем learning rate для обучения только последнего слоя
        num_train_epochs=5,  # Увеличиваем количество эпох, так как обучаем меньше параметров
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="tensorboard",
        fp16=True,
        remove_unused_columns=False,
    )

    # Создаем небольшой датасет для примеров
    example_indices = np.random.choice(len(val_dataset), min(5, len(val_dataset)), replace=False)
    example_df = pd.DataFrame([processed_df.iloc[val_dataset.indices[i]] for i in example_indices])

    # Определяем кастомный колбэк для вывода примеров
    class ExampleCallback(TrainerCallback):
        def __init__(self, model, tokenizer, example_df, device):
            self.model = model
            self.tokenizer = tokenizer
            self.example_df = example_df
            self.device = device
            self.steps_to_print = [100, 500, 1000, 2000]  # На каких шагах печатать примеры

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.steps_to_print:
                print_examples(self.model, self.tokenizer, self.example_df, self.device, num_examples=2)

    # Инициализируем тренер с кастомным колбэком
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
        callbacks=[ExampleCallback(model, tokenizer, example_df, model.device)],
    )

    # Печатаем примеры до обучения
    print("\nПРИМЕРЫ ДО ОБУЧЕНИЯ:")
    print_examples(model, tokenizer, example_df, model.device, num_examples=2)

    # Запускаем обучение
    trainer.train()

    # Печатаем примеры после обучения
    print("\nПРИМЕРЫ ПОСЛЕ ОБУЧЕНИЯ:")
    print_examples(model, tokenizer, example_df, model.device, num_examples=2)

    # Сохраняем модель и токенизатор
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def evaluate_model(model, tokenizer, test_df):
    """Функция для оценки модели на тестовых данных"""
    model.eval()
    rouge = Rouge()

    results = []

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Оценка модели"):
        # Подготовка входных данных
        input_text = f"Замени англицизмы на русские эквиваленты: {row['tagged_text']}"

        # Токенизация
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Генерация ответа
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=1024,
                num_return_sequences=1,
                do_sample=False
            )

        # Декодирование результата
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Проверка изменённых англицизмов
        unchanged_anglicisms = []
        for anglicism in row['anglicisms']:
            if re.search(r'\b' + re.escape(anglicism) + r'\b', generated_text):
                unchanged_anglicisms.append(anglicism)

        # Расчёт метрики ROUGE
        rouge_scores = rouge.get_scores(generated_text, row['expected_output'])[0]

        result_item = {
            "original_text": row['original_text'],
            "tagged_text": row['tagged_text'],
            "generated_text": generated_text,
            "expected_text": row['expected_output'],
            "unchanged_anglicisms": unchanged_anglicisms,
            "rouge-1_f": rouge_scores['rouge-1']['f'],
            "rouge-2_f": rouge_scores['rouge-2']['f'],
            "rouge-l_f": rouge_scores['rouge-l']['f']
        }

        # Добавляем процент замененных англицизмов для каждого примера
        total_anglicisms = len(row['anglicisms'])
        replaced_anglicisms = total_anglicisms - len(unchanged_anglicisms)
        replacement_rate = replaced_anglicisms / total_anglicisms if total_anglicisms > 0 else 1.0
        result_item["replacement_rate"] = replacement_rate

        results.append(result_item)

        # Печатаем некоторые примеры оценки
        if i % max(1, len(test_df) // 5) == 0:  # Печатаем примерно 5 примеров
            print(f"\nПример {i + 1}/{len(test_df)}:")
            print(f"Исходный текст (фрагмент): {row['original_text'][:100]}...")
            print(f"Процент замененных англицизмов: {replacement_rate * 100:.2f}%")
            print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")

            # Выводим список незамененных англицизмов, если они есть
            if unchanged_anglicisms:
                print(f"Незамененные англицизмы: {', '.join(unchanged_anglicisms)}")

    return pd.DataFrame(results)


def main():
    # Путь к файлу с данными
    csv_path = "assets/anglicisms_dataset.csv"

    # Подготовка данных
    print("Подготовка данных...")
    processed_df = prepare_dataset(csv_path)

    # Выводим статистику по датасету
    anglicisms_counts = [len(row) for row in processed_df['anglicisms']]
    print(f"\nСтатистика по датасету:")
    print(f"Всего примеров: {len(processed_df)}")
    print(f"Общее количество англицизмов: {sum(anglicisms_counts)}")
    print(f"Среднее количество англицизмов на пример: {sum(anglicisms_counts) / len(processed_df):.2f}")
    print(f"Минимальное количество англицизмов: {min(anglicisms_counts)}")
    print(f"Максимальное количество англицизмов: {max(anglicisms_counts)}")

    # Разделение на обучающую и тестовую выборки
    train_df, test_df = train_test_split(processed_df, test_size=0.1, random_state=42)
    print(f"Размер обучающей выборки: {len(train_df)}")
    print(f"Размер тестовой выборки: {len(test_df)}")

    # Обучение модели
    print("\nЗапуск обучения модели...")
    model, tokenizer = train_model(train_df)

    # Оценка модели
    print("\nОценка модели...")
    eval_results = evaluate_model(model, tokenizer, test_df)

    # Сохранение результатов оценки
    eval_results.to_csv("evaluation_results.csv", index=False)

    # Вывод статистики
    print("\nРезультаты оценки:")
    print(f"Среднее значение ROUGE-L: {eval_results['rouge-l_f'].mean():.4f}")
    print(f"Среднее значение ROUGE-1: {eval_results['rouge-1_f'].mean():.4f}")
    print(f"Среднее значение ROUGE-2: {eval_results['rouge-2_f'].mean():.4f}")

    # Статистика по замене англицизмов
    avg_replacement_rate = eval_results['replacement_rate'].mean() * 100
    print(f"Средний процент замененных англицизмов: {avg_replacement_rate:.2f}%")

    # Выводим примеры самых успешных и неудачных замен
    eval_results = eval_results.sort_values(by='replacement_rate', ascending=False)

    print("\nПример успешной замены англицизмов:")
    best_example = eval_results.iloc[0]
    print(f"Исходный текст (фрагмент): {best_example['original_text'][:150]}...")
    print(f"Сгенерированный текст (фрагмент): {best_example['generated_text'][:150]}...")
    print(f"Процент замененных англицизмов: {best_example['replacement_rate'] * 100:.2f}%")

    if len(eval_results) > 1 and eval_results.iloc[-1]['replacement_rate'] < 1.0:
        print("\nПример неудачной замены англицизмов:")
        worst_example = eval_results.iloc[-1]
        print(f"Исходный текст (фрагмент): {worst_example['original_text'][:150]}...")
        print(f"Сгенерированный текст (фрагмент): {worst_example['generated_text'][:150]}...")
        print(f"Процент замененных англицизмов: {worst_example['replacement_rate'] * 100:.2f}%")
        print(f"Незамененные англицизмы: {', '.join(worst_example['unchanged_anglicisms'])}")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    main()
import os
import pandas as pd
import torch
import wandb
import evaluate
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training

# Устанавливаем сиды для воспроизводимости
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Определение системного промпта
SYSTEM_PROMPT = """
Твоя задача - заменить англицизмы в русском тексте на их русские эквиваленты.
Англицизмы - это заимствованные из английского языка слова, которые могут быть заменены русскими аналогами.
В тексте англицизмы будут помечены тегами <англицизм> и </англицизм>.
Например: "Он дал <англицизм>интервью</англицизм>" должно быть заменено на "Он дал беседу".
Замени только слова в этих тегах и верни текст без тегов разметки.
Постарайся сохранить исходный смысл текста.
"""


# Функция форматирования промпта для Qwen2.5
def format_prompt_for_qwen(system_prompt, user_text):
    """
    Форматирует промпт в соответствии с требованиями модели Qwen2.5-Instruct

    Args:
        system_prompt: Системный промпт
        user_text: Текст пользователя

    Returns:
        str: Отформатированный промпт
    """
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


class AnglicismDataset(Dataset):
    """Датасет для задачи замены англицизмов"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, data_fraction: float = 1.0):
        """
        Инициализация датасета

        Args:
            data_path: Путь к CSV файлу с данными
            tokenizer: Токенизатор для модели
            max_length: Максимальная длина последовательности
            data_fraction: Доля данных для использования (от 0 до 1)
        """
        # Загружаем полный датасет
        full_data = pd.read_csv(data_path)

        # Проверяем и корректируем долю данных
        data_fraction = max(0.0, min(1.0, data_fraction))  # Убеждаемся, что значение между 0 и 1

        if data_fraction < 1.0:
            # Выбираем случайную долю данных
            sample_size = int(len(full_data) * data_fraction)
            self.data = full_data.sample(sample_size, random_state=42)
            print(
                f"Используем {sample_size} примеров ({data_fraction:.2%} от полного датасета с {len(full_data)} записями)")
        else:
            # Используем все данные
            self.data = full_data
            print(f"Используем полный датасет: {len(self.data)} записей")

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Добавляем специальные токены для разметки англицизмов
        self.special_tokens = ["<англицизм>", "</англицизм>"]

        # Убеждаемся, что у токенизатора есть pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        # Формируем входной текст с правильным форматом для Qwen2.5-Instruct
        input_text = format_prompt_for_qwen(SYSTEM_PROMPT, item['original'])

        # Целевой текст - только ожидаемый ответ
        target_text = f"{item['replaced']}"

        # Токенизируем входной текст
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Токенизируем целевой текст для получения меток
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Подготавливаем метки: -100 для игнорирования при расчете loss
        labels = targets["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Формируем словарь с тензорами
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


class AnglicismTrainer:
    """Класс для обучения модели замены англицизмов с использованием LoRA"""

    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-3B-Instruct",
            train_path: str = "/kaggle/input/anglicism-train-dataset/train_dataset.csv",
            val_path: str = "/kaggle/input/anglicism-val-dataset/val_dataset.csv",
            save_path: str = "/kaggle/working/assets/llm_models/",
            batch_size: int = 1,
            max_length: int = 128,
            learning_rate: float = 2e-5,
            num_epochs: int = 3,
            warmup_steps: int = 100,
            weight_decay: float = 0.01,
            logging_steps: int = 1,  # Логирование каждый шаг
            eval_steps: int = 10,  # Валидация каждые 10 шагов
            save_steps: int = 10,  # Сохранение каждые 10 шагов
            gradient_accumulation_steps: int = 4,  # Аккумуляция градиента
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            train_data_fraction: float = 1.0,
            val_data_fraction: float = 1.0,
    ):
        """
        Инициализация класса обучения

        Args:
            model_name: Название модели
            train_path: Путь к тренировочному датасету
            val_path: Путь к валидационному датасету
            save_path: Путь для сохранения модели
            batch_size: Размер батча
            max_length: Максимальная длина последовательности
            learning_rate: Скорость обучения
            num_epochs: Количество эпох
            warmup_steps: Количество шагов прогрева
            weight_decay: Коэффициент затухания весов
            logging_steps: Шаги логирования (n)
            eval_steps: Шаги валидации (m)
            save_steps: Шаги сохранения
            gradient_accumulation_steps: Количество шагов для аккумуляции градиента
            lora_r: Ранг адаптера LoRA
            lora_alpha: Параметр альфа для LoRA
            lora_dropout: Вероятность дропаута для LoRA
            train_data_fraction: Доля тренировочных данных
            val_data_fraction: Доля валидационных данных
        """
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.train_data_fraction = train_data_fraction
        self.val_data_fraction = val_data_fraction

        # Инициализируем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Добавляем специальные токены для разметки англицизмов и формата Qwen2.5
        self.special_tokens = ["<англицизм>", "</англицизм>", "<|im_start|>", "<|im_end|>"]
        special_tokens_dict = {"additional_special_tokens": self.special_tokens}

        # Убеждаемся, что у токенизатора есть pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens: {self.special_tokens}")

        # Инициализируем модель
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )

        # Изменяем размер токенов в модели
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Настраиваем LoRA
        self._setup_lora()

        # Инициализируем метрики
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

        # Инициализируем датасеты с возможностью использования только части данных
        self.train_dataset = AnglicismDataset(train_path, self.tokenizer, max_length,
                                              data_fraction=self.train_data_fraction)
        self.val_dataset = AnglicismDataset(val_path, self.tokenizer, max_length, data_fraction=self.val_data_fraction)

        # Инициализируем аргументы обучения
        self.training_args = TrainingArguments(
            output_dir=self.save_path,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,  # Логирование каждый шаг
            eval_strategy="steps",
            eval_steps=self.eval_steps,  # Валидация каждые eval_steps шагов
            save_strategy="steps",
            save_steps=self.save_steps,  # Сохранение каждые save_steps шагов
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            report_to="wandb",
            save_total_limit=3,
            push_to_hub=False,
            remove_unused_columns=False,
            prediction_loss_only=False,  # Важно для получения предсказаний, а не только потерь
            run_name=f"anglicism-{wandb.util.generate_id()}"  # Уникальное имя для запуска
        )

    def _setup_lora(self):
        """Настраиваем LoRA для эффективного обучения модели"""
        print("Setting up LoRA...")

        # Подготавливаем модель для обучения
        try:
            self.model = prepare_model_for_kbit_training(self.model)
            print("Модель успешно подготовлена для kbit обучения")
        except Exception as e:
            print(f"Предупреждение: не удалось подготовить модель для kbit тренировки: {e}")
            print("Продолжаем с LoRA настройкой без kbit подготовки")

        # Соберем все возможные имена модулей для LoRA
        print("Анализ структуры модели для LoRA...")
        module_names = []
        for name, _ in self.model.named_modules():
            if any(x in name for x in ["query", "key", "value", "attention", "mlp", "dense", "feed_forward"]):
                module_names.append(name)

        # Определим имена целевых модулей на основе структуры модели
        target_modules = []

        # Проверим структуру модели и найдем общие паттерны
        if any(("q_proj" in name) for name, _ in self.model.named_modules()):
            target_modules.extend(["q_proj", "v_proj", "k_proj", "o_proj"])

        if any(("query" in name) for name, _ in self.model.named_modules()):
            target_modules.extend(["query", "value", "key", "dense"])

        if any(("gate_proj" in name) for name, _ in self.model.named_modules()):
            target_modules.extend(["gate_proj", "up_proj", "down_proj"])

        if any(("mlp.dense_h_to_4h" in name) for name, _ in self.model.named_modules()):
            target_modules.extend(["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"])

        # Если не удалось определить структуру, используем универсальное решение
        if not target_modules:
            print("Не удалось определить структуру модели для LoRA, использую универсальный подход")
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

        # Удаляем дубликаты в target_modules
        target_modules = list(set(target_modules))
        print(f"Выбраны целевые модули для LoRA: {target_modules}")

        # Настраиваем LoRA конфигурацию
        lora_config = LoraConfig(
            r=self.lora_r,  # Ранг адаптера
            lora_alpha=self.lora_alpha,  # Параметр альфа для LoRA
            target_modules=target_modules,  # Таргетные модули
            lora_dropout=self.lora_dropout,  # Вероятность дропаута
            bias="none",  # Не обучаем смещения
            task_type=TaskType.CAUSAL_LM  # Тип задачи
        )

        # Создаем LoRA модель
        self.model = get_peft_model(self.model, lora_config)

        # Выводим информацию о LoRA модели
        self.model.print_trainable_parameters()

    def compute_metrics(self, eval_pred):
        """
        Вычисление метрик для валидации и логирование примеров в W&B

        Args:
            eval_pred: Предсказания модели

        Returns:
            dict: Словарь с метриками
        """
        predictions, labels = eval_pred

        # Преобразуем предсказания в ожидаемый формат
        if isinstance(predictions, list) and isinstance(predictions[0], list):
            # Если предсказания в виде списка списков, преобразуем их в плоский список
            predictions = [item for sublist in predictions for item in sublist]

        # Проверяем, может ли это быть логитами (трехмерный тензор: [batch, seq_len, vocab_size])
        if isinstance(predictions, np.ndarray) and len(predictions.shape) == 3:
            # Берем индекс максимального логита для каждого токена
            predictions = np.argmax(predictions, axis=-1)

        # Обеспечиваем, что предсказания являются numpy массивом
        predictions = np.array(predictions) if not isinstance(predictions, np.ndarray) else predictions

        try:
            # Декодируем предсказания и метки
            decoded_preds = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )

            # Преобразуем -100 обратно в pad_token_id для корректного декодирования
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )

            # Очищаем предсказания и метки
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            # Очищаем предсказания от системных промптов и форматных токенов
            cleaned_preds = []
            for pred in decoded_preds:
                # Проверяем и извлекаем только часть ответа ассистента
                if "<|im_start|>assistant" in pred:
                    # Извлекаем текст между маркерами начала ответа ассистента и конца сообщения
                    parts = pred.split("<|im_start|>assistant\n")
                    if len(parts) > 1:
                        assistant_response = parts[1]
                        # Проверяем наличие маркера конца
                        if "<|im_end|>" in assistant_response:
                            assistant_response = assistant_response.split("<|im_end|>")[0]
                        cleaned_preds.append(assistant_response.strip())
                    else:
                        cleaned_preds.append(pred.strip())
                else:
                    # Если формат не соответствует ожидаемому, используем оригинальное предсказание
                    cleaned_preds.append(pred.strip())

            # Логируем примеры в W&B, если он инициализирован
            if wandb.run is not None:
                # Создаем таблицу с примером текста для отображения в интерфейсе wandb
                example_table = wandb.Table(columns=["Target", "Prediction"])

                # Выбираем первый пример для логирования согласно требованиям
                if len(cleaned_preds) > 0:
                    # Добавляем пример в таблицу
                    example_table.add_data(decoded_labels[0], cleaned_preds[0])

                    # Логируем таблицу и отдельные строки
                    wandb.log({
                        "examples_table": example_table,
                        "example_target": decoded_labels[0],
                        "example_prediction": cleaned_preds[0]
                    })

            # Вычисляем ROUGE
            rouge_results = self.rouge.compute(
                predictions=cleaned_preds,
                references=decoded_labels,
                use_stemmer=True
            )

            # Вычисляем BLEU
            bleu_results = self.bleu.compute(
                predictions=cleaned_preds,
                references=[[label] for label in decoded_labels]
            )

            # Объединяем метрики
            results = {
                "rouge1": rouge_results["rouge1"],
                "rouge2": rouge_results["rouge2"],
                "rougeL": rouge_results["rougeL"],
                "bleu": bleu_results["bleu"]
            }

            return results
        except Exception as e:
            print(f"Ошибка при вычислении метрик: {e}")
            print(f"Форма предсказаний: {predictions.shape if hasattr(predictions, 'shape') else 'Нет атрибута shape'}")
            print(f"Пример предсказания: {predictions[0] if len(predictions) > 0 else 'Нет предсказаний'}")

            # В случае ошибки возвращаем базовые метрики
            return {"error": str(e), "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

    def data_collator(self, features):
        """
        Оптимизированный коллатор данных для обучения

        Args:
            features: Список фич

        Returns:
            dict: Словарь с тензорами
        """
        # Проверяем, что список фич не пустой
        if len(features) == 0:
            return {}

        # Извлекаем все ключи из первого элемента
        keys = features[0].keys()

        # Создаем словарь для batch
        batch = {}

        for key in keys:
            # Для каждого ключа сначала преобразуем в numpy массивы, затем в тензор
            if all(isinstance(feature[key], torch.Tensor) for feature in features):
                # Если все элементы уже тензоры, просто собираем их в стек
                if len(features[0][key].shape) == 0:  # Скаляры
                    batch[key] = torch.stack([feature[key] for feature in features])
                else:
                    try:
                        batch[key] = torch.stack([feature[key] for feature in features])
                    except:
                        # Если не удается сделать стек, используем cat с размерностью 0
                        batch[key] = torch.cat([feature[key].unsqueeze(0) for feature in features], dim=0)
            else:
                # Для не-тензорных данных, сначала преобразуем в numpy массив
                try:
                    numpy_array = np.array([feature[key] for feature in features])
                    batch[key] = torch.tensor(numpy_array, dtype=torch.int64)
                except:
                    # Если не удается сконвертировать, используем индивидуальное преобразование
                    batch[key] = [feature[key] for feature in features]

        return batch

    def train(self):
        """Обучение модели"""
        print("Starting training...")

        # Инициализируем W&B
        try:
            wandb.init(
                project="anglicism-replacement",
                name=f"{self.model_name.split('/')[-1]}-finetuned-{wandb.util.generate_id()}",
                config={
                    "model_name": self.model_name,
                    "batch_size": self.batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "max_length": self.max_length,
                    "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                    "train_data_fraction": self.train_data_fraction,
                    "val_data_fraction": self.val_data_fraction,
                    "train_examples": len(self.train_dataset),
                    "val_examples": len(self.val_dataset),
                    "total_training_steps": len(self.train_dataset) * self.num_epochs // (
                            self.batch_size * self.gradient_accumulation_steps),
                    "transformers_version": transformers.__version__,  # Добавляем версию библиотеки
                }
            )
        except Exception as e:
            print(f"Предупреждение: не удалось инициализировать W&B: {e}")
            print("Продолжение без W&B логирования...")

        # Проверяем и исправляем настройки для совместимости с вашей версией transformers
        if hasattr(self.training_args, "report_to") and "wandb" in self.training_args.report_to:
            if wandb.run is None:
                # Если W&B не инициализирован, отключаем отчеты
                self.training_args.report_to = []

        # Добавляем метку для решения проблемы с PeftModelForCausalLM
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

        # Инициализируем тренер
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        # Обучаем модель
        try:
            print("Начало обучения модели...")
            trainer.train()
            print("Обучение успешно завершено!")
        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
            import traceback
            traceback.print_exc()

            # Если возможно, сохраняем модель даже при ошибке
            try:
                print("Попытка сохранения модели после ошибки...")
                trainer.save_model(os.path.join(self.save_path, "model_after_error"))
                self.tokenizer.save_pretrained(os.path.join(self.save_path, "model_after_error"),
                                               save_embedding_layers=True)
                print("Модель сохранена после ошибки!")
            except Exception as save_error:
                print(f"Не удалось сохранить модель: {save_error}")

            # Перезапускаем wandb, если он запущен
            if wandb.run is not None:
                wandb.finish()

            raise e  # Повторно вызываем исключение для информирования о проблеме

        # Сохраняем модель
        trainer.save_model(os.path.join(self.save_path, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.save_path, "final_model"), save_embedding_layers=True)

        # Завершаем W&B
        wandb.finish()

        print("Training completed!")

    def generate_replacement(self, text):
        """
        Генерация замен для текста с англицизмами

        Args:
            text: Входной текст с размеченными англицизмами

        Returns:
            str: Текст с замененными англицизмами
        """
        # Формируем запрос к модели с системным промптом в формате Qwen2.5
        input_text = format_prompt_for_qwen(SYSTEM_PROMPT, text)

        # Токенизируем входной текст
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Генерируем предсказание с температурной выборкой для разнообразия
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        # Декодируем предсказание
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем только ответ ассистента из форматированного вывода
        if "<|im_start|>assistant" in output_text:
            parts = output_text.split("<|im_start|>assistant\n")
            if len(parts) > 1:
                output_text = parts[1]
                # Удаляем конечный маркер, если он есть
                if "<|im_end|>" in output_text:
                    output_text = output_text.split("<|im_end|>")[0]
        # Если формат не соответствует, проверяем наличие системного промпта
        elif SYSTEM_PROMPT in output_text:
            output_text = output_text.replace(SYSTEM_PROMPT, "").strip()

        # Убеждаемся, что все теги разметки удалены из выходного текста
        import re
        output_text = re.sub(r'<англицизм>|</англицизм>', '', output_text)
        # Удаляем также все оставшиеся маркеры формата Qwen2.5
        output_text = re.sub(r'<\|im_start\|>|<\|im_end\|>|system|user|assistant', '', output_text)

        return output_text.strip()


# Пример использования
def main():
    print("\n\n==== ЗАПУСК ОСНОВНОЙ ФУНКЦИИ ====\n\n")
    # Инициализируем тренер с LoRA
    trainer = AnglicismTrainer(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        train_path="/kaggle/input/anglicism-train-dataset/train_dataset.csv",
        val_path="/kaggle/input/anglicism-val-dataset/val_dataset.csv",
        save_path="/kaggle/working/assets/llm_models/",
        batch_size=1,
        max_length=256,
        learning_rate=2e-5,
        num_epochs=3,
        logging_steps=1,  # Логирование каждый шаг
        eval_steps=100,  # Валидация каждые 100 шагов
        save_steps=100,  # Сохранение каждые 100 шагов
        gradient_accumulation_steps=4,  # Аккумуляция градиента
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        train_data_fraction=0.005,
        val_data_fraction=0.01
    )

    # Оценка количества итераций
    print(f"Размер тренировочного датасета: {len(trainer.train_dataset)} примеров")
    print(f"Размер батча: {trainer.batch_size}")
    print(f"Аккумуляция градиента: {trainer.gradient_accumulation_steps} шагов")
    print(f"Эффективный размер батча: {trainer.batch_size * trainer.gradient_accumulation_steps}")
    print(f"Количество эпох: {trainer.num_epochs}")
    print(
        f"Ожидаемое количество итераций в эпохе: {len(trainer.train_dataset) // (trainer.batch_size * trainer.gradient_accumulation_steps)}")
    total_iterations = (len(trainer.train_dataset) // (
            trainer.batch_size * trainer.gradient_accumulation_steps)) * trainer.num_epochs
    print(f"Общее ожидаемое количество итераций: {total_iterations}")

    # Обучаем модель
    trainer.train()

    # Сохраняем только адаптеры LoRA и токенизатор
    final_model_path = os.path.join(trainer.save_path, "final_lora_model")
    trainer.model.save_pretrained(final_model_path, save_embedding_layers=True)
    trainer.tokenizer.save_pretrained(final_model_path, save_embedding_layers=True)


if __name__ == "__main__":
    main()
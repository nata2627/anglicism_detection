import os
import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import wandb
import re
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback


class WandbTableCallback(TrainerCallback):
    """Callback для создания таблицы с примерами в wandb."""

    def __init__(self, trainer, dataset, tokenizer, model, num_fixed_examples=3, num_random_examples=3,
                 log_examples_every=50):
        self.trainer = trainer
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.num_fixed_examples = num_fixed_examples
        self.num_random_examples = num_random_examples
        self.log_examples_every = log_examples_every

        # Создаем таблицу wandb
        self.examples_data = []  # Храним данные для таблицы
        self.columns = ["Шаг", "Тип", "Оригинал", "Таргет", "Сгенерированное"]

        # Размер датасета
        self.dataset_size = len(dataset)

        # Выбираем фиксированные примеры
        self.fixed_indices = random.sample(range(self.dataset_size), min(num_fixed_examples, self.dataset_size))

    def on_step_end(self, args, state, control, **kwargs):
        """Вызывается после каждого шага обучения."""
        if state.global_step % self.log_examples_every == 0:
            # Безопасное получение значения loss из log_history
            try:
                # Перебираем записи в обратном порядке, чтобы найти последнюю с ключом 'loss'
                current_loss = 0
                if state.log_history:
                    for entry in reversed(state.log_history):
                        if 'loss' in entry:
                            current_loss = entry['loss']
                            break
                wandb.log({"train/loss": current_loss})  # Убран параметр step
            except Exception as e:
                print(f"Ошибка при логировании loss: {e}")
                # Продолжаем выполнение даже при ошибке логирования

            # Создаем таблицу для текущего шага
            current_examples = []

            # 1. Обрабатываем фиксированные примеры
            for idx in self.fixed_indices:
                # Получаем оригинальный текст
                original_text = self._extract_original_text(idx)

                # Получаем целевой текст (таргет)
                target_text = self._extract_target_text(idx)

                # Генерируем текст с помощью модели
                generated_text = self._generate_text(original_text)

                # Сохраняем данные
                current_examples.append(
                    [state.global_step, "Фиксированный", original_text, target_text, generated_text])

            # 2. Выбираем и обрабатываем случайные примеры
            if self.num_random_examples > 0:
                # Выбираем новые случайные индексы (исключая фиксированные)
                available_indices = list(set(range(self.dataset_size)) - set(self.fixed_indices))
                if len(available_indices) > 0:
                    random_indices = random.sample(
                        available_indices,
                        min(self.num_random_examples, len(available_indices))
                    )

                    for idx in random_indices:
                        # Получаем оригинальный текст
                        original_text = self._extract_original_text(idx)

                        # Получаем целевой текст (таргет)
                        target_text = self._extract_target_text(idx)

                        # Генерируем текст с помощью модели
                        generated_text = self._generate_text(original_text)

                        # Сохраняем данные
                        current_examples.append(
                            [state.global_step, "Случайный", original_text, target_text, generated_text])

            # Добавляем данные текущего шага в общий список
            self.examples_data.extend(current_examples)

            # Создаем новую таблицу из всех накопленных данных
            table = wandb.Table(columns=self.columns, data=self.examples_data)

            # Логируем обновленную таблицу (убран параметр step)
            wandb.log({"examples_table": table})

    def _extract_original_text(self, idx):
        """Извлекает оригинальный текст из датасета."""
        return self.dataset.data.iloc[idx]['original']

    def _extract_target_text(self, idx):
        """Извлекает целевой текст из датасета."""
        return self.dataset.data.iloc[idx]['replaced']

    def _generate_text(self, original_text):
        """Генерирует текст с помощью модели."""
        # Создаем инструкцию для модели
        instruction = (
            "Инструкция: Замените англицизмы в тексте на их русские аналоги.\n\n"
            f"Текст: {original_text}\n\n"
            f"Результат:"
        )

        # Форматируем для модели
        formatted_input = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        # Токенизация
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)

        # Генерация текста
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Декодирование и очистка результата
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Извлекаем только ответ ассистента
        assistant_part = generated_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]

        return assistant_part.strip()


class AngliclsmReplacementTrainer:
    def __init__(
            self,
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            dataset_path="/kaggle/input/anglicism-train-dataset/train_dataset.csv",
            validation_path="/kaggle/input/anglicism-val-dataset/val_dataset.csv",
            # Добавлен путь к валидационному набору
            validation_fraction=0.1,  # Доля валидационных данных для использования
            output_dir="/kaggle/working/assets/train/",
            wandb_project="anglicism-replacement",
            wandb_entity=None,  # Ваш логин или организация на wandb
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            batch_size=1,
            num_epochs=3,
            learning_rate=1e-4,
            max_length=256,
            weight_decay=0.01,
            warmup_ratio=0.1,
            save_steps=100,  # Сохранять модель каждые 100 шагов
            eval_steps=1000,  # Проводить валидацию каждые 1000 шагов
            save_total_limit=5,  # Хранить не более 5 моделей
            log_examples_every=50,  # Логировать примеры каждые N шагов
            num_fixed_examples=3,  # Количество фиксированных примеров
            num_random_examples=3,  # Количество случайных примеров
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.validation_path = validation_path  # Добавлен путь к валидационному набору
        self.validation_fraction = validation_fraction  # Доля валидационных данных
        self.output_dir = output_dir
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.save_steps = save_steps  # Добавлен параметр для сохранения по шагам
        self.eval_steps = eval_steps  # Добавлен параметр для валидации по шагам
        self.save_total_limit = save_total_limit  # Добавлено ограничение на количество сохраняемых моделей
        self.log_examples_every = log_examples_every
        self.num_fixed_examples = num_fixed_examples
        self.num_random_examples = num_random_examples
        self.device = device

        # Создаем директорию для сохранения модели, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Инициализация wandb
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            config={
                "model_name": model_name,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "save_steps": save_steps,  # Добавлено в конфигурацию
                "eval_steps": eval_steps,  # Добавлено в конфигурацию
                "save_total_limit": save_total_limit,  # Добавлено в конфигурацию
                "validation_fraction": validation_fraction,  # Доля валидационных данных
                "num_fixed_examples": num_fixed_examples,
                "num_random_examples": num_random_examples,
            }
        )

    def _load_tokenizer_and_model(self):
        """Загрузка токенизатора и модели"""
        print(f"Загрузка модели {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Добавляем специальные токены для начала и конца, если их нет
        special_tokens = {"additional_special_tokens": ["<англицизм>", "</англицизм>"]}
        self.tokenizer.add_special_tokens(special_tokens)

        # Загружаем модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Изменяем размер эмбеддингов для новых токенов
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Подготовка модели для 4-битного обучения и настройка LoRA
        self.model = prepare_model_for_kbit_training(self.model)

        # Настройка конфигурации LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        # Получаем PEFT модель
        self.model = get_peft_model(self.model, lora_config)
        print(f"Модель загружена и подготовлена для обучения с LoRA")

    def _create_dataset(self, data_path, fraction=1.0):
        """Создание датасета из CSV файла с возможностью выбора доли данных"""

        class AnglicismDataset(Dataset):
            def __init__(self, data_path, tokenizer, max_length, fraction=1.0):
                # Загружаем данные
                full_data = pd.read_csv(data_path)

                # Если нужна только часть данных, берем случайную выборку
                if fraction < 1.0:
                    self.data = full_data.sample(frac=fraction, random_state=42)
                    print(f"Используется {len(self.data)} из {len(full_data)} примеров ({fraction * 100:.1f}%)")
                else:
                    self.data = full_data

                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                original = self.data.iloc[idx]['original']
                replaced = self.data.iloc[idx]['replaced']

                # Создаем инструкцию для модели
                instruction = (
                    "Инструкция: Замените англицизмы в тексте на их русские аналоги.\n\n"
                    f"Текст: {original}\n\n"
                    f"Результат: {replaced}"
                )

                # Создаем правильный формат ввода для модели Qwen2
                formatted_input = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{replaced}<|im_end|>"

                # Токенизация ввода
                encoded = self.tokenizer(
                    formatted_input,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                # Готовим inputs и labels для обучения
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)

                # Для обучения labels должны быть такими же, как input_ids
                labels = input_ids.clone()

                # Устанавливаем -100 для токенов, которые не нужно учитывать при обучении (часть пользователя)
                user_part_end = formatted_input.find("<|im_start|>assistant")
                if user_part_end != -1:
                    user_tokens = self.tokenizer(
                        formatted_input[:user_part_end],
                        add_special_tokens=False
                    )["input_ids"]
                    labels[:len(user_tokens)] = -100

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }

        return AnglicismDataset(data_path, self.tokenizer, self.max_length, fraction=fraction)

    def _prepare_dataset(self):
        """Подготовка тренировочного датасета"""
        print("Загрузка и подготовка тренировочного датасета...")
        self.dataset = self._create_dataset(self.dataset_path, fraction=1.0)
        print(f"Тренировочный датасет подготовлен, количество примеров: {len(self.dataset)}")

    def _prepare_validation_dataset(self):
        """Подготовка валидационного датасета с использованием указанной доли данных"""
        print(
            f"Загрузка и подготовка валидационного датасета (используется {self.validation_fraction * 100:.1f}% данных)...")
        self.validation_dataset = self._create_dataset(
            self.validation_path,
            fraction=self.validation_fraction
        )
        print(f"Валидационный датасет подготовлен, количество примеров: {len(self.validation_dataset)}")

    def train(self):
        """Обучение модели"""
        self._load_tokenizer_and_model()
        self._prepare_dataset()
        self._prepare_validation_dataset()  # Загружаем валидационный датасет

        # Настройка параметров обучения с новыми опциями
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,  # Размер батча для валидации
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            # Новые параметры для сохранения и валидации
            save_strategy="steps",  # Стратегия сохранения по шагам
            save_steps=self.save_steps,  # Сохраняем каждые N шагов
            save_total_limit=self.save_total_limit,  # Максимальное количество сохраняемых моделей
            eval_strategy="steps",  # Стратегия оценки по шагам
            eval_steps=self.eval_steps,  # Оцениваем каждые N шагов
            load_best_model_at_end=True,  # Загрузить лучшую модель в конце
            metric_for_best_model="eval_loss",  # Метрика для определения лучшей модели
            greater_is_better=False,  # Для loss меньше значит лучше
            fp16=True,
            report_to="wandb",  # Включаем отчеты в wandb
            remove_unused_columns=False,
            gradient_accumulation_steps=16,
            optim="adamw_torch",
            logging_steps=1
        )

        # Инициализация тренера с добавлением валидационного датасета
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.validation_dataset,  # Добавляем валидационный датасет
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )

        # Добавляем собственный колбэк для логирования примеров
        wandb_table_callback = WandbTableCallback(
            trainer=trainer,
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            model=self.model,
            num_fixed_examples=self.num_fixed_examples,
            num_random_examples=self.num_random_examples
        )

        trainer.add_callback(wandb_table_callback)

        # Запуск обучения
        print("Начало обучения...")
        trainer.train()

        # Сохранение модели и токенизатора
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Обучение завершено. Модель сохранена в {self.output_dir}")

        # Завершаем сессию wandb
        wandb.finish()


if __name__ == "__main__":
    # Пример использования класса с новыми параметрами
    trainer = AngliclsmReplacementTrainer(
        validation_fraction=0.05,  # Использовать % валидационных данных
        save_steps=250,  # Сохранение
        eval_steps=250,  # Валидация
        save_total_limit=5  # Хранить не более 5 моделей
    )
    trainer.train()
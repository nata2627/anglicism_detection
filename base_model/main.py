import os
import json
import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
import logging
import argparse
from tqdm import tqdm
import random
import sys
import time

# Константы для специальных токенов замены
TOKEN_START_REPLACE = "<АНГЛ>"
TOKEN_END_REPLACE = "</АНГЛ>"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assets/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Класс для мониторинга процесса обучения и вывода примеров
class ExampleOutputCallback(TrainerCallback):
    """
    Колбэк для вывода примеров во время обучения.
    Показывает входные и выходные тексты для мониторинга прогресса замены англицизмов.
    """

    def __init__(self, tokenizer, eval_examples, display_freq=100):
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples
        self.display_freq = display_freq  # Показывать примеры каждые N шагов
        self.examples_shown = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Вызывается в конце каждого шага"""
        # Проверяем, нужно ли показать пример
        if state.global_step % self.display_freq == 0 and model is not None:
            # Выбираем случайный пример из валидационного датасета
            example_idx = random.randint(0, len(self.eval_examples) - 1)
            example = self.eval_examples[example_idx]

            input_text = example["input"]
            original_text = example.get("original_text", "")
            anglicisms = example.get("anglicisms", [])
            expected_target = example["target"].split("\n\nПреобразованный текст:\n")[-1]

            # Выводим исходный текст с отмеченными англицизмами
            print(f"\n\n{'=' * 70}")
            print(f"ПРИМЕР #{self.examples_shown + 1} (ШАГ {state.global_step})")
            print(f"{'=' * 70}")
            print(f"ВХОДНОЙ ТЕКСТ:")
            print(f"{input_text}")
            print(f"\nОРИГИНАЛЬНЫЙ ТЕКСТ: {original_text}")
            print(f"АНГЛИЦИЗМЫ: {anglicisms}")

            # Генерируем ответ модели
            try:
                # Переводим модель в режим оценки
                model.eval()

                # Подготавливаем входные данные
                inputs = self.tokenizer(input_text, return_tensors="pt").to(model.device)

                # Генерируем выход
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        num_return_sequences=1
                    )

                # Декодируем результат
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Извлекаем только преобразованный текст
                parts = generated_text.split("Преобразованный текст:")
                if len(parts) > 1:
                    result_text = parts[1].strip()
                else:
                    result_text = generated_text.strip()

                # Проверяем, были ли заменены все англицизмы
                replaced_count = 0
                for word in anglicisms:
                    # Проверяем наличие слова в оригинальном и преобразованном тексте
                    pattern = r'\b' + re.escape(word) + r'\b'
                    if re.search(pattern, original_text, re.IGNORECASE) and not re.search(pattern, result_text,
                                                                                          re.IGNORECASE):
                        replaced_count += 1

                replacement_rate = replaced_count / len(anglicisms) if anglicisms else 0

                # Выводим сгенерированный текст
                print("\nВЫХОД МОДЕЛИ:")
                print(result_text)

                # Выводим статистику
                print(f"\nЗАМЕНЕНО АНГЛИЦИЗМОВ: {replaced_count}/{len(anglicisms)} ({replacement_rate * 100:.1f}%)")

                # Увеличиваем счетчик показанных примеров
                self.examples_shown += 1

                # Сбрасываем буфер вывода для немедленного отображения
                sys.stdout.flush()

                # Даем время на просмотр результата
                time.sleep(1)

                # Восстанавливаем режим обучения
                model.train()

            except Exception as e:
                print(f"Ошибка при генерации примера: {e}")

            print(f"{'=' * 70}\n")


# Проверка и создание необходимых директорий
def setup_paths():
    """Создает необходимые директории для сохранения моделей и результатов"""
    os.makedirs("assets/trained_model", exist_ok=True)
    logger.info("Проверка директорий для сохранения модели")


# Класс для нашего датасета
class AnglicismsDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example.get("input", "")
        target_text = example.get("target", "")

        # Токенизируем входные данные
        input_tokens = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Токенизируем целевой текст
        target_tokens = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Создаем метки для Supervised Fine-Tuning
        # -100 для входного текста (не учитывается в loss) и actual token IDs для целевого
        labels = torch.clone(input_tokens.input_ids[0])
        labels[:] = -100  # Все метки инициализируем как -100 (игнорируются при расчете loss)

        # Объединяем ввод и вывод для обучения
        combined_input_ids = torch.cat([input_tokens.input_ids[0], target_tokens.input_ids[0]])
        combined_attention_mask = torch.cat([input_tokens.attention_mask[0], target_tokens.attention_mask[0]])

        # Добавляем метки только для целевой части
        combined_labels = torch.cat([labels, target_tokens.input_ids[0]])

        # Обрезаем, если превышаем максимальную длину
        if len(combined_input_ids) > self.max_length:
            combined_input_ids = combined_input_ids[:self.max_length]
            combined_attention_mask = combined_attention_mask[:self.max_length]
            combined_labels = combined_labels[:self.max_length]

        return {
            'input_ids': combined_input_ids,
            'attention_mask': combined_attention_mask,
            'labels': combined_labels
        }


# Функция для подготовки обучающих примеров
def prepare_training_examples(dataset_path):
    """
    Создает обучающие примеры для замены англицизмов.
    Каждый пример состоит из:
    1. Входной текст с англицизмами, выделенными специальными токенами
    2. Целевой текст с замененными англицизмами
    """
    print(f"Подготовка обучающих примеров из файла {dataset_path}")
    logger.info(f"Подготовка обучающих примеров из файла {dataset_path}")

    # Загружаем датасет
    try:
        df = pd.read_csv(dataset_path)
        print(f"Загружен датасет с {len(df)} записями")
        logger.info(f"Загружен датасет с {len(df)} записями")
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        logger.error(f"Ошибка при загрузке датасета: {e}")
        return []

    # Загружаем модель для оценки семантической близости
    similarity_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Загружена модель для оценки семантической близости")
    logger.info("Загружена модель для оценки семантической близости")

    # Подготовим примеры для обучения
    training_examples = []
    stats = {"total": 0, "valid": 0, "anglicisms": 0}

    # Отключаем tqdm на время обработки для лучшего вывода информации
    for index, row in df.iterrows():
        try:
            stats["total"] += 1

            # Показываем прогресс каждые 10 записей
            if index % 10 == 0:
                print(f"Обработано {index}/{len(df)} записей")

            original_text = row['original_text']

            # Извлекаем список англицизмов
            try:
                anglicisms = json.loads(row['anglicisms'])
                if not isinstance(anglicisms, list):
                    print(f"Некорректный формат англицизмов в строке {index}: {row['anglicisms']}")
                    logger.warning(f"Некорректный формат англицизмов в строке {index}: {row['anglicisms']}")
                    continue
            except Exception as e:
                print(f"Ошибка при разборе англицизмов в строке {index}: {e}")
                logger.warning(f"Ошибка при разборе англицизмов в строке {index}: {e}")
                continue

            # Пропускаем, если нет англицизмов
            if not anglicisms:
                continue

            stats["anglicisms"] += len(anglicisms)

            # Создаем входной текст с отмеченными англицизмами
            input_text = original_text
            for word in anglicisms:
                # Заменяем все вхождения англицизма на выделенную версию
                pattern = r'\b' + re.escape(word) + r'\b'
                input_text = re.sub(pattern, f"{TOKEN_START_REPLACE} {word} {TOKEN_END_REPLACE}", input_text)

            # Добавляем инструкцию к входному тексту
            instruction = """Замените отмеченные англицизмы на подходящие русские синонимы, сохраняя смысл текста.
Англицизмы отмечены тегами <АНГЛ> и </АНГЛ>.

Текст с отмеченными англицизмами:
"""
            input_with_instruction = instruction + input_text

            # Целевой текст - тот же, что и входной, но не помеченный
            # (на этапе обучения модель должна сама научиться находить замены)
            target_text = f"\n\nПреобразованный текст:\n{original_text}"

            # Сохраняем пример
            example = {
                "input": input_with_instruction,
                "target": target_text,
                "original_text": original_text,
                "anglicisms": anglicisms
            }

            training_examples.append(example)
            stats["valid"] += 1

            # Выводим пример каждые 100 обработанных примеров
            if stats["valid"] % 100 == 0:
                print(f"\n----- ПРИМЕР #{stats['valid']} -----")
                print(f"Подготовлено {stats['valid']} примеров из {stats['total']} записей")
                print(f"Оригинальный текст: {original_text}...")
                print(f"Англицизмы: {anglicisms}")
                print(f"Отмеченный текст: {input_text}...")
                print("------------------------------\n")

                logger.info(f"Подготовлено {stats['valid']} примеров из {stats['total']} записей")
                logger.info(f"Пример #{stats['valid']}:")
                logger.info(f"Оригинальный текст: {original_text}...")
                logger.info(f"Англицизмы: {anglicisms}")
                logger.info(f"Отмеченный текст: {input_text}...")
                logger.info("------------------------------")

                # Сброс буфера вывода для немедленного отображения
                sys.stdout.flush()

        except Exception as e:
            print(f"Ошибка при обработке строки {index}: {e}")
            logger.error(f"Ошибка при обработке строки {index}: {e}")

    print(f"Подготовка завершена: всего {stats['valid']} примеров, {stats['anglicisms']} англицизмов")
    logger.info(f"Подготовка завершена: всего {stats['valid']} примеров, {stats['anglicisms']} англицизмов")
    return training_examples


# Функция для обучения модели
def train_model(args):
    """Обучает модель для замены англицизмов"""
    # Настройка путей
    setup_paths()

    # Пути к файлам и директориям
    dataset_path = os.path.join("assets", "anglicisms_dataset.csv")
    output_dir = os.path.join("assets", "trained_model")
    model_name = args.model_name

    # Проверяем существование датасета
    if not os.path.exists(dataset_path):
        logger.error(f"Датасет не найден: {dataset_path}")
        return None, None

    # Подготовка обучающих примеров
    logger.info("Подготовка обучающих данных...")
    training_examples = prepare_training_examples(dataset_path)

    if not training_examples:
        logger.error("Ошибка: датасет пуст или не содержит примеров с англицизмами")
        return None, None

    # Разделение на обучающую и валидационную выборки
    train_examples, eval_examples = train_test_split(
        training_examples, test_size=0.1, random_state=42
    )

    logger.info(f"Размер обучающей выборки: {len(train_examples)}")
    logger.info(f"Размер валидационной выборки: {len(eval_examples)}")

    # Конфигурация для загрузки модели в 4-битном формате (для экономии памяти)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logger.info(f"Загрузка модели и токенизатора {model_name}...")
    # Загрузка модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Добавляем специальные токены для обозначения англицизмов
    special_tokens = {
        "additional_special_tokens": [TOKEN_START_REPLACE, TOKEN_END_REPLACE]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Установка специальных токенов, если их нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Изменяем размер embedding для новых токенов
    model.resize_token_embeddings(len(tokenizer))

    # Подготовка модели для обучения в 4-битном режиме
    model = prepare_model_for_kbit_training(model)

    # Конфигурация LoRA для эффективной тонкой настройки
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Применение LoRA к модели
    model = get_peft_model(model, peft_config)

    # Создаем колбэк для вывода примеров во время обучения
    example_callback = ExampleOutputCallback(
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        display_freq=args.display_freq  # Показывать примеры каждые N шагов
    )

    # Создание объектов датасета
    train_dataset = AnglicismsDataset(train_examples, tokenizer)
    eval_dataset = AnglicismsDataset(eval_examples, tokenizer)

    # Конфигурация обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=25,  # Чаще логировать для мониторинга
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # Отключаем логирование в wandb и т.д.
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=42
    )

    # Создание коллатора данных
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Создание тренера с колбэком
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[example_callback]  # Добавляем наш колбэк
    )

    # Обучение модели
    logger.info("Начинаем обучение модели...")
    print("\n" + "=" * 70)
    print("НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ")
    print(f"Во время обучения будут выводиться примеры работы модели каждые {args.display_freq} шагов")
    print("=" * 70 + "\n")

    trainer.train()

    # Сохранение обученной модели и токенизатора
    logger.info(f"Сохранение обученной модели в {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Тестирование модели на нескольких примерах
    test_model(model, tokenizer, dataset_path)

    logger.info("Обучение завершено!")
    return model, tokenizer


# Функция для тестирования обученной модели
def test_model(model, tokenizer, dataset_path):
    """Тестирует обученную модель на примерах из датасета"""
    logger.info("Тестирование модели на примерах...")

    df = pd.read_csv(dataset_path)
    # Выбираем несколько случайных примеров для тестирования
    test_samples = df.sample(min(5, len(df))).reset_index(drop=True)

    # Модель для вычисления семантической близости
    similarity_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    results = []

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ")
    print("=" * 70 + "\n")

    for i, row in test_samples.iterrows():
        original_text = row['original_text']

        try:
            anglicisms = json.loads(row['anglicisms'])
        except (json.JSONDecodeError, TypeError):
            anglicisms = []

        if not anglicisms:
            logger.info(f"Пример {i + 1}: нет англицизмов для замены, пропускаем")
            continue

        # Создаем входной текст с отмеченными англицизмами
        input_text = original_text
        for word in anglicisms:
            pattern = r'\b' + re.escape(word) + r'\b'
            input_text = re.sub(pattern, f"{TOKEN_START_REPLACE} {word} {TOKEN_END_REPLACE}", input_text)

        # Добавляем инструкцию
        prompt = f"""Замените отмеченные англицизмы на подходящие русские синонимы, сохраняя смысл текста.
Англицизмы отмечены тегами <АНГЛ> и </АНГЛ>. Начало слово тегом <АНГЛ>, конец слова тегом </АНГЛ>.
УКАЗАННОЕ СЛОВО НУЖНО ЗАМЕНИТЬ ОБЯЗАТЕЛЬНО.

Текст с отмеченными англицизмами:
{input_text}

Преобразованный текст:
"""

        # Генерация замены англицизмов с помощью модели
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print(f"ТЕСТ #{i + 1}:")
        print(f"Оригинальный текст: {original_text}")
        print(f"Англицизмы: {anglicisms}")
        print("Генерация ответа...")

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    num_return_sequences=1
                )

            # Декодируем результат
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Извлекаем только преобразованный текст
            parts = generated_text.split("Преобразованный текст:")
            if len(parts) > 1:
                result_text = parts[1].strip()
            else:
                result_text = generated_text.strip()

            # Проверяем, были ли заменены все англицизмы
            replaced_count = 0
            for word in anglicisms:
                # Проверяем наличие слова в оригинальном и преобразованном тексте
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, original_text, re.IGNORECASE) and not re.search(pattern, result_text,
                                                                                      re.IGNORECASE):
                    replaced_count += 1

            replacement_rate = replaced_count / len(anglicisms) if anglicisms else 0

            # Вычисляем семантическую близость между оригинальным и преобразованным текстами
            original_embedding = similarity_model.encode([original_text])
            result_embedding = similarity_model.encode([result_text])

            similarity = util.pytorch_cos_sim(original_embedding, result_embedding).item()

            # Сохраняем результаты для этого примера
            result = {
                'original_text': original_text,
                'anglicisms': anglicisms,
                'result_text': result_text,
                'similarity': similarity,
                'replacement_rate': replacement_rate
            }

            results.append(result)

            # Выводим информацию о результате
            print(f"Сгенерированный текст: {result_text}")
            print(f"Семантическая близость: {similarity:.4f}")
            print(f"Заменено англицизмов: {replaced_count}/{len(anglicisms)} ({replacement_rate * 100:.1f}%)")
            print("-" * 70 + "\n")

        except Exception as e:
            print(f"Ошибка при обработке примера {i + 1}: {e}")
            logger.error(f"Ошибка при обработке примера {i + 1}: {e}")

    # Выводим примеры и общую статистику
    if results:
        # Вычисляем и выводим общую статистику
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        avg_replacement = sum(r['replacement_rate'] for r in results) / len(results)

        print("\n" + "=" * 70)
        print("ОБЩАЯ СТАТИСТИКА")
        print("=" * 70)
        print(f"Средняя семантическая близость: {avg_similarity:.4f}")
        print(f"Средний процент замененных англицизмов: {avg_replacement * 100:.1f}%")
        print("=" * 70 + "\n")

        logger.info(f"Средняя семантическая близость: {avg_similarity:.4f}")
        logger.info(f"Средний процент замененных англицизмов: {avg_replacement * 100:.1f}%")

    return results


# Основная функция
def main():
    parser = argparse.ArgumentParser(description="Обучение модели для замены англицизмов")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Имя или путь к предобученной модели")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Количество эпох обучения")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Только оценка модели без обучения")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Путь к ранее обученной модели для оценки")
    parser.add_argument("--display_freq", type=int, default=1,
                        help="Частота отображения примеров во время обучения (в шагах)")

    args = parser.parse_args()

    if args.evaluate_only and args.model_path:
        # Только оценка модели
        print(f"Загрузка модели из {args.model_path} для оценки...")
        logger.info(f"Загрузка модели из {args.model_path} для оценки...")

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            trust_remote_code=True
        )

        # Путь к датасету
        dataset_path = os.path.join("assets", "anglicisms_dataset.csv")

        # Оценка модели
        test_model(model, tokenizer, dataset_path)
    else:
        # Обучение модели
        train_model(args)


if __name__ == "__main__":
    main()
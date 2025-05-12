#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge import Rouge
import nltk
import os


def main():
    # Установка необходимых пакетов NLTK
    nltk.download('punkt')

    # Количество примеров
    N = 11

    # Путь к тестовым данным
    test_dataset_path = os.path.join('assets', 'llm_datasets', 'inf_dataset.csv')

    # Загрузка тестового датасета
    test_data = pd.read_csv(test_dataset_path)
    originals_df = test_data[['original']].head(N)
    references_df = test_data[['replaced']].head(N)
    original = originals_df['original'].tolist()
    references = references_df['replaced'].tolist()

    # Функция загрузки модели
    def load_model():
        model = AutoPeftModelForCausalLM.from_pretrained(
            "nata2627/angl_detection_tokenizer_qwen",
            device_map="auto",
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True
        )
        tokenizer = AutoTokenizer.from_pretrained("nata2627/angl_detection_tokenizer_qwen")
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return model, tokenizer

    # Загрузка модели и токенизатора
    print("Загрузка модели...")
    model, tokenizer = load_model()
    print("Модель загружена успешно!")

    # Генерация предсказаний с правильным форматированием промпта
    print("Генерация предсказаний...")
    predictions = []
    for i, input_text in enumerate(original):
        print(f"Обработка примера {i + 1}/{len(original)}")
        # Создаем инструкцию для модели (как во время обучения)
        instruction = (
            "Инструкция: Замените англицизмы в тексте на их русские аналоги.\n\n"
            f"Текст: {input_text}\n\n"
            f"Результат:"
        )

        # Форматируем промпт как во время обучения
        formatted_input = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        # Токенизация
        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

        # Генерация с теми же параметрами, что и в обучении
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Декодирование и извлечение только ответа ассистента
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_part = pred_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        predictions.append(assistant_part.strip())

    print("Предсказания сгенерированы!")

    # Метрики оценки
    print("Расчет метрик...")
    smoothie = SmoothingFunction().method4
    rouge = Rouge()
    metrics_table = []

    for pred, ref in zip(predictions, references):
        exact_match = int(pred == ref)
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = [word_tokenize(ref.lower())]
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        rouge_score = rouge.get_scores(pred, ref)[0]

        metrics_table.append({
            'prediction': pred,
            'reference': ref,
            'exact_match': exact_match,
            'bleu': bleu,
            'rouge-1': rouge_score['rouge-1']['f'],
            'rouge-2': rouge_score['rouge-2']['f'],
            'rouge-l': rouge_score['rouge-l']['f'],
            'prediction_length': len(pred.split()),
            'reference_length': len(ref.split())
        })

    # Сохранение и вывод результатов
    df = pd.DataFrame(metrics_table)
    df.to_csv('qwen_metrics_table.csv', index=False)

    summary_metrics = {
        'exact_match_accuracy': df['exact_match'].mean(),
        'bleu_score': df['bleu'].mean(),
        'rouge-1': df['rouge-1'].mean(),
        'rouge-2': df['rouge-2'].mean(),
        'rouge-l': df['rouge-l'].mean(),
        'avg_prediction_length': df['prediction_length'].mean(),
        'avg_reference_length': df['reference_length'].mean()
    }

    # Преобразуем сводные метрики в DataFrame
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_excel('qwen_summary_metrics.xlsx', index=False)

    print("\nТаблица метрик:")
    print(df)
    print("\nСводные метрики:")
    print(summary_metrics)
    print("\nРезультаты сохранены в файлы 'qwen_metrics_table.csv' и 'qwen_summary_metrics.xlsx'")


if __name__ == "__main__":
    # Проверка установки необходимых пакетов
    try:
        import rouge
    except ImportError:
        print("Установка пакета rouge...")
        import pip

        pip.main(['install', 'rouge'])
        print("Пакет rouge установлен!")

    main()
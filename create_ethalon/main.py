import os
import csv
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(file_path):
    """Load the dataset from CSV file."""
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                # Очищаем текст от лишних кавычек в начале и конце, если они есть
                text = row[0]
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]

                try:
                    # Обрабатываем список англицизмов
                    anglicisms_str = row[1]
                    anglicisms = json.loads(anglicisms_str)

                    # Проверяем, что это действительно список
                    if isinstance(anglicisms, list):
                        dataset.append((text, anglicisms))
                    else:
                        print(f"Warning: Anglicisms not in list format: {anglicisms_str}")
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse anglicisms in row: {row}")

    print(f"Successfully loaded {len(dataset)} rows from dataset")
    return dataset


def create_prompt(text, anglicisms):
    """Create a prompt for the model to replace anglicisms."""
    system_prompt = """Ты эксперт по русскому языку и замене англицизмов. Твоя задача - заменить ТОЛЬКО указанные англицизмы в тексте на их точные русские эквиваленты. Следуй строгой инструкции:

1. Замени ТОЛЬКО те слова, которые указаны в списке англицизмов, в точной форме, в которой они указаны
2. Сохрани ТОЧНО такую же структуру предложения
3. Не изменяй пунктуацию
4. Сохрани регистр букв (заглавные/строчные)
5. Сохрани все числа и имена собственные в их оригинальной форме
6. Если нужно заменить слово, используй точное русское слово, соответствующее значению:
   - субсидия → дотация или финансовая поддержка
   - автомобиль → машина
   - тариф → пошлина или сбор

7. Верни ТОЛЬКО измененный текст без каких-либо объяснений, комментариев или дополнительного текста
8. Не добавляй "Текст:", "Ответ:" или другие маркеры в начало или конец текста
9. Если слово указано как англицизм, но не встречается в точной форме в тексте, НЕ ИЗМЕНЯЙ текст
10. Не заменяй валюты, имена, страны, или названия."""

    user_prompt = f"Оригинальный текст: {text}\n\nТочные англицизмы для замены: {', '.join(anglicisms)}\n\nЗамени ТОЛЬКО эти англицизмы на русские эквиваленты и верни только измененный текст без дополнительных пояснений."

    return system_prompt, user_prompt


def replace_anglicisms(text, anglicisms, model, tokenizer, device):
    """Replace anglicisms in the text using the model."""
    system_prompt, user_prompt = create_prompt(text, anglicisms)

    # Формирование сообщений согласно новому формату
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Применение шаблона чата
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Токенизация входных данных
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # Генерация ответа
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.1,  # Снижаем температуру для более точных ответов
            top_p=0.9,
            do_sample=True
        )

    # Выделение только сгенерированной части (без входного текста)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Декодирование ответа
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Проверим, не содержит ли ответ маркеры "Текст:" или другие инструкции
    if "Текст:" in response or "Англицизмы для замены:" in response:
        # Если содержит, попробуем очистить ответ и взять только измененный текст
        lines = response.strip().split("\n")
        for line in lines:
            if not line.startswith("Текст:") and not line.startswith("Англицизмы") and not line.startswith("Замени"):
                response = line.strip()
                break

    return response


def verify_replacement(original_text, anglicisms, replaced_text):
    """Verify if the replacement was actually done and looks reasonable."""
    if original_text == replaced_text:
        # Проверим, содержит ли текст хотя бы один из англицизмов
        # Если содержит, но замена не произошла - это проблема
        for anglicism in anglicisms:
            if anglicism.lower() in original_text.lower():
                return False, "No replacement occurred for existing anglicisms"

    # Проверка на случай, если текст содержит инструкции
    if "Текст:" in replaced_text or "Англицизмы" in replaced_text:
        return False, "Text contains prompt instructions"

    return True, ""


def process_dataset(dataset, model, tokenizer, device):
    """Process the entire dataset."""
    processed_data = []
    issues_count = 0

    # Правильная инициализация tqdm с общим количеством элементов и описанием
    progress_bar = tqdm(
        total=len(dataset),
        desc="Processing dataset",
        unit="example",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for i, (text, anglicisms) in enumerate(dataset):
        try:
            # Попытка замены англицизмов
            replaced_text = replace_anglicisms(text, anglicisms, model, tokenizer, device)

            # Проверка замены
            is_valid, issue = verify_replacement(text, anglicisms, replaced_text)
            if not is_valid:
                issues_count += 1
                print(f"Issue with item {i + 1}: {issue}")

            processed_data.append((text, json.dumps(anglicisms, ensure_ascii=False), replaced_text))

        except Exception as e:
            print(f"Error processing item {i + 1}: {e}")
            processed_data.append((text, json.dumps(anglicisms, ensure_ascii=False), "ERROR"))

        # Обновляем прогресс-бар после каждого обработанного элемента
        progress_bar.update(1)

    # Закрываем прогресс-бар
    progress_bar.close()
    print(f"Processing completed. Found {issues_count} potential issues.")
    return processed_data


def save_dataset(data, output_path):
    """Save the processed dataset to CSV file."""
    # Сначала записываем заголовок, затем данные
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        # Записываем заголовок отдельно
        writer.writerow(["Original Text", "Anglicisms", "Replaced Text"])

        # Записываем данные
        for row in data:
            writer.writerow(row)

    print(f"Dataset saved to {output_path} with {len(data)} rows")

    # Проверка сохраненного файла
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size} bytes")

        # Проверка содержимого файла
        with open(output_path, 'r', encoding='utf-8') as f:
            first_lines = [next(f) for _ in range(min(5, len(data) + 1))]  # +1 для заголовка

        print("First few lines of the saved file:")
        for line in first_lines:
            print(line.strip())


def main():
    # Paths
    input_path = "assets/anglicisms_dataset.csv"
    output_path = "assets/y_dataset.csv"

    # Ensure directories exist
    ensure_dir('assets')

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(input_path)
    print(f"Loaded {len(dataset)} examples.")

    # Process dataset
    print("Processing dataset...")
    processed_data = process_dataset(dataset, model, tokenizer, device)

    # Save results
    print("Saving results...")
    save_dataset(processed_data, output_path)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
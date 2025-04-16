import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def parse_anglicisms(file_path, cfg=None):

    # Загрузка настроек паттернов из конфигурации
    language_section_pattern = r'== Из (.*?) =='
    anglicism_pattern = r'\[\[(.*?)\]\](.*?(?=\[\[|$))'
    through_english_pattern = r'через англ'

    if cfg is not None and hasattr(cfg, 'patterns'):
        language_section_pattern = cfg.patterns.language_section
        anglicism_pattern = cfg.patterns.anglicism
        through_english_pattern = cfg.patterns.through_english

    # Чтение файла
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        logger.error(f"Файл не найден: {file_path}")
        return {"by_language": {}, "all_anglicisms": []}
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        return {"by_language": {}, "all_anglicisms": []}

    # Словарь для хранения англицизмов по языкам происхождения
    anglicisms_by_language = defaultdict(list)

    # Текущий раздел языка
    current_language = None

    # Обработка строк файла
    for line in content.split('\n'):
        # Проверка, является ли строка заголовком раздела
        language_match = re.search(language_section_pattern, line)
        if language_match:
            current_language = language_match.group(1)
            continue

        # Если текущий язык определен, ищем англицизмы
        if current_language:
            matches = re.findall(anglicism_pattern, line)
            for match in matches:
                word = match[0]
                description = match[1].strip()

                # Проверяем, содержит ли описание упоминание английского языка
                is_through_english = bool(re.search(through_english_pattern, description))

                # Если слово содержит |, берем только часть до |
                if '|' in word:
                    word = word.split('|')[0]

                # Очистка названия языка от викиразметки
                clean_language = clean_wiki_markup(current_language)

                anglicisms_by_language[clean_language].append({
                    'word': word.strip(),
                    'description': description,
                    'through_english': is_through_english
                })

    # Создаем общий список всех англицизмов
    all_anglicisms = []
    for language, words in anglicisms_by_language.items():
        for word_info in words:
            all_anglicisms.append({
                'word': word_info['word'],
                'origin_language': language,
                'description': word_info['description'],
                'through_english': word_info['through_english']
            })

    return {
        'by_language': anglicisms_by_language,
        'all_anglicisms': all_anglicisms
    }


def clean_wiki_markup(text):
    # Удаление двойных квадратных скобок и содержимого между вертикальной чертой и закрывающими скобками
    text = re.sub(r'\[\[([^|]+)\|[^\]]+\]\]', r'\1', text)

    # Удаление двойных квадратных скобок без вертикальной черты
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Удаление одиночных квадратных скобок
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)

    # Удаление других возможных элементов разметки
    text = re.sub(r'<[^>]+>', '', text)

    return text
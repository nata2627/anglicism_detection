import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict


def get_letter_urls():
    """Возвращает словарь соответствий русских букв и URL-частей для сайта."""
    return {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
        'Е': 'E', 'Ё': 'Jo', 'Ж': 'Zh', 'З': 'Z', 'И': 'I',
        'Й': 'J', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
        'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
        'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'Tc', 'Ч': 'Ch',
        'Ш': 'Sh', 'Щ': 'Shh', 'Ъ': 'Tvz', 'Ы': 'Y', 'Ь': 'Myz',
        'Э': 'Eu', 'Ю': 'Iu', 'Я': 'Ia'
    }


def parse_web_anglicisms(base_url="http://anglicismdictionary.ru/"):
    """
    Парсит англицизмы с веб-сайта.

    Args:
        base_url: Базовый URL сайта с англицизмами.

    Returns:
        dict: Содержит списки англицизмов, структурированные по буквам
              и общий список всех англицизмов.
    """
    letter_urls = get_letter_urls()
    results = []
    anglicisms_by_letter = defaultdict(list)

    for rus_letter, url_part in letter_urls.items():
        url = base_url + url_part
        print(f"Парсим {rus_letter} -> {url}")

        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Не удалось загрузить {url}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='windows-1251')

            for p in soup.find_all('p', style="text-align: justify;"):
                text = p.get_text(strip=True)

                # Ищем первое слово перед скобкой или пробелом
                match = re.match(r'^([^\(\s]+)', text)
                if not match:
                    continue

                word = match.group(1).strip()
                definition = text.replace(word, "", 1).strip()

                entry = {
                    'letter': rus_letter,
                    'word': word,
                    'definition': definition
                }

                results.append(entry)
                anglicisms_by_letter[rus_letter].append(entry)

        except Exception as e:
            print(f"Ошибка при обработке {url}: {e}")

    return {
        'by_letter': dict(anglicisms_by_letter),
        'all_anglicisms': results
    }


def extract_word_features(text):
    """
    Извлекает особенности из определения англицизма.
    Может использоваться для дополнительного анализа.

    Args:
        text: Текст определения

    Returns:
        dict: Словарь с извлеченными особенностями
    """
    features = {}

    # Попытка найти язык происхождения (если указан)
    origin_match = re.search(r'от\s+([а-яА-Я]+\.)', text)
    if origin_match:
        features['possible_origin'] = origin_match.group(1)

    # Поиск года или периода (если указаны)
    year_match = re.search(r'(\d{4}|\d{2}-е)', text)
    if year_match:
        features['year_mention'] = year_match.group(1)

    return features
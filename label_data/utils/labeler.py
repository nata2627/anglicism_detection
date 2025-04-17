import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Set, Optional
from tqdm import tqdm
import re
import spacy
import json

from label_data.utils.logger import CustomLogger
from label_data.utils.analyzer import analyze_anglicisms


class AngliсismLabeler:
    """
    Класс для разметки англицизмов в текстах.

    Выполняет разметку текстов на основе предоставленного списка англицизмов.
    Осуществляет как бинарную классификацию (0/1), так и последовательную
    разметку в формате BIO (Begin-Inside-Outside).

    Attributes:
        config: Конфигурационный объект с настройками разметки
        logger: Объект для логирования процесса разметки
        anglicisms_set: Множество англицизмов для поиска в текстах
        texts_df: DataFrame с текстами для разметки
        nlp: Объект spaCy для обработки текстов
        countries: Множество названий стран для исключения из разметки
    """

    def __init__(self, config) -> None:
        """
        Инициализация разметчика англицизмов.

        Args:
            config: Объект конфигурации, содержащий настройки разметки
        """
        self.config = config
        self.logger = CustomLogger(config)
        self.anglicisms_set = set()
        self.texts_df = None
        self.countries = self._initialize_countries()
        self.nlp = self._initialize_spacy()

    def _initialize_spacy(self) -> spacy.language.Language:
        """
        Инициализирует модель spaCy для русского языка.

        Returns:
            spacy.language.Language: Загруженная модель spaCy
        """
        try:
            self.logger.info("Загрузка spaCy модели для русского языка")
            nlp = spacy.load('ru_core_news_sm')
            self.logger.info("Модель spaCy успешно загружена")
            return nlp
        except:
            # Если модель не установлена, пробуем её скачать
            import subprocess
            self.logger.warning("Модель spaCy не найдена. Выполняется установка...")
            try:
                subprocess.run(["python", "-m", "spacy", "download", "ru_core_news_sm"])
                nlp = spacy.load('ru_core_news_sm')
                self.logger.info("Модель spaCy успешно установлена и загружена")
                return nlp
            except Exception as e:
                self.logger.error(f"Ошибка при установке модели spaCy: {str(e)}")
                raise

    def label_data(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Выполняет разметку англицизмов в текстах.

        Осуществляет бинарную классификацию и BIO-разметку англицизмов
        в текстах на основе предоставленного списка.

        Returns:
            Tuple[List[Dict], List[Dict], Dict]:
                - Результаты бинарной разметки
                - Результаты BIO разметки
                - Статистика разметки
        """
        self.logger.info("Начало процесса разметки англицизмов")

        # Проверяем, что данные загружены
        if self.texts_df is None or self.anglicisms_set is None:
            self.logger.error("Данные не загружены. Сначала выполните метод load_data()")
            raise ValueError("Данные не загружены")

        # Подготавливаем списки для результатов
        binary_results = []  # для бинарной классификации
        bio_results = []  # для BIO разметки

        # Статистика для анализа
        stats = {
            'total_tokens': 0,
            'total_anglicisms': 0,
            'texts_with_anglicisms': 0,
            'anglicism_frequency': {},  # частота каждого англицизма
            'texts_anglicism_count': [],  # кол-во англицизмов в каждом тексте
            'texts_anglicism_percent': []  # процент англицизмов в каждом тексте
        }

        # Обработка каждого текста
        self.logger.info(f"Разметка {len(self.texts_df)} текстов")
        for idx, row in tqdm(self.texts_df.iterrows(), total=len(self.texts_df), desc="Разметка текстов"):
            # Получаем токенизированный текст
            text = row[self.config.labeler.tokenized_column]

            # Проверяем, что текст не пустой
            if pd.isna(text) or text == "":
                binary_results.append({'text_id': idx, 'tokens': [], 'labels': []})
                bio_results.append({'text_id': idx, 'tokens': [], 'labels': []})
                stats['texts_anglicism_count'].append(0)
                stats['texts_anglicism_percent'].append(0)
                continue

            # Токенизируем текст (предполагаем, что токены разделены пробелами)
            tokens = text.split()

            # Обрабатываем текст через spaCy для определения частей речи
            doc = self.nlp(" ".join(tokens))

            # Создаем словарь для быстрой проверки, является ли слово именем собственным
            is_propn = {token.text.lower(): token.pos_ == 'PROPN' for token in doc}

            # Также проверяем начинается ли слово с заглавной буквы
            is_capitalized = {}
            for token in doc:
                if len(token.text) > 0:
                    is_capitalized[token.text.lower()] = token.text[0].isupper()

            # Бинарная разметка (0 - не англицизм, 1 - англицизм)
            binary_labels = []

            # BIO разметка (O - не англицизм, B-ANG - начало англицизма, I-ANG - продолжение англицизма)
            bio_labels = []

            # Счетчик англицизмов в тексте
            text_anglicism_count = 0

            # Разметка токенов
            for token in tokens:
                token_lower = token.lower()
                stats['total_tokens'] += 1

                # Проверяем условия для определения англицизма
                is_anglicism = False

                if token_lower in self.anglicisms_set:  # Слово есть в списке англицизмов
                    # Проверяем исключения
                    excluded = False

                    # Исключаем имена собственные, если это указано в конфигурации
                    if self.config.labeler.exclude_proper_nouns:
                        if token_lower in is_propn and is_propn[token_lower]:
                            excluded = True

                    # Исключаем названия стран, если это указано в конфигурации
                    if self.config.labeler.exclude_country_names:
                        if token_lower in self.countries:
                            excluded = True

                    if not excluded:
                        is_anglicism = True

                # Обновляем статистику и метки
                if is_anglicism:
                    stats['total_anglicisms'] += 1
                    text_anglicism_count += 1
                    binary_labels.append(1)  # 1 - англицизм

                    # Определяем BIO метку
                    # Если предыдущий токен не был англицизмом, то это B-ANG
                    if len(bio_labels) == 0 or bio_labels[-1] == 'O':
                        bio_labels.append('B-ANG')
                    else:
                        # Иначе это продолжение последовательности - I-ANG
                        bio_labels.append('I-ANG')

                    # Подсчет частоты каждого англицизма
                    stats['anglicism_frequency'][token_lower] = stats['anglicism_frequency'].get(token_lower, 0) + 1
                else:
                    binary_labels.append(0)  # 0 - не англицизм
                    bio_labels.append('O')  # O - не англицизм

            # Добавляем результаты разметки для этого текста
            binary_results.append({
                'text_id': idx,
                'tokens': tokens,
                'labels': binary_labels
            })

            bio_results.append({
                'text_id': idx,
                'tokens': tokens,
                'labels': bio_labels
            })

            # Обновляем статистику по тексту
            stats['texts_anglicism_count'].append(text_anglicism_count)

            # Рассчитываем процент англицизмов в тексте
            tokens_count = len(tokens)
            if tokens_count > 0:
                stats['texts_anglicism_percent'].append(text_anglicism_count / tokens_count * 100)
            else:
                stats['texts_anglicism_percent'].append(0)

            # Обновляем счетчик текстов с англицизмами
            if text_anglicism_count > 0:
                stats['texts_with_anglicisms'] += 1

        # Рассчитываем итоговую статистику
        stats['total_texts'] = len(self.texts_df)
        stats['anglicism_percentage'] = (stats['total_anglicisms'] / stats['total_tokens'] * 100) if stats[
                                                                                                         'total_tokens'] > 0 else 0
        stats['texts_with_anglicisms_percentage'] = (stats['texts_with_anglicisms'] / stats['total_texts'] * 100) if \
        stats['total_texts'] > 0 else 0

        # Сохраняем результаты разметки
        self._save_results(binary_results, bio_results, stats)

        self.logger.info("Разметка англицизмов завершена")
        self.logger.info(f"Всего токенов: {stats['total_tokens']}")
        self.logger.info(f"Всего англицизмов: {stats['total_anglicisms']}")
        self.logger.info(f"Процент англицизмов: {stats['anglicism_percentage']:.2f}%")
        self.logger.info(
            f"Количество текстов с англицизмами: {stats['texts_with_anglicisms']} из {stats['total_texts']} ({stats['texts_with_anglicisms_percentage']:.2f}%)")

        return binary_results, bio_results, stats

    def _save_results(self, binary_results, bio_results, stats) -> None:
        """
        Сохраняет результаты разметки в файлы.

        Args:
            binary_results: Результаты бинарной разметки
            bio_results: Результаты BIO разметки
            stats: Статистика разметки
        """
        output_dir = Path(self.config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Подготавливаем данные для сохранения в CSV
        binary_data = []
        bio_data = []

        for br, bior in zip(binary_results, bio_results):
            text_id = br['text_id']
            tokens = br['tokens']
            binary_labels = br['labels']
            bio_labels = bior['labels']

            for token, bl, biol in zip(tokens, binary_labels, bio_labels):
                binary_data.append({
                    'text_id': text_id,
                    'token': token,
                    'is_anglicism': bl
                })

                bio_data.append({
                    'text_id': text_id,
                    'token': token,
                    'label': biol
                })

        # Сохраняем бинарную разметку в CSV
        binary_df = pd.DataFrame(binary_data)
        binary_path = output_dir / f"{self.config.labeler.output_prefix}_binary.csv"
        binary_df.to_csv(binary_path, index=False, encoding=self.config.output.encoding)
        self.logger.info(f"Бинарная разметка сохранена в: {binary_path}")

        # Сохраняем BIO разметку в CSV
        bio_df = pd.DataFrame(bio_data)
        bio_path = output_dir / f"{self.config.labeler.output_prefix}_bio.csv"
        bio_df.to_csv(bio_path, index=False, encoding=self.config.output.encoding)
        self.logger.info(f"BIO разметка сохранена в: {bio_path}")

        # Сохраняем статистику в JSON
        stats_path = output_dir / f"{self.config.labeler.output_prefix}_stats.json"

        # Преобразуем словарь частот англицизмов в список для JSON
        freq_list = [{"anglicism": k, "frequency": v} for k, v in stats['anglicism_frequency'].items()]

        # Создаем копию статистики с преобразованными данными для JSON
        json_stats = {
            'total_tokens': stats['total_tokens'],
            'total_anglicisms': stats['total_anglicisms'],
            'total_texts': stats['total_texts'],
            'texts_with_anglicisms': stats['texts_with_anglicisms'],
            'anglicism_percentage': stats['anglicism_percentage'],
            'texts_with_anglicisms_percentage': stats['texts_with_anglicisms_percentage'],
            'anglicism_frequency': freq_list,
            'texts_statistics': [
                {
                    'text_id': i,
                    'anglicism_count': count,
                    'anglicism_percent': percent
                }
                for i, (count, percent) in
                enumerate(zip(stats['texts_anglicism_count'], stats['texts_anglicism_percent']))
            ]
        }

        with open(stats_path, 'w', encoding=self.config.output.encoding) as f:
            json.dump(json_stats, f, ensure_ascii=False, indent=4)

        self.logger.info(f"Статистика разметки сохранена в: {stats_path}")

    def analyze_results(self, binary_results, stats, output_dir) -> None:
        """
        Анализирует и визуализирует результаты разметки.

        Args:
            binary_results: Результаты бинарной разметки
            stats: Статистика разметки
            output_dir: Директория для сохранения результатов анализа
        """
        self.logger.info("Начало анализа и визуализации результатов")

        # Создаем директорию для результатов, если она не существует
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Запускаем анализ и визуализацию результатов
        analyze_anglicisms(binary_results, stats, output_dir, self.config)

        self.logger.info(f"Анализ и визуализация результатов сохранены в: {output_dir}")

    def _initialize_countries(self) -> Set[str]:
        """
        Инициализирует множество названий стран для исключения из разметки.

        Returns:
            Set[str]: Множество названий стран и географических объектов
        """
        # Список стран и географических объектов для исключения из разметки
        return set([
            'россия', 'сша', 'америка', 'англия', 'великобритания', 'канада',
            'австралия', 'германия', 'франция', 'италия', 'испания', 'китай',
            'япония', 'индия', 'бразилия', 'мексика', 'европа', 'азия',
            'африка', 'москва', 'лондон', 'нью-йорк', 'париж', 'берлин', 'рим',
            'токио', 'пекин', 'вашингтон', 'нью-джерси', 'калифорния',
            'флорида', 'техас', 'чикаго', 'бостон', 'сибирь', 'урал',
            'петербург', 'санкт-петербург'
        ])

    def load_data(self) -> None:
        """
        Загружает данные из файлов: список англицизмов и тексты для разметки.

        Raises:
            FileNotFoundError: Если входные файлы не найдены
            ValueError: При проблемах с форматом входных данных
        """
        # Загрузка списка англицизмов
        anglicisms_path = os.path.join(self.config.paths.data_dir, self.config.labeler.anglicisms_file)
        self.logger.info(f"Загрузка списка англицизмов из: {anglicisms_path}")

        try:
            with open(anglicisms_path, 'r', encoding=self.config.output.encoding) as f:
                anglicisms = [line.strip().lower() for line in f if line.strip()]
            self.anglicisms_set = set(anglicisms)
            self.logger.info(f"Загружено {len(self.anglicisms_set)} уникальных англицизмов")
        except FileNotFoundError:
            self.logger.error(f"Файл со списком англицизмов не найден: {anglicisms_path}")
            raise
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке списка англицизмов: {str(e)}")
            raise

        # Загрузка текстов для разметки
        texts_path = os.path.join(self.config.paths.data_dir, self.config.labeler.texts_file)
        self.logger.info(f"Загрузка текстов из: {texts_path}")

        try:
            self.texts_df = pd.read_csv(texts_path, encoding=self.config.output.encoding)

            # Проверка наличия необходимой колонки
            if self.config.labeler.tokenized_column not in self.texts_df.columns:
                self.logger.error(f"В файле {texts_path} отсутствует колонка '{self.config.labeler.tokenized_column}'")
                raise ValueError(f"Колонка '{self.config.labeler.tokenized_column}' не найдена в данных")

            self.logger.info(f"Загружено {len(self.texts_df)} текстов")
        except FileNotFoundError:
            self.logger.error(f"Файл с текстами не найден: {texts_path}")
            raise
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке текстов: {str(e)}")
            raise
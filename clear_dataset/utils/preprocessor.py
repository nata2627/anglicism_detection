import os
import re
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Optional, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem

# Импортируем компоненты Natasha для распознавания именованных сущностей
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)

# Импортируем дополнительный словарь именованных сущностей
from clear_dataset.utils.manual_ner_removal import COMMON_NAMED_ENTITIES, extend_named_entities

# Импортируем кастомный логгер
from clear_dataset.utils.logger import CustomLogger


class TextPreprocessor:
    """
    Класс для предобработки текстовых данных из CSV файлов.

    Выполняет загрузку CSV-файлов с текстами, их очистку,
    токенизацию и лемматизацию, а также сохранение результатов
    в новый CSV-файл. Теперь включает функционал для распознавания
    и удаления именованных сущностей с помощью Natasha.

    Attributes:
        config: Конфигурационный объект с настройками предобработки
        logger: Объект для логирования процесса предобработки
        mystem: Лемматизатор для русского языка
        russian_stopwords: Набор стоп-слов для русского языка
        word_tokenizer: Токенизатор для разделения текста на отдельные слова
        ner_components: Компоненты Natasha для распознавания именованных сущностей
    """

    def __init__(self, config) -> None:
        """
        Инициализация предобработчика текстов.

        Args:
            config: Объект конфигурации, содержащий настройки предобработки
        """
        self.config = config
        self.logger = CustomLogger(config)

        # Скачиваем необходимые ресурсы для NLTK
        try:
            self.logger.info("Инициализация ресурсов NLTK")
            nltk.download('stopwords', quiet=True)

            # Инициализация лемматизатора
            self.mystem = Mystem()

            # Загружаем стоп-слова
            self.russian_stopwords = set(stopwords.words('russian'))

            # Инициализируем токенизатор
            self.word_tokenizer = RegexpTokenizer(r'\w+')

            self.logger.info("Ресурсы NLTK успешно инициализированы")

            # Инициализация компонентов Natasha для NER
            self.logger.info("Инициализация компонентов Natasha для распознавания именованных сущностей")

            self.ner_components = {
                'segmenter': Segmenter(),
                'morph_vocab': MorphVocab(),
                'emb': NewsEmbedding(),
            }

            # Загружаем NER-таггер
            self.ner_components['ner_tagger'] = NewsNERTagger(self.ner_components['emb'])

            self.logger.info("Компоненты Natasha успешно инициализированы")

        except Exception as e:
            self.logger.error(f"Ошибка при инициализации ресурсов: {str(e)}")
            raise

    def _find_input_files(self) -> List[str]:
        """
        Находит входные CSV-файлы по шаблону имени.

        Returns:
            List[str]: Список путей к найденным файлам
        """
        pattern = os.path.join(self.config.paths.data_dir, self.config.processor.file_pattern)
        files = glob.glob(pattern)

        if not files:
            self.logger.warning(f"Не найдено файлов по шаблону: {pattern}")
        else:
            self.logger.info(f"Найдено {len(files)} файлов для обработки")
            for file in files:
                self.logger.debug(f"Найден файл: {file}")

        return files

    def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Загружает данные из CSV-файла.

        Args:
            file_path: Путь к CSV-файлу

        Returns:
            Optional[pd.DataFrame]: DataFrame с загруженными данными или None в случае ошибки
        """
        try:
            self.logger.info(f"Загрузка данных из файла: {file_path}")
            df = pd.read_csv(file_path, encoding=self.config.output.encoding)

            # Проверяем наличие необходимой колонки с текстом
            if self.config.processor.text_column not in df.columns:
                self.logger.error(
                    f"В файле {file_path} отсутствует колонка '{self.config.processor.text_column}'"
                )
                return None

            self.logger.info(f"Загружено {len(df)} строк из файла {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке файла {file_path}: {str(e)}")
            return None

    def _extract_named_entities(self, text: str) -> Set[str]:
        """
        Извлекает именованные сущности из текста с помощью Natasha.

        Args:
            text: Исходный текст

        Returns:
            Set[str]: Множество найденных именованных сущностей
        """
        if not text:
            return set()

        try:
            # Важно: Natasha лучше работает с текстом в оригинальном регистре,
            # поэтому используем текст до приведения к нижнему регистру
            # Создаем документ Natasha
            doc = Doc(text)

            # Сегментируем текст
            doc.segment(self.ner_components['segmenter'])

            # Применяем NER-таггер
            doc.tag_ner(self.ner_components['ner_tagger'])

            # Извлекаем найденные сущности
            entities = set()
            for span in doc.spans:
                # PER - персоны, LOC - локации, ORG - организации
                if span.type in ('PER', 'LOC', 'ORG'):
                    entities.add(span.text.lower())
                    # Также добавляем отдельные слова из многословных сущностей
                    for word in span.text.lower().split():
                        if len(word) > 2:  # Игнорируем короткие слова
                            entities.add(word)

            # Добавляем сущности из словаря
            text_lower = text.lower()
            entities = extend_named_entities(entities, text_lower)

            return entities

        except Exception as e:
            self.logger.warning(f"Ошибка при извлечении именованных сущностей: {str(e)}")
            return set()

    def _remove_named_entities(self, text: str, entities: Set[str]) -> str:
        """
        Удаляет именованные сущности из текста.

        Args:
            text: Исходный текст
            entities: Множество именованных сущностей для удаления

        Returns:
            str: Текст с удаленными именованными сущностями
        """
        if not text or not entities:
            return text

        # Создаем копию текста для обработки
        processed_text = text

        # Сортируем сущности по длине (от самых длинных к коротким)
        # для корректной замены вложенных сущностей
        sorted_entities = sorted(entities, key=len, reverse=True)

        # Заменяем каждую сущность на пробел
        for entity in sorted_entities:
            # Используем регулярное выражение с границами слов
            pattern = r'\b' + re.escape(entity) + r'\b'
            processed_text = re.sub(pattern, ' ', processed_text, flags=re.IGNORECASE)

        # Нормализуем пробелы
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        return processed_text

    def preprocess_corpus(self, texts: List[str]) -> Tuple[List[str], List[List[str]]]:
        """
        Выполняет предобработку корпуса текстов.

        Args:
            texts: Список текстов для предобработки

        Returns:
            Tuple[List[str], List[List[str]]]: Кортеж, содержащий предобработанные тексты
            и список токенизированных текстов (без именованных сущностей)
        """
        processed_texts = []
        tokenized_texts = []

        total_entities_removed = 0
        total_texts_with_entities = 0

        advanced = self.config.processor.advanced_preprocessing
        min_len = self.config.processor.min_token_length
        max_len = self.config.processor.max_token_length
        remove_entities = self.config.processor.get('remove_named_entities', True)

        self.logger.info(
            f"Начата предобработка {len(texts)} текстов (расширенная: {advanced}, удаление именованных сущностей: {remove_entities})")

        for text_idx, text in enumerate(tqdm(texts, desc="Предобработка текстов")):
            # Проверяем, что текст не None и не пустой
            if text is None or text == "":
                processed_texts.append("")
                tokenized_texts.append([])
                continue

            # Сохраняем оригинальный текст для NER
            original_text = text

            # Базовая предобработка
            text = text.lower()
            text = text.replace('\xa0', ' ')
            text = re.sub(r'[«»""„]', '"', text)
            text = re.sub(r'[''‚]', "'", text)

            # Удаление URL
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

            # Удаление email
            text = re.sub(r'\S+@\S+', '', text)

            # Удаление чисел
            text = re.sub(r'\d+', '', text)

            # Удаление специальных символов и пунктуации
            text = re.sub(r'[^\w\s]', ' ', text)

            # Нормализация пробелов
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            # Извлечение именованных сущностей с помощью Natasha для последующего удаления
            # Используем оригинальный текст (с сохранением регистра) для лучшего распознавания
            entities = set()
            if remove_entities:
                entities = self._extract_named_entities(original_text)
                if entities:
                    total_texts_with_entities += 1
                    total_entities_removed += len(entities)

                    # Выводим некоторые статистики для отладки
                    if text_idx % 100 == 0 or len(
                            entities) > 10:  # Выводим статистику каждые 100 текстов или при большом числе сущностей
                        self.logger.info(
                            f"Текст #{text_idx}: Найдено {len(entities)} сущностей: {', '.join(sorted(entities)[:10])}{'...' if len(entities) > 10 else ''}")

                        # Логируем до и после для наглядности
                        before_tokens = self.word_tokenizer.tokenize(text)
                        text = self._remove_named_entities(text, entities)
                        after_tokens = self.word_tokenizer.tokenize(text)
                        self.logger.info(
                            f"До: {' '.join(before_tokens[:30])}{'...' if len(before_tokens) > 30 else ''}")
                        self.logger.info(
                            f"После: {' '.join(after_tokens[:30])}{'...' if len(after_tokens) > 30 else ''}")
                    else:
                        text = self._remove_named_entities(text, entities)

            processed_texts.append(text)

            # Токенизация
            tokens = self.word_tokenizer.tokenize(text)

            # Фильтрация токенов по длине
            tokens = [token for token in tokens if min_len <= len(token) <= max_len]

            if advanced:
                # Удаление стоп-слов
                tokens = [token for token in tokens if token not in self.russian_stopwords]

                # Лемматизация для русских слов
                if tokens:  # Проверка на пустой список
                    lemmatized_text = ' '.join(tokens)
                    try:
                        lemmas = self.mystem.lemmatize(lemmatized_text)
                        tokens = [lemma.strip() for lemma in lemmas if lemma.strip()]
                    except Exception as e:
                        self.logger.warning(f"Ошибка при лемматизации: {str(e)}")

            tokenized_texts.append(tokens)

        # Выводим общую статистику по удаленным сущностям
        self.logger.info(
            f"Всего удалено {total_entities_removed} именованных сущностей в {total_texts_with_entities} текстах")
        self.logger.info(
            f"В среднем {total_entities_removed / max(1, total_texts_with_entities):.1f} сущностей на текст с сущностями")
        self.logger.info("Предобработка текстов завершена")

        return processed_texts, tokenized_texts

    def _save_results(self, df: pd.DataFrame) -> bool:
        """
        Сохраняет результаты предобработки в CSV-файл.

        Args:
            df: DataFrame с предобработанными данными

        Returns:
            bool: True, если сохранение успешно, иначе False
        """
        try:
            output_dir = Path(self.config.paths.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / self.config.output.file_name

            df.to_csv(output_path, index=False, encoding=self.config.output.encoding)

            self.logger.info(f"Результаты сохранены в файл: {output_path}")
            self.logger.info(f"Сохранено {len(df)} предобработанных текстов")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            return False

    def preprocess(self) -> Optional[pd.DataFrame]:
        """
        Основной метод для запуска процесса предобработки.

        Находит входные файлы, загружает данные, выполняет
        предобработку и сохраняет результаты.

        Returns:
            Optional[pd.DataFrame]: DataFrame с предобработанными данными или None в случае ошибки
        """
        # Находим входные файлы
        input_files = self._find_input_files()

        if not input_files:
            return None

        # Загружаем и объединяем данные из всех файлов
        all_data = []
        for file_path in input_files:
            df = self._load_data(file_path)
            if df is not None:
                all_data.append(df)

        if not all_data:
            self.logger.error("Не удалось загрузить данные ни из одного файла")
            return None

        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Всего загружено {len(combined_df)} строк из {len(all_data)} файлов")

        # Получаем тексты для предобработки
        texts = combined_df[self.config.processor.text_column].tolist()

        # Выполняем предобработку
        processed_texts, tokenized_texts = self.preprocess_corpus(texts)

        # Создаем новый DataFrame с результатами
        result_df = pd.DataFrame({
            'original_text': texts,
            'processed_text': processed_texts,
            'tokenized_text': [' '.join(tokens) for tokens in tokenized_texts],
            'tokens_count': [len(tokens) for tokens in tokenized_texts]
        })

        # Сохраняем результаты
        if self._save_results(result_df):
            return result_df
        else:
            return None
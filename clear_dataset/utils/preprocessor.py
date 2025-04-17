import os
import re
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem

# Импортируем кастомный логгер
from clear_dataset.utils.logger import CustomLogger


class TextPreprocessor:
    """
    Класс для предобработки текстовых данных из CSV файлов.

    Выполняет загрузку CSV-файлов с текстами, их очистку,
    токенизацию и лемматизацию, а также сохранение результатов
    в новый CSV-файл.

    Attributes:
        config: Конфигурационный объект с настройками предобработки
        logger: Объект для логирования процесса предобработки
        mystem: Лемматизатор для русского языка
        russian_stopwords: Набор стоп-слов для русского языка
        word_tokenizer: Токенизатор для разделения текста на отдельные слова
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
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации ресурсов NLTK: {str(e)}")
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

    def preprocess_corpus(self, texts: List[str]) -> Tuple[List[str], List[List[str]]]:
        """
        Выполняет предобработку корпуса текстов.

        Args:
            texts: Список текстов для предобработки

        Returns:
            Tuple[List[str], List[List[str]]]: Кортеж, содержащий предобработанные тексты
            и список токенизированных текстов
        """
        processed_texts = []
        tokenized_texts = []

        advanced = self.config.processor.advanced_preprocessing
        min_len = self.config.processor.min_token_length
        max_len = self.config.processor.max_token_length

        self.logger.info(f"Начата предобработка {len(texts)} текстов (расширенная: {advanced})")

        for text in tqdm(texts, desc="Предобработка текстов"):
            # Проверяем, что текст не None и не пустой
            if text is None or text == "":
                processed_texts.append("")
                tokenized_texts.append([])
                continue

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
import os
import pandas as pd
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional

# Импортируем кастомный логгер
from label_data.utils.logger import CustomLogger


class DataLabeler:
    """
    Класс для маркировки данных, определяющий англицизмы в корпусе текстов.

    Загружает токенизированные тексты и список англицизмов, затем формирует
    словарь уникальных слов с соответствующими метками.

    Attributes:
        config: Конфигурационный объект с настройками маркировки
        logger: Объект для логирования процесса маркировки
        anglicisms: Множество слов-англицизмов для сравнения
    """

    def __init__(self, config) -> None:
        """
        Инициализация модуля маркировки данных.

        Args:
            config: Объект конфигурации, содержащий настройки маркировки
        """
        self.config = config
        self.logger = CustomLogger(config, logger_name="DataLabeler")
        self.anglicisms = set()

    def _load_anglicisms(self) -> Set[str]:
        """
        Загружает список англицизмов из файла.

        Returns:
            Set[str]: Множество англицизмов

        Raises:
            FileNotFoundError: Если файл с англицизмами не найден
        """
        try:
            anglicisms_file = os.path.join(
                self.config.paths.data_dir,
                self.config.processor.input_anglicisms_file
            )

            self.logger.info(f"Загрузка англицизмов из файла: {anglicisms_file}")

            # Проверяем наличие файла
            if not os.path.exists(anglicisms_file):
                self.logger.error(f"Файл с англицизмами не найден: {anglicisms_file}")
                return set()

            # Загружаем англицизмы
            with open(anglicisms_file, 'r', encoding=self.config.output.encoding) as f:
                anglicisms = {line.strip().lower() for line in f if line.strip()}

            self.logger.info(f"Загружено {len(anglicisms)} англицизмов")
            return anglicisms

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке англицизмов: {str(e)}")
            return set()

    def _load_texts(self) -> Optional[pd.DataFrame]:
        """
        Загружает токенизированные тексты из CSV-файла.

        Returns:
            Optional[pd.DataFrame]: DataFrame с текстами или None в случае ошибки

        Raises:
            FileNotFoundError: Если файл с текстами не найден
        """
        try:
            texts_file = os.path.join(
                self.config.paths.data_dir,
                self.config.processor.input_texts_file
            )

            self.logger.info(f"Загрузка текстов из файла: {texts_file}")

            # Проверяем наличие файла
            if not os.path.exists(texts_file):
                self.logger.error(f"Файл с текстами не найден: {texts_file}")
                return None

            # Загружаем тексты
            df = pd.read_csv(texts_file, encoding=self.config.output.encoding)

            # Проверяем наличие нужной колонки
            if self.config.processor.text_column not in df.columns:
                self.logger.error(
                    f"В файле отсутствует колонка '{self.config.processor.text_column}'"
                )
                return None

            self.logger.info(f"Загружено {len(df)} текстов")
            return df

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке текстов: {str(e)}")
            return None

    def _extract_unique_words(self, df: pd.DataFrame) -> List[str]:
        """
        Извлекает уникальные слова из токенизированных текстов.

        Args:
            df: DataFrame с текстами

        Returns:
            List[str]: Список уникальных слов
        """
        try:
            # Получаем колонку с текстами
            text_column = self.config.processor.text_column

            # Объединяем все тексты в один список
            all_tokens = []
            for text in df[text_column]:
                if isinstance(text, str):
                    # Разделяем строку на токены по пробелам
                    tokens = text.split()
                    all_tokens.extend(tokens)

            # Получаем уникальные токены
            unique_words = sorted(set(all_tokens))

            self.logger.info(
                f"Извлечено {len(unique_words)} уникальных слов из {len(all_tokens)} токенов"
            )

            return unique_words

        except Exception as e:
            self.logger.error(f"Ошибка при извлечении уникальных слов: {str(e)}")
            return []

    def _label_words(self, unique_words: List[str]) -> Dict[str, int]:
        """
        Маркирует слова (1 - англицизм, 0 - не англицизм).

        Args:
            unique_words: Список уникальных слов для маркировки

        Returns:
            Dict[str, int]: Словарь слов с метками
        """
        labeled_words = {}

        for word in unique_words:
            # Для сравнения в одном регистре, если настроено
            if not self.config.processor.case_sensitive:
                compare_word = word.lower()
            else:
                compare_word = word

            # Помечаем слова
            if compare_word in self.anglicisms:
                labeled_words[word] = 1
            else:
                labeled_words[word] = 0

        # Считаем количество англицизмов
        anglicism_count = sum(1 for label in labeled_words.values() if label == 1)
        self.logger.info(
            f"Размечено {len(labeled_words)} слов, из них англицизмов: {anglicism_count}"
        )

        return labeled_words

    def _save_results(self, labeled_words: Dict[str, int]) -> Tuple[bool, str]:
        """
        Сохраняет результаты маркировки в CSV-файл.

        Args:
            labeled_words: Словарь слов с метками

        Returns:
            Tuple[bool, str]: Кортеж с флагом успеха и путем к файлу

        Raises:
            Exception: В случае ошибки при сохранении
        """
        try:
            # Создаем директорию, если её нет
            output_dir = Path(self.config.paths.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Формируем путь к файлу
            output_path = output_dir / self.config.output.file_name

            # Создаем DataFrame из словаря
            result_df = pd.DataFrame({
                'word': list(labeled_words.keys()),
                'is_anglicism': list(labeled_words.values())
            })

            # Сохраняем в CSV
            result_df.to_csv(
                output_path,
                index=False,
                encoding=self.config.output.encoding
            )

            self.logger.info(f"Результаты сохранены в файл: {output_path}")
            self.logger.info(f"Сохранено {len(result_df)} слов с метками")

            return True, str(output_path)

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
            return False, ""

    def process(self) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Основной метод для выполнения всего процесса маркировки.

        Returns:
            Tuple[bool, Optional[pd.DataFrame]]: Кортеж с флагом успеха и
            DataFrame с результатами (или None в случае ошибки)
        """
        # Загружаем англицизмы
        self.anglicisms = self._load_anglicisms()
        if not self.anglicisms:
            self.logger.error("Не удалось загрузить список англицизмов")
            return False, None

        # Загружаем тексты
        texts_df = self._load_texts()
        if texts_df is None:
            self.logger.error("Не удалось загрузить тексты")
            return False, None

        # Извлекаем уникальные слова
        unique_words = self._extract_unique_words(texts_df)
        if not unique_words:
            self.logger.error("Не удалось извлечь уникальные слова")
            return False, None

        # Маркируем слова
        labeled_words = self._label_words(unique_words)

        # Сохраняем результаты
        success, output_path = self._save_results(labeled_words)

        if success:
            # Создаем DataFrame с результатами для возврата
            result_df = pd.DataFrame({
                'word': list(labeled_words.keys()),
                'is_anglicism': list(labeled_words.values())
            })
            return True, result_df
        else:
            return False, None
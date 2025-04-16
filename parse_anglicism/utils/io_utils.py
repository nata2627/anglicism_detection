import os
import logging
from utils.parser import clean_wiki_markup

logger = logging.getLogger(__name__)


def setup_directory_structure(paths_config):
    """
    Создает структуру директорий проекта.

    Args:
        paths_config: Конфигурация путей
    """
    for path_name, path in paths_config.items():
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Создана директория: {path}")


def save_anglicisms(df, output_file, csv_output=None):
    """
    Сохраняет обработанные англицизмы в файл.

    Args:
        df (DataFrame): DataFrame с англицизмами
        output_file (str): Путь к файлу для сохранения списка англицизмов
        csv_output (str, optional): Путь для сохранения полных данных в CSV
    """
    # Проверяем наличие директории для output_file
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Сохраняем только колонку с самими словами
    df[['word']].to_csv(output_file, index=False, header=False)
    logger.info(f"Англицизмы сохранены в файл: {output_file}")

    # Если указан путь для CSV, сохраняем данные в CSV
    if csv_output:
        # Проверяем наличие директории для csv_output
        csv_dir = os.path.dirname(csv_output)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        # Создаем копию DataFrame с наиболее важными колонками
        export_df = df[['word', 'origin_language', 'word_length']].copy()

        # Применяем дополнительную очистку языка происхождения
        export_df['origin_language'] = export_df['origin_language'].apply(lambda x: clean_wiki_markup(x))

        # Переименовываем колонки для удобства
        export_df.columns = ['Англицизм', 'Язык происхождения', 'Длина слова']

        # Сохраняем в CSV
        try:
            export_df.to_csv(csv_output, index=False)
            logger.info(f"Данные сохранены в CSV: {csv_output}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении CSV-файла: {e}")
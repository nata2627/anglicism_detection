import os
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модули для предобработки
from clear_dataset.utils.preprocessor import TextPreprocessor


@hydra.main(
    version_base=None,
    config_path="../configs/clear_dataset",
    config_name="main"
)
def main(config: DictConfig) -> None:
    """
    Основная функция для запуска предобработки текстов.

    Создает экземпляр предобработчика TextPreprocessor с заданной
    конфигурацией и запускает процесс очистки и предобработки текстов.
    Сохраняет результаты в выходной файл.

    Args:
        config: Конфигурационный объект Hydra, содержащий все
               необходимые настройки для работы предобработчика

    Returns:
        None
    """
    # Инициализация предобработчика с загруженной конфигурацией
    preprocessor = TextPreprocessor(config)

    # Запуск процесса предобработки текстов
    processed_df = preprocessor.preprocess()

    # Вывод результатов предобработки
    if processed_df is not None:
        print(f"Обработано текстов: {len(processed_df)}")
        print(f"Результаты сохранены в: {os.path.join(config.paths.output_dir, config.output.file_name)}")


if __name__ == "__main__":
    # Настраиваем переменные окружения для Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_LOGGING.LEVEL"] = "WARN"  # Уменьшаем вывод логов Hydra

    # Запуск через Hydra
    main()
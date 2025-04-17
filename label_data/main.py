import os
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модуль для маркировки данных
from label_data.utils.labeler import DataLabeler


@hydra.main(
    version_base=None,
    config_path="../configs/label_data",
    config_name="main"
)
def main(config: DictConfig) -> None:
    """
    Основная функция для запуска процесса маркировки данных.

    Создает экземпляр маркировщика данных DataLabeler с заданной
    конфигурацией и запускает процесс маркировки. Выводит информацию
    о результатах маркировки.

    Args:
        config: Конфигурационный объект Hydra, содержащий все
               необходимые настройки для работы модуля маркировки

    Returns:
        None
    """
    # Инициализация маркировщика данных с загруженной конфигурацией
    labeler = DataLabeler(config)

    # Запуск процесса маркировки данных
    success, result_df = labeler.process()

    # Вывод результатов маркировки
    if success and result_df is not None:
        anglicism_count = result_df['is_anglicism'].sum()
        total_words = len(result_df)

        print(f"Обработка завершена успешно:")
        print(f"Всего уникальных слов: {total_words}")
        print(f"Количество англицизмов: {anglicism_count}")
        print(f"Процент англицизмов: {anglicism_count / total_words * 100:.2f}%")
        print(f"Результаты сохранены в: {os.path.join(config.paths.output_dir, config.output.file_name)}")
    else:
        print("Обработка не выполнена из-за ошибок. Проверьте логи для получения деталей.")


if __name__ == "__main__":
    # Настраиваем переменные окружения для Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_LOGGING.LEVEL"] = "WARN"  # Уменьшаем вывод логов Hydra

    # Запуск через Hydra
    main()
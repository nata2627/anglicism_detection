"""
Главный модуль для запуска парсера статей RBC.

Модуль использует библиотеку Hydra для управления конфигурацией
и запускает процесс парсинга статей с сайта RBC.ru.
"""

import hydra
from omegaconf import DictConfig
from utils.parser import RBCParser


@hydra.main(
    version_base=None,
    config_path="../configs/parse_dataset",
    config_name="main"
)
def main(config: DictConfig) -> None:
    """
    Основная функция для запуска парсера статей.

    Создает экземпляр парсера RBC с заданной конфигурацией
    и запускает процесс сбора статей. Выводит информацию
    о количестве собранных статей.

    Args:
        config: Конфигурационный объект Hydra, содержащий все
               необходимые настройки для работы парсера

    Returns:
        None
    """
    # Инициализация парсера с загруженной конфигурацией
    parser = RBCParser(config)

    # Запуск процесса парсинга статей
    articles_df = parser.parse_articles()

    # Вывод результатов парсинга
    print(f"Собрано статей: {len(articles_df)}")


if __name__ == "__main__":
    # Запуск главной функции с автоматической загрузкой конфигурации через Hydra
    main()
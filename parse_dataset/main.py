import hydra
from omegaconf import DictConfig
from utils.parser import RBCParser


@hydra.main(
    version_base=None,
    config_path="../configs/parse_dataset",
    config_name="main"
)
def main(config: DictConfig) -> None:
    # Инициализация парсера с загруженной конфигурацией
    parser = RBCParser(config)

    # Запуск процесса парсинга статей
    articles_df = parser.parse_articles()

    # Вывод результатов парсинга
    print(f"Общее количество собранных статей: {len(articles_df)}")


if __name__ == "__main__":
    # Запуск главной функции с автоматической загрузкой конфигурации через Hydra
    main()
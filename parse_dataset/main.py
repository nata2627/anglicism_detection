import hydra
from omegaconf import DictConfig
from utils.parser import RBCParser

@hydra.main(version_base=None, config_path="../configs/parse_dataset", config_name="main")
def main(config: DictConfig):
    parser = RBCParser(config)
    articles_df = parser.parse_articles()
    print(f"Parsed {len(articles_df)} articles")

if __name__ == "__main__":
    main()
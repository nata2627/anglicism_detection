# main.py
import logging
from datetime import datetime, timedelta
import hydra
from omegaconf import DictConfig
from utils.logging_setup import setup_logging
from utils.parser import RBCParser
from utils.date_utils import date_range, create_search_params
from utils.file_utils import save_to_csv

@hydra.main(version_base=None, config_path="../configs/parse_dataset", config_name="main")
def main(cfg: DictConfig):
    # Setup logging with config
    setup_logging(
        logs_dir=cfg.paths.logs_dir,
        file_level=cfg.logging.file_level,
        console_level=cfg.logging.console_level,
        log_format=cfg.logging.format
    )

    # Initialize parser with config
    parser = RBCParser(base_url=cfg.parser.base_url)

    # Configure parsing parameters
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=cfg.dates.days_back)).strftime('%Y-%m-%d')

    logging.info(f"Starting article collection from {start_date} to {end_date}")

    all_articles = []

    # Iterate through date ranges
    for date_range_dict in date_range(start_date, end_date, step_days=cfg.dates.step_days):
        params = create_search_params(
            category=cfg.parser.category,
            date_from=date_range_dict['dateFrom'],
            date_to=date_range_dict['dateTo'],
            query=cfg.parser.query,
            project=cfg.parser.project,
            page=cfg.parser.page
        )

        logging.info(f"Fetching articles for {date_range_dict['dateFrom']} - {date_range_dict['dateTo']}")

        # Get articles
        articles = parser.get_articles_batch(params)
        if articles:
            all_articles.extend(articles)
            logging.info(f"Found {len(articles)} articles")
        else:
            logging.warning(f"No articles found for date range")

    # Save all articles to CSV
    if all_articles:
        csv_file = save_to_csv(
            articles=all_articles,
            category=cfg.parser.category,
            data_dir=cfg.paths.data_dir,
            file_prefix=cfg.output.file_prefix,
            encoding=cfg.output.encoding
        )
        logging.info(f"Saved {len(all_articles)} articles to {csv_file}")
    else:
        logging.warning("No articles collected during the entire run")

if __name__ == "__main__":
    main()
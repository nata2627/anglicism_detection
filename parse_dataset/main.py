import logging
from datetime import datetime, timedelta
from utils.logging_setup import setup_logging
from utils.parser import RBCParser
from utils.date_utils import date_range, create_search_params
from utils.file_utils import save_to_csv

def main():
    # Setup logging
    setup_logging()

    # Initialize parser
    parser = RBCParser()

    # Configure parsing parameters
    category = 'TopRbcRu_economics'  # Change as needed
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    logging.info(f"Starting article collection from {start_date} to {end_date}")

    all_articles = []

    # Iterate through date ranges
    for date_range_dict in date_range(start_date, end_date, step_days=1):
        params = create_search_params(
            category=category,
            date_from=date_range_dict['dateFrom'],
            date_to=date_range_dict['dateTo']
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
        csv_file = save_to_csv(all_articles, category)
        logging.info(f"Saved {len(all_articles)} articles to {csv_file}")
    else:
        logging.warning("No articles collected during the entire run")

if __name__ == "__main__":
    main()
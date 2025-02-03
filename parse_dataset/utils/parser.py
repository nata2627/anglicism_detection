import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime
from pathlib import Path
from .logger import CustomLogger


class RBCParser:
    def __init__(self, config):
        self.config = config
        self.logger = CustomLogger(config)

    def _get_url(self, param_dict: dict) -> str:
        url = 'https://www.rbc.ru/search/ajax/?' + \
              '&'.join(f"{k}={v}" for k, v in param_dict.items())
        self.logger.debug(f"Generated URL: {url}")
        return url

    def _get_article_data(self, url: str):
        try:
            r = rq.get(url)
            soup = bs(r.text, features="lxml")
            div_overview = soup.find('div', {'class': 'article__text__overview'})
            overview = div_overview.text.strip() if div_overview else None

            p_text = soup.find_all('p')
            text = ' '.join(p.text.strip() for p in p_text) if p_text else None

            return overview, text
        except Exception as e:
            self.logger.error(f"Error parsing article {url}: {str(e)}")
            return None, None

    def _get_search_table(self, param_dict: dict) -> pd.DataFrame:
        try:
            url = self._get_url(param_dict)
            r = rq.get(url)
            search_table = pd.DataFrame(r.json()['items'])

            if not search_table.empty and self.config.parser.include_text:
                self.logger.info(f"Found {len(search_table)} articles on page {param_dict['page']}")
                results = [self._get_article_data(row['fronturl']) for _, row in search_table.iterrows()]
                overviews, texts = zip(*results)
                search_table['overview'] = overviews
                search_table['text'] = texts

            if 'publish_date_t' in search_table.columns:
                search_table.sort_values('publish_date_t', inplace=True, ignore_index=True)

            return search_table
        except Exception as e:
            self.logger.error(f"Error getting search table: {str(e)}")
            return pd.DataFrame()

    def parse_articles(self):
        param_dict = {
            'query': self.config.parser.query,
            'project': self.config.parser.project,
            'category': self.config.parser.category,
            'material': self.config.parser.material,
            'dateFrom': datetime.strptime(self.config.parser.dateFrom, '%Y-%m-%d').strftime('%d.%m.%Y'),
            'dateTo': datetime.strptime(self.config.parser.dateTo, '%Y-%m-%d').strftime('%d.%m.%Y'),
            'page': str(self.config.parser.initial_page)
        }

        self.logger.info(f"Starting parsing with parameters: {param_dict}")

        results = []
        page = self.config.parser.initial_page

        while True:
            # Проверяем ограничение по страницам
            if page >= self.config.parser.max_pages:
                self.logger.info(f"Reached maximum number of pages ({self.config.parser.max_pages})")
                break

            param_dict['page'] = str(page)
            result = self._get_search_table(param_dict)

            if result.empty:
                self.logger.info(f"No more results found after page {page}")
                break

            results.append(result)
            self.logger.info(f"Successfully parsed page {page}")
            page += 1

        if results:
            final_df = pd.concat(results, ignore_index=True)
            self._save_results(final_df)
            return final_df
        return pd.DataFrame()

    def _save_results(self, df: pd.DataFrame):
        try:
            data_dir = Path(self.config.paths.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            # Формируем более информативное имя файла
            filename = f"{self.config.output.file_prefix}_{self.config.parser.dateFrom}_to_{self.config.parser.dateTo}_pages_{self.config.parser.max_pages}.csv"
            filepath = data_dir / filename

            df.to_csv(filepath, index=False, encoding=self.config.output.encoding)
            self.logger.info(f"Results saved to {filepath}")
            self.logger.info(f"Total articles parsed: {len(df)}")
            self.logger.info(f"Absolute path to saved file: {filepath.absolute()}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
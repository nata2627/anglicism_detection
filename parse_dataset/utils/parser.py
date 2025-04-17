from typing import Dict, Tuple, Optional
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime
from pathlib import Path


class RBCParser:
    def __init__(self, config) -> None:
        self.config = config

    def _get_url(self, param_dict: Dict[str, str]) -> str:
        url = 'https://www.rbc.ru/search/ajax/?' + \
              '&'.join(f"{k}={v}" for k, v in param_dict.items())
        print(f"Сгенерирован URL: {url}")
        return url

    def _get_article_data(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            response = rq.get(url)
            soup = bs(response.text, features="lxml")

            # Получаем обзор статьи
            div_overview = soup.find('div', {'class': 'article__text__overview'})
            overview = div_overview.text.strip() if div_overview else None

            # Получаем полный текст статьи
            p_text = soup.find_all('p')
            text = ' '.join(p.text.strip() for p in p_text) if p_text else None

            return overview, text
        except Exception as e:
            print(f"Ошибка при парсинге статьи {url}: {str(e)}")
            return None, None

    def _get_search_table(self, param_dict: Dict[str, str]) -> pd.DataFrame:
        try:
            url = self._get_url(param_dict)
            response = rq.get(url)
            search_table = pd.DataFrame(response.json()['items'])

            if not search_table.empty and self.config.parser.include_text:
                print(
                    f"Найдено {len(search_table)} статей на странице {param_dict['page']}"
                )
                # Получаем данные для каждой статьи
                results = [
                    self._get_article_data(row['fronturl'])
                    for _, row in search_table.iterrows()
                ]
                overviews, texts = zip(*results)
                search_table['overview'] = overviews
                search_table['text'] = texts

            # Сортируем по дате публикации, если такой столбец есть
            if 'publish_date_t' in search_table.columns:
                search_table.sort_values(
                    'publish_date_t',
                    inplace=True,
                    ignore_index=True
                )

            return search_table
        except Exception as e:
            print(f"Ошибка при получении таблицы поиска: {str(e)}")
            return pd.DataFrame()

    def parse_articles(self) -> pd.DataFrame:
        param_dict = {
            'query': self.config.parser.query,
            'project': self.config.parser.project,
            'category': self.config.parser.category,
            'material': self.config.parser.material,
            'dateFrom': datetime.strptime(
                self.config.parser.dateFrom, '%Y-%m-%d'
            ).strftime('%d.%m.%Y'),
            'dateTo': datetime.strptime(
                self.config.parser.dateTo, '%Y-%m-%d'
            ).strftime('%d.%m.%Y'),
            'page': str(self.config.parser.initial_page)
        }

        print(f"Начало парсинга с параметрами: {param_dict}")

        results = []
        page = self.config.parser.initial_page

        while True:
            # Проверяем ограничение по страницам
            if page >= self.config.parser.max_pages:
                print(
                    f"Достигнуто максимальное количество страниц "
                    f"({self.config.parser.max_pages})"
                )
                break

            param_dict['page'] = str(page)
            result = self._get_search_table(param_dict)

            if result.empty:
                print(f"Больше результатов не найдено после страницы {page}")
                break

            results.append(result)
            print(f"Успешно обработана страница {page}")
            page += 1

        if results:
            final_df = pd.concat(results, ignore_index=True)
            self._save_results(final_df)
            return final_df
        return pd.DataFrame()

    def _save_results(self, df: pd.DataFrame) -> None:
        try:
            data_dir = Path(self.config.paths.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            # Формируем информативное имя файла
            filename = (
                f"{self.config.output.file_prefix}_"
                f"{self.config.parser.dateFrom}_to_"
                f"{self.config.parser.dateTo}_pages_"
                f"{self.config.parser.max_pages}.csv"
            )
            filepath = data_dir / filename

            df.to_csv(filepath, index=False, encoding=self.config.output.encoding)
            print(f"Результаты сохранены в {filepath}")
            print(f"Всего обработано статей: {len(df)}")
            print(f"Абсолютный путь к сохраненному файлу: {filepath.absolute()}")
        except Exception as e:
            print(f"Ошибка при сохранении результатов: {str(e)}")
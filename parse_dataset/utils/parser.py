from typing import Dict, Tuple, Optional
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime
from pathlib import Path
from .logger import CustomLogger


class RBCParser:
    """
    Класс для парсинга статей с сайта RBC.ru.

    Осуществляет поиск и извлечение статей по заданным параметрам,
    включая текст статей, их обзоры и метаданные.

    Attributes:
        config: Конфигурационный объект с настройками парсера
        logger: Объект для логирования процесса работы парсера
    """

    def __init__(self, config) -> None:
        """
        Инициализация парсера RBC.

        Args:
            config: Объект конфигурации, содержащий настройки парсера
        """
        self.config = config
        self.logger = CustomLogger(config)

    def _get_url(self, param_dict: Dict[str, str]) -> str:
        """
        Формирует URL для поискового запроса на основе переданных параметров.

        Args:
            param_dict: Словарь параметров запроса (query, project, category и т.д.)

        Returns:
            str: Сформированный URL для поискового запроса
        """
        url = 'https://www.rbc.ru/search/ajax/?' + \
              '&'.join(f"{k}={v}" for k, v in param_dict.items())
        self.logger.debug(f"Сгенерирован URL: {url}")
        return url

    def _get_article_data(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Извлекает содержимое статьи по указанному URL.

        Args:
            url: URL статьи для парсинга

        Returns:
            Tuple[Optional[str], Optional[str]]: Кортеж, содержащий обзор и полный текст статьи.
            В случае ошибки возвращает (None, None)
        """
        try:
            response = rq.get(url)
            soup = bs(response.text, features="lxml")

            # Получаем обзор статьи
            div_overview = soup.find('div', {'class': 'article__text__overview'})
            overview = div_overview.text.strip() if div_overview else None

            # Получаем основной контент статьи
            # Ищем основной контейнер с текстом статьи
            article_container = soup.find('div', {'class': 'article__text'})

            if article_container:
                # Извлекаем только параграфы основного текста, исключая рекламу и другие элементы
                p_text = article_container.find_all('p')
                text = ' '.join(p.text.strip() for p in p_text) if p_text else None
            else:
                # Если не нашли основной контейнер, ищем все параграфы на странице
                p_text = soup.find_all('p')
                text = ' '.join(p.text.strip() for p in p_text) if p_text else None

                # Удаляем повторяющиеся блоки в конце статьи
                if text and "РБК в Telegram" in text:
                    text = text.split("РБК в Telegram")[0]

            return overview, text
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге статьи {url}: {str(e)}")
            return None, None

    def _get_search_table(self, param_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Получает таблицу результатов поиска для заданных параметров.

        Args:
            param_dict: Словарь параметров поискового запроса

        Returns:
            pd.DataFrame: DataFrame с результатами поиска. Пустой DataFrame в случае ошибки.
        """
        try:
            url = self._get_url(param_dict)
            response = rq.get(url)
            search_results = response.json()['items']

            # Создаем список для хранения отфильтрованных данных
            filtered_results = []

            # Определяем столбцы, которые нужно сохранить
            # Исключаем: picture, badge, pay_option, data, _score
            columns_to_keep = ['id', 'project', 'project_nick', 'type', 'category',
                               'title', 'body', 'publish_date', 'publish_date_t', 'fronturl']

            # Фильтруем каждую запись
            for item in search_results:
                filtered_item = {key: item.get(key) for key in columns_to_keep if key in item}
                filtered_results.append(filtered_item)

            # Создаем DataFrame с отфильтрованными данными
            search_table = pd.DataFrame(filtered_results)

            if not search_table.empty and self.config.parser.include_text:
                self.logger.info(
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
            self.logger.error(f"Ошибка при получении таблицы поиска: {str(e)}")
            return pd.DataFrame()

    def parse_articles(self) -> pd.DataFrame:
        """
        Основной метод для парсинга статей. Собирает все статьи согласно
        заданным в конфигурации параметрам.

        Returns:
            pd.DataFrame: DataFrame со всеми собранными статьями и их данными.
            В случае ошибки возвращает пустой DataFrame.
        """
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

        self.logger.info(f"Начало парсинга с параметрами: {param_dict}")

        results = []
        page = self.config.parser.initial_page

        while True:
            # Проверяем ограничение по страницам
            if page >= self.config.parser.max_pages:
                self.logger.info(
                    f"Достигнуто максимальное количество страниц "
                    f"({self.config.parser.max_pages})"
                )
                break

            param_dict['page'] = str(page)
            result = self._get_search_table(param_dict)

            if result.empty:
                self.logger.info(f"Больше результатов не найдено после страницы {page}")
                break

            results.append(result)
            self.logger.info(f"Успешно обработана страница {page}")
            page += 1

        if results:
            final_df = pd.concat(results, ignore_index=True)
            self._save_results(final_df)
            return final_df
        return pd.DataFrame()

    def _save_results(self, df: pd.DataFrame) -> None:
        """
        Сохраняет результаты парсинга в CSV файл.

        Args:
            df: DataFrame с результатами парсинга

        Raises:
            Exception: В случае ошибки при сохранении результатов
        """
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

            # Проверяем порядок столбцов для лучшей читаемости файла
            # Нужные столбцы перемещаем в начало
            columns_order = [
                'id', 'project', 'project_nick', 'type', 'category',
                'title', 'body', 'publish_date', 'publish_date_t', 'fronturl',
                'overview', 'text'
            ]

            # Переупорядочиваем столбцы, сохраняя только существующие
            existing_columns = [col for col in columns_order if col in df.columns]
            df = df[existing_columns]

            # Сохраняем файл
            df.to_csv(filepath, index=False, encoding=self.config.output.encoding)
            self.logger.info(f"Результаты сохранены в {filepath}")
            self.logger.info(f"Всего обработано статей: {len(df)}")
            self.logger.info(f"Абсолютный путь к сохраненному файлу: {filepath.absolute()}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {str(e)}")
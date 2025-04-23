from typing import Dict, Tuple, Optional, List
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
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
        self.logger.info(f"Сгенерирован URL: {url}")
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

    def _parse_single_date(self, date_str: str) -> pd.DataFrame:
        """
        Парсит статьи за конкретную дату.

        Args:
            date_str: Дата в формате 'YYYY-MM-DD'

        Returns:
            pd.DataFrame: DataFrame со всеми собранными статьями за указанную дату
        """
        param_dict = {
            'query': self.config.parser.query,
            'project': self.config.parser.project,
            'category': self.config.parser.category,
            'material': self.config.parser.material,
            'dateFrom': datetime.strptime(date_str, '%Y-%m-%d').strftime('%d.%m.%Y'),
            'dateTo': datetime.strptime(date_str, '%Y-%m-%d').strftime('%d.%m.%Y'),
            'page': '1'  # Начинаем с первой страницы
        }

        self.logger.info(f"Начало парсинга для даты {date_str} с параметрами: {param_dict}")

        results = []
        page = 1

        while True:
            # Проверяем ограничение по страницам
            if page > self.config.parser.max_pages:
                self.logger.info(
                    f"Достигнуто максимальное количество страниц для даты {date_str} "
                    f"({self.config.parser.max_pages})"
                )
                break

            param_dict['page'] = str(page)
            result = self._get_search_table(param_dict)

            if result.empty:
                self.logger.info(f"Больше результатов не найдено для даты {date_str} после страницы {page}")
                break

            results.append(result)
            self.logger.info(f"Успешно обработана страница {page} для даты {date_str}")
            page += 1

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def _get_date_range(self) -> List[str]:
        """
        Получает список всех дат в заданном диапазоне.

        Returns:
            List[str]: Список дат в формате 'YYYY-MM-DD'
        """
        start_date = datetime.strptime(self.config.parser.dateFrom, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.parser.dateTo, '%Y-%m-%d')

        date_list = []
        current_date = start_date

        while current_date <= end_date:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        return date_list

    def parse_articles(self) -> pd.DataFrame:
        """
        Основной метод для парсинга статей. Собирает статьи отдельно для каждой даты
        в заданном диапазоне и сохраняет результаты в отдельные файлы.

        Returns:
            pd.DataFrame: Объединенный DataFrame со всеми собранными статьями.
            В случае ошибки возвращает пустой DataFrame.
        """
        date_range = self._get_date_range()
        self.logger.info(f"Будет обработано {len(date_range)} дат с {date_range[0]} по {date_range[-1]}")

        all_results = []

        # Используем tqdm для отображения прогресса
        for date_str in tqdm(date_range, desc="Парсинг по датам"):
            df = self._parse_single_date(date_str)

            if not df.empty:
                # Сохраняем результаты для текущей даты
                self._save_results(df, date_str)
                all_results.append(df)
                self.logger.info(f"Обработана дата {date_str}, собрано {len(df)} статей")
            else:
                self.logger.info(f"Для даты {date_str} статьи не найдены")

        # Объединяем все результаты в один DataFrame
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            self.logger.info(f"Всего обработано статей: {len(final_df)}")
            return final_df

        self.logger.warning("Не найдено ни одной статьи за весь период")
        return pd.DataFrame()

    def _save_results(self, df: pd.DataFrame, date_str: str) -> None:
        """
        Сохраняет результаты парсинга в CSV файл.

        Args:
            df: DataFrame с результатами парсинга
            date_str: Дата в формате 'YYYY-MM-DD', для которой сохраняются результаты

        Raises:
            Exception: В случае ошибки при сохранении результатов
        """
        try:
            data_dir = Path(self.config.paths.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            # Формируем информативное имя файла
            filename = (
                f"{self.config.output.file_prefix}_"
                f"{date_str}_pages_"
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
            self.logger.info(f"Результаты для даты {date_str} сохранены в {filepath}")
            self.logger.info(f"Абсолютный путь к сохраненному файлу: {filepath.absolute()}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов для даты {date_str}: {str(e)}")
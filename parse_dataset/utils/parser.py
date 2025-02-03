from datetime import datetime
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, Tuple

class RBCParser:
    """Parser for RBC news articles with focus on core content extraction"""

    BASE_URL = 'https://www.rbc.ru/search/ajax/'

    def __init__(self):
        self.session = requests.Session()

    def _build_url(self, params: Dict[str, str]) -> str:
        """Constructs search URL from parameters"""
        query_params = {
            'project': params['project'],
            'category': params['category'],
            'dateFrom': params['dateFrom'],
            'dateTo': params['dateTo'],
            'page': params['page'],
            'query': params['query']
        }
        return f"{self.BASE_URL}?{'&'.join(f'{k}={v}' for k, v in query_params.items())}"

    def _get_article_content(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract article title, text and overview from URL"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            title = soup.find('h1', {'class': 'article__header__title'})
            title = title.text.strip() if title else None

            overview = soup.find('div', {'class': 'article__text__overview'})
            overview = overview.text.strip() if overview else None

            paragraphs = soup.find_all('p')
            text = ' '.join(p.text.strip() for p in paragraphs) if paragraphs else None

            return title, overview, text
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return None, None, None

    def get_articles_batch(self, params: Dict[str, str]) -> list:
        """Fetch a batch of articles based on search parameters"""
        try:
            url = self._build_url(params)
            response = self.session.get(url)
            response.raise_for_status()

            articles = []
            for item in response.json()['items']:
                title, overview, text = self._get_article_content(item['fronturl'])

                article = {
                    'title': title,
                    'url': item['fronturl'],
                    'category': params['category'],
                    'date': datetime.fromtimestamp(item['publish_date_t']),
                    'overview': overview,
                    'text': text
                }
                articles.append(article)

            return articles
        except Exception as e:
            print(f"Error fetching batch: {str(e)}")
            return []
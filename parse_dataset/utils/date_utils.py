from datetime import datetime, timedelta
from typing import Dict, Generator

def date_range(start_date: str, end_date: str, step_days: int = 1) -> Generator[Dict[str, str], None, None]:
    """Generate date ranges for fetching articles"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    current = start
    while current <= end:
        next_date = min(current + timedelta(days=step_days), end)
        yield {
            'dateFrom': current.strftime('%d.%m.%Y'),
            'dateTo': next_date.strftime('%d.%m.%Y')
        }
        current = next_date + timedelta(days=1)

def create_search_params(
        category: str,
        date_from: str,
        date_to: str,
        query: str = '',
        project: str = 'rbcnews',
        page: str = '0'
) -> Dict[str, str]:
    """Create parameter dictionary for RBC parser"""
    return {
        'project': project,
        'category': category,
        'dateFrom': date_from,
        'dateTo': date_to,
        'page': page,
        'query': query
    }
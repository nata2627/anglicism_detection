import os
import pandas as pd
from datetime import datetime

def create_directory_structure():
    """Create necessary directories for logs and data"""
    directories = ['logs', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_to_csv(articles: list, category: str):
    """Save articles to CSV file in data directory"""
    if not articles:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join('data', f'rbc_articles_{category}_{timestamp}.csv')

    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False, encoding='utf-8')
    return filename
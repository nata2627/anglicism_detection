import os
import pandas as pd
from datetime import datetime

def save_to_csv(articles: list, category: str, data_dir: str, file_prefix: str, encoding: str):
    """Save articles to CSV file in data directory"""
    if not articles:
        return

    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(data_dir, f'{file_prefix}_{category}_{timestamp}.csv')

    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False, encoding=encoding)
    return filename
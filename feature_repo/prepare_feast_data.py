"""
Подготовка данных для Feast Feature Store
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

def prepare_feast_data():
    # Читаем обработанные данные
    data_path = "/opt/airflow/data/processed/processed.csv"
    if not os.path.exists(data_path):
        print(f"Файл не найден: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    
    # Добавляем обязательные поля для Feast
    df['currency_pair_id'] = 'USD_RUB'  # Идентификатор валютной пары
    df['created_at'] = pd.Timestamp.now()  # Время создания записи
    
    # Убедимся что date в правильном формате
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        # Создаем искусственную временную шкалу
        start_date = datetime(2020, 1, 1)
        df['date'] = [start_date + pd.Timedelta(days=i) for i in range(len(df))]
    
    # Сохраняем в формате для Feast
    feast_data_path = "feature_repo/data/currency_features.parquet"
    os.makedirs(os.path.dirname(feast_data_path), exist_ok=True)
    
    # Сохраняем в Parquet (лучше для Feast)
    df.to_parquet(feast_data_path, index=False)
    
    print(f"Данные подготовлены для Feast: {feast_data_path}")
    print(f"Размер: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    prepare_feast_data()
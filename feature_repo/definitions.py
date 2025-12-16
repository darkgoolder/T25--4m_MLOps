# feature_repo/definitions.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int32

# 1. Определяем СУЩНОСТЬ (Entity) — объект, для которого храним фичи.
# В нашем случае это "валютная пара". Мы будем предсказывать для USD/RUB.
currency_entity = Entity(
    name="currency_pair",               # Название сущности
    join_keys=["currency_pair_id"],     # Ключ для объединения данных
    value_type=ValueType.STRING,
    description="Идентификатор валютной пары (например, USD/RUB)",
)

# 2. Определяем ИСТОЧНИК ДАННЫХ.
# Говорим Feast, где лежат сырые данные для создания фич.
# Указываем путь к вашему обработанному CSV-файлу.
currency_stats_source = FileSource(
    name="currency_stats_source",
    path="../data/processed/processed_for_feast.csv",  # Путь ОТНОСИТЕЛЬНО папки feature_repo
    timestamp_field="date",                  # Колонка с датой в вашем CSV
    created_timestamp_column="created_at",
)

# 3. Определяем ПРЕДСТАВЛЕНИЕ ФИЧ (Feature View).
# Это набор признаков, связанных с нашей сущностью, который мы хотим использовать.
technical_features_view = FeatureView(
    name="currency_technical_features",
    entities=[currency_entity],  # Связываем с нашей сущностью
    ttl=timedelta(days=365),     # "Срок годности" фич (в нашем случае условно)
    schema=[                     # Здесь перечисляем ВСЕ признаки, которые есть в вашем CSV
        Field(name="USD_RUB", dtype=Float32),
        Field(name="EUR_RUB", dtype=Float32),
        Field(name="GBP_RUB", dtype=Float32),
        Field(name="day_of_week", dtype=Int32),
        Field(name="is_weekend", dtype=Int32),
        Field(name="USD_RUB_lag_1", dtype=Float32),
        Field(name="USD_RUB_lag_2", dtype=Float32),
        Field(name="date", dtype=Float32),
        Field(name="departure_hour_bucket", dtype=Float32),
        # ... Добавьте сюда ВСЕ остальные колонки из processed.csv
        # Просто скопируйте их названия из файла или вывода вашего кода.
        # Для примера я добавил только несколько. ВАЖНО: добавьте все, которые использует модель.
    ],
    source=currency_stats_source,
    online=True,  # Разрешаем использовать фичи для онлайн-предсказаний
)
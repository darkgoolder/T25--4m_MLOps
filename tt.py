# fix_dataset.py
import pandas as pd
import chardet

# 1. Определите кодировку исходного файла
with open('data/raw/flights_sample.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Определена кодировка: {encoding} (уверенность: {result['confidence']:.2f})")

# 2. Прочитайте файл с правильной кодировкой
# Попробуйте разные разделители
try:
    # Сначала пробуем точку с запятой
    data = pd.read_csv('data/raw/flights_sample.csv', 
                      encoding=encoding, 
                      sep=';')
    print("Успешно прочитано с разделителем ';'")
except:
    try:
        # Пробуем запятую
        data = pd.read_csv('data/raw/flights_sample.csv',
                          encoding=encoding,
                          sep=',')
        print("Успешно прочитано с разделителем ','")
    except:
        # Пробуем табуляцию
        data = pd.read_csv('data/raw/flights_sample.csv',
                          encoding=encoding,
                          sep='\t')
        print("Успешно прочитано с разделителем '\\t'")

# 3. Переименуйте колонки если нужно
# Проверьте названия колонок
print("Текущие колонки:", data.columns.tolist())

# Если колонки имеют другие имена, переименуйте их
column_mapping = {
    # Пример: 'Доллар' -> 'USD_RUB'
    # 'Евро' -> 'EUR_RUB'
    # 'Фунт' -> 'GBP_RUB'
    # 'Дата' -> 'date'
}

for old_name, new_name in column_mapping.items():
    if old_name in data.columns:
        data.rename(columns={old_name: new_name}, inplace=True)
        print(f"Переименовано: {old_name} -> {new_name}")

# 4. Сохраните в правильном формате
data.to_csv('data/raw/flights_sample_fixed.csv', 
           sep=';', 
           index=False, 
           encoding='utf-8')

print(f"Файл сохранен: data/raw/flights_sample_fixed.csv")
print(f"Форма: {data.shape}")
print(data.head())
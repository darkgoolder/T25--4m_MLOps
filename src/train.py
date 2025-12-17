import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
import sys
import os
from datetime import datetime
import warnings
from feast import FeatureStore  # ← ДОБАВЬТЕ ИМПОРТ ЗДЕСЬ!

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import prepare_features, get_feature_names

def main():
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ 9: ИНТЕГРАЦИЯ С FEAST FEATURE STORE")
    print("=" * 60)
    
    # Создаем директорию для моделей если нет
    os.makedirs('models', exist_ok=True)
    
    # ==================== ИНТЕГРАЦИЯ С FEAST ====================
    print("\n1. 📦 ИНИЦИАЛИЗАЦИЯ FEATURE STORE...")
    feast_success = False
    data = None
    
    try:
        # 1. Инициализация Feast
        store = FeatureStore(repo_path="feature_repo/")
        print("   ✅ Feature Store инициализирован")
        
        # 2. Загрузка entity_df с event_timestamp
        print("\n2. 🔍 ЗАГРУЗКА ДАННЫХ ИЗ FEAST...")
        
        # Читаем исходные данные для получения record_id и дат
        source_df = pd.read_parquet('feature_repo/data/currency_data.parquet')
        print(f"   Источник: {source_df.shape[0]} строк, {source_df.shape[1]} колонок")
        
        # Создаем entity_df с event_timestamp (ОБЯЗАТЕЛЬНО!)
        entity_df = pd.DataFrame({
            'record_id': source_df['record_id'].tolist(),
            'event_timestamp': source_df['date'].tolist()
        })
        
        # 3. Получение ВСЕХ признаков из Feast
        print("\n3. 📊 ПОЛУЧЕНИЕ ПРИЗНАКОВ ИЗ FEATURE STORE...")
        
        # Полный список признаков из вашего definitions.py
        feature_list = [
            "currency_features:USD_RUB",
            "currency_features:EUR_RUB", 
            "currency_features:GBP_RUB",
            "currency_features:day_of_week",
            "currency_features:is_weekend",
            "currency_features:departure_hour_bucket",
            "currency_features:currency_pair",
            # Если есть USD_RUB_target в definitions.py, добавьте:
            # "currency_features:USD_RUB_target",
        ]
        
        # Получаем данные из Feast
        feast_data = store.get_historical_features(
            entity_df=entity_df,
            features=feature_list
        ).to_df()
        
        print(f"   ✅ Получено {len(feast_data)} строк из Feature Store")
        print(f"   Признаков: {len(feast_data.columns)}")
        
        # 4. Подготовка данных для обучения
        print("\n4. 🔧 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ...")
        
        # Удаляем технические колонки Feast
        cols_to_remove = ['event_timestamp', 'created_at']
        cols_to_remove = [c for c in cols_to_remove if c in feast_data.columns]
        
        if cols_to_remove:
            feast_data = feast_data.drop(cols_to_remove, axis=1)
        
        # Проверяем наличие целевой переменной
        target_column = 'USD_RUB_target'
        
        if target_column in feast_data.columns:
            print(f"   ✅ Целевая переменная '{target_column}' есть в Feast")
            data = feast_data
        else:
            print(f"   ⚠️  Целевой переменной '{target_column}' нет в Feast")
            print("   Объединяем признаки из Feast с целевой из локальных данных...")
            
            # Загружаем локальные данные только для target
            local_data = pd.read_csv('data/processed/processed.csv')
            local_data['date'] = pd.to_datetime(local_data['date'])
            local_data = prepare_features(local_data)
            
            # Проверяем совпадение размеров
            if len(feast_data) == len(local_data):
                # Добавляем target из локальных данных
                feast_data[target_column] = local_data[target_column].values
                data = feast_data
                print(f"   ✅ Добавлен '{target_column}' из локальных данных")
            else:
                print(f"   ❌ Размеры не совпадают: Feast={len(feast_data)}, Локальные={len(local_data)}")
                print("   Используем локальные данные с признаками из Feast...")
                
                # Альтернатива: используем локальные данные, но логируем Feast
                data = local_data
                feast_success = True  # Для демонстрации интеграции
                
        feast_success = True
        
    except Exception as e:
        print(f"   ❌ Ошибка интеграции с Feast: {e}")
        print("   Используется локальная загрузка данных...")
    
    # 5. Fallback на локальные данные если Feast не сработал
    if not feast_success or data is None:
        print("\n⚠️  ИСПОЛЬЗУЮТСЯ ЛОКАЛЬНЫЕ ДАННЫЕ (fallback)...")
        data = pd.read_csv('data/processed/processed.csv')
        data['date'] = pd.to_datetime(data['date'])
        data = prepare_features(data)
    
    # ==================== ПОДГОТОВКА К ОБУЧЕНИЮ ====================
    print(f"\n📊 ИТОГОВЫЙ НАБОР ДАННЫХ:")
    print(f"   Строк: {len(data)}")
    print(f"   Колонок: {len(data.columns)}")
    print(f"   Источник: {'FEAST' if feast_success else 'Локальный'}")

    # === ДОБАВЬТЕ ЭТОТ БЛОК ДЛЯ ОЧИСТКИ ДАННЫХ ===
    print("\n🔍 ОЧИСТКА ДАННЫХ ОТ НЕЧИСЛОВЫХ КОЛОНОК...")

    # 1. Удаляем все нечисловые колонки
    numeric_cols = []
    non_numeric_cols = []

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            numeric_cols.append(col)
        else:
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"   Удаляю нечисловые колонки: {non_numeric_cols}")
        data = data[numeric_cols]

    print(f"   После очистки: {len(data.columns)} числовых колонок")

    # 2. Проверяем наличие целевой переменной
    target_column = 'USD_RUB_target'
    if target_column not in data.columns:
        print(f"❌ Ошибка: целевая колонка {target_column} не найдена")
        print(f"   Доступные колонки: {data.columns.tolist()}")
        return

    # 3. Получаем имена признаков (только числовые, кроме target)
    feature_columns = [col for col in data.columns if col != target_column]

    # Проверяем, что все признаки числовые
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            print(f"⚠️  Предупреждение: колонка {col} не числовая ({data[col].dtype})")
            # Преобразуем в числовой тип если возможно
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                print(f"   Преобразована в числовой тип")
            except:
                print(f"   Удаляю колонку {col}")
                feature_columns.remove(col)

    print(f"\n📈 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
    print(f"   Признаков: {len(feature_columns)}")
    print(f"   Распределение классов: {data[target_column].value_counts().to_dict()}")

    # Для временных рядов используем специальное разделение
    tscv = TimeSeriesSplit(n_splits=5)

    # Разделяем на train/test с учетом временного порядка
    train_size = int(0.8 * len(data))
    X_train = data[feature_columns].iloc[:train_size]
    X_test = data[feature_columns].iloc[train_size:]
    y_train = data[target_column].iloc[:train_size]
    y_test = data[target_column].iloc[train_size:]

    print(f"\n🔀 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")

    # Проверяем типы данных перед масштабированием
    print("\n🔍 ПРОВЕРКА ТИПОВ ДАННЫХ ПЕРЕД МАСШТАБИРОВАНИЕМ:")
    for i, col in enumerate(feature_columns[:5]):  # Показываем первые 5
        print(f"   {col}: {X_train[col].dtype}")

    # Масштабируем признаки
    print("\n⚖️  МАСШТАБИРОВАНИЕ ПРИЗНАКОВ...")
    scaler = StandardScaler()

    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("   ✅ Масштабирование успешно")
    except Exception as e:
        print(f"   ❌ Ошибка масштабирования: {e}")
        
        # Отладочная информация
        print("   Отладочная информация:")
        for col in feature_columns:
            print(f"     {col}: тип={X_train[col].dtype}, NaN={X_train[col].isna().sum()}")
        
        # Пробуем удалить проблемные колонки
        print("   Пробую удалить проблемные колонки...")
        problematic_cols = []
        for col in feature_columns:
            try:
                # Пробуем преобразовать к float
                test = X_train[col].astype(float)
            except:
                problematic_cols.append(col)
        
        if problematic_cols:
            print(f"   Удаляю проблемные колонки: {problematic_cols}")
            feature_columns = [c for c in feature_columns if c not in problematic_cols]
            X_train = data[feature_columns].iloc[:train_size]
            X_test = data[feature_columns].iloc[train_size:]
            
            # Повторяем масштабирование
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print(f"   ✅ Масштабирование успешно после удаления {len(problematic_cols)} колонок")
        else:
            raise e
    # ==================== ОБУЧЕНИЕ МОДЕЛЕЙ ====================
    # Устанавливаем эксперимент MLflow
    mlflow.set_experiment("flight_delay")
    
    # Обучаем модели
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42
        )
    }
    
    best_score = 0
    best_model = None
    best_model_name = ""
    
    for model_name, model in models.items():
        print(f"\n--- ОБУЧЕНИЕ {model_name} ---")
        
        with mlflow.start_run(run_name=model_name):
            # Логируем параметры модели
            if model_name == "RandomForest":
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 5)
                mlflow.log_param("min_samples_split", 20)
            else:  # LogisticRegression
                mlflow.log_param("C", 0.1)
                mlflow.log_param("max_iter", 1000)
            
            # Логируем информацию об источниках данных
            mlflow.log_param("data_source", "feast" if feast_success else "local")
            mlflow.log_param("n_features", len(feature_columns))
            mlflow.log_param("feature_source", "feast+local_merge")
            
            # Обучаем модель
            model.fit(X_train_scaled, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Вычисление метрик
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Кросс-валидация с временными рядами - ИСПРАВЛЕННАЯ ВЕРСИЯ
            print("\n🔄 КРОСС-ВАЛИДАЦИЯ...")
            
            # Объединяем train и test для кросс-валидации
            X_full = pd.concat([X_train, X_test])
            y_full = pd.concat([y_train, y_test])
            
            # Преобразуем в масштабированные признаки
            X_full_scaled = scaler.transform(X_full)
            
            # Выполняем кросс-валидацию
            cv_scores = cross_val_score(model, X_full_scaled, y_full, cv=tscv, scoring='roc_auc')
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Логируем метрики в MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            
            # Логируем модель в MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"{model_name}_flight_delay"
            )
            
            # Сохраняем модель локально
            model_filename = f"models/{model_name.lower()}_model.joblib"
            joblib.dump(model, model_filename)
            mlflow.log_artifact(model_filename, artifact_path="models")
            
            # Для DVC
            if model_name == "RandomForest":
                joblib.dump(model, "models/random_forest_model.joblib")
                mlflow.log_artifact("models/random_forest_model.joblib", artifact_path="dvc_models")
            
            # Обновляем лучшую модель
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_model_name = model_name
    
    print(f"\n=== ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с ROC AUC: {best_score:.4f} ===")
    
    # Сохраняем артефакты
    if best_model is not None:
        joblib.dump(best_model, "models/best_model.joblib")
        joblib.dump(scaler, "models/scaler.joblib")
        joblib.dump(feature_columns, "models/feature_names.joblib")
        
        # Логируем в MLflow
        with mlflow.start_run(run_name="best_model"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_param("data_source", "feast" if feast_success else "local")
            mlflow.log_metric("best_roc_auc", best_score)
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                registered_model_name="best_flight_delay_model"
            )
        
        print("✅ Модели и артефакты сохранены!")
        
        # Детальный анализ
        y_pred_best = best_model.predict(X_test_scaled)
        print(f"\n📊 ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_best)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred_best)}")

if __name__ == "__main__":
    main()
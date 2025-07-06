import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# 1. Carga y preprocesamiento de datos
def load_and_process_data(data_path="data"):
    """
    Carga todos los archivos .xlsx de la ruta especificada, los procesa y 
    los combina en un único DataFrame.
    """
    all_files = glob.glob(os.path.join(data_path, "*.xlsx"))
    df_list = []

    for f in all_files:
        try:
            # Extraer estación y año del nombre del archivo
            filename = os.path.basename(f)
            parts = filename.replace(".xlsx", "").split("_")
            station = "_".join(parts[:-1])  # Todo menos el año
            year = int(parts[-1])

            # Leer el archivo de Excel, especificando la fila del encabezado y las columnas a usar
            df_temp = pd.read_excel(f, engine='openpyxl', header=5, usecols="B:N")

            # Limpiar nombres de columnas (eliminar espacios, etc.)
            df_temp.columns = [str(col).strip().lower() for col in df_temp.columns]
            
            # Renombrar primera columna a 'dia' si es necesario
            if 'día' in df_temp.columns:
                df_temp = df_temp.rename(columns={'día': 'dia'})

            # Filtrar filas no deseadas (como 'Tot.')
            df_temp = df_temp[pd.to_numeric(df_temp['dia'], errors='coerce').notna()]
            df_temp['dia'] = df_temp['dia'].astype(int)

            # Transformar de formato ancho a largo
            df_melted = df_temp.melt(id_vars='dia', var_name='mes', value_name='precipitacion')
            
            # Mapear meses a números
            mes_map = {
                'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
            }
            df_melted['mes'] = df_melted['mes'].str.strip().map(mes_map)

            # Eliminar filas con meses inválidos
            df_melted.dropna(subset=['mes'], inplace=True)
            df_melted['mes'] = df_melted['mes'].astype(int)
            df_melted['año'] = year

            # Crear columna de fecha
            df_melted['fecha'] = pd.to_datetime(df_melted.assign(day=df_melted.dia, month=df_melted.mes, year=df_melted.año)[['year', 'month', 'day']], errors='coerce')
            
            # Limpieza final
            df_melted.dropna(subset=['fecha'], inplace=True)
            df_melted['station'] = station
            
            # Limpiar valores de precipitación
            # Reemplazar comas por puntos y convertir a numérico
            if df_melted['precipitacion'].dtype == 'object':
                 df_melted['precipitacion'] = df_melted['precipitacion'].astype(str).str.replace(',', '.').astype(float)
            
            df_melted['precipitacion'] = pd.to_numeric(df_melted['precipitacion'], errors='coerce').fillna(0.0)

            df_list.append(df_melted[['fecha', 'station', 'precipitacion']])

        except Exception as e:
            print(f"Error procesando el archivo {f}: {e}")

    if not df_list:
        print("No se cargaron datos. Revisa la ruta y el formato de los archivos.")
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

# 2. Ingeniería de características
def feature_engineering(df):
    """Crea características basadas en la fecha."""
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia'] = df['fecha'].dt.day
    df['dia_del_año'] = df['fecha'].dt.dayofyear
    df['dia_de_la_semana'] = df['fecha'].dt.dayofweek
    df['semana_del_año'] = df['fecha'].dt.isocalendar().week.astype(int)
    return df

# --- Script Principal ---
if __name__ == "__main__":
    print("1. Cargando y procesando datos...")
    main_df = load_and_process_data()

    if not main_df.empty:
        print("2. Creando características...")
        main_df = feature_engineering(main_df)

        # Filtrar solo las estaciones con al menos 5 años de datos
        estaciones_validas = [
            'C14-Mindo_Captación',
            'M5182F-Mindo_Guagua_Pichincha',
            'P31-Pichán'
        ]
        main_df = main_df[main_df['station'].isin(estaciones_validas)].copy()

        # Convertir 'station' a una variable categórica numérica
        main_df['station_code'] = main_df['station'].astype('category').cat.codes
        station_mapping = dict(zip(main_df['station'], main_df['station_code']))
        print("Estaciones usadas y sus códigos:", station_mapping)

        # Crear variable objetivo binaria: 1 si hay lluvia, 0 si no
        main_df['lluvia_binaria'] = (main_df['precipitacion'] > 0).astype(int)

        # 3. Preparación para el modelo de clasificación
        X = main_df[['station_code', 'año', 'mes', 'dia', 'dia_del_año', 'dia_de_la_semana', 'semana_del_año']]
        y = main_df['lluvia_binaria']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("3. Entrenando el modelo XGBoostClassifier con optimización de hiperparámetros...")
        
        # 4. Optimización de Hiperparámetros con GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }

        xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, use_label_encoder=False)
        grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, 
                                   cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        print("\nMejores hiperparámetros encontrados:")
        print(grid_search.best_params_)

        # 5. Evaluación del modelo
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:,1]
        
        print(f"\nPrecisión (accuracy) en el conjunto de prueba: {accuracy_score(y_test, y_pred):.4f}")
        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))

        # 6. Predicción para el 06/07/2025 en cada estación válida
        print("\n--- Predicción de Lluvia para el 06/07/2025 en cada estación ---")
        print("Diccionario de estaciones disponibles para predicción:", station_mapping)
        fecha_pred = pd.to_datetime('2025-07-06')
        for nombre_estacion in estaciones_validas:
            if nombre_estacion in station_mapping:
                sample_station_code = station_mapping[nombre_estacion]
                sample_data = pd.DataFrame({
                    'station_code': [sample_station_code],
                    'año': [fecha_pred.year],
                    'mes': [fecha_pred.month],
                    'dia': [fecha_pred.day],
                    'dia_del_año': [fecha_pred.dayofyear],
                    'dia_de_la_semana': [fecha_pred.dayofweek],
                    'semana_del_año': [fecha_pred.isocalendar().week]
                })
                proba_lluvia = best_model.predict_proba(sample_data)[0,1]
                pred_lluvia = best_model.predict(sample_data)[0]
                print(f"Estación: {nombre_estacion:35s} | Probabilidad de lluvia: {proba_lluvia*100:5.1f}% | ¿Lloverá? {'Sí' if pred_lluvia==1 else 'No'}")
            else:
                print(f"Estación: {nombre_estacion:35s} | No hay datos suficientes para predecir.")
    else:
        print("El script no pudo continuar porque no se cargaron datos.")

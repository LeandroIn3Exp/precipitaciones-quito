import os
import glob
import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Carga y preprocesamiento de datos (idéntico a main.py, pero solo para entrenamiento)
def load_and_process_data(data_path=None):
    # Usar ruta absoluta relativa al archivo actual si no se pasa data_path
    if data_path is None:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    all_files = glob.glob(os.path.join(data_path, "*.xlsx"))
    df_list = []
    for f in all_files:
        try:
            filename = os.path.basename(f)
            parts = filename.replace(".xlsx", "").split("_")
            station = "_".join(parts[:-1])  # Todo menos el año
            year = int(parts[-1])
            df_temp = pd.read_excel(f, engine='openpyxl', header=5, usecols="B:N")
            df_temp.columns = [str(col).strip().lower() for col in df_temp.columns]
            if 'día' in df_temp.columns:
                df_temp = df_temp.rename(columns={'día': 'dia'})
            df_temp = df_temp[pd.to_numeric(df_temp['dia'], errors='coerce').notna()]
            df_temp['dia'] = df_temp['dia'].astype(int)
            df_melted = df_temp.melt(id_vars='dia', var_name='mes', value_name='precipitacion')
            mes_map = {'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
                       'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12}
            df_melted['mes'] = df_melted['mes'].str.strip().map(mes_map)
            df_melted.dropna(subset=['mes'], inplace=True)
            df_melted['mes'] = df_melted['mes'].astype(int)
            df_melted['año'] = year
            df_melted['fecha'] = pd.to_datetime(df_melted.assign(day=df_melted.dia, month=df_melted.mes, year=df_melted.año)[['year', 'month', 'day']], errors='coerce')
            df_melted.dropna(subset=['fecha'], inplace=True)
            df_melted['station'] = station
            if df_melted['precipitacion'].dtype == 'object':
                df_melted['precipitacion'] = df_melted['precipitacion'].astype(str).str.replace(',', '.').astype(float)
            df_melted['precipitacion'] = pd.to_numeric(df_melted['precipitacion'], errors='coerce').fillna(0.0)
            df_list.append(df_melted[['fecha', 'station', 'precipitacion', 'año', 'mes', 'dia']])
        except Exception as e:
            print(f"Error procesando el archivo {f}: {e}")
    if not df_list:
        print("No se cargaron datos. Revisa la ruta y el formato de los archivos.")
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def feature_engineering(df):
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia'] = df['fecha'].dt.day
    df['dia_del_año'] = df['fecha'].dt.dayofyear
    df['dia_de_la_semana'] = df['fecha'].dt.dayofweek
    df['semana_del_año'] = df['fecha'].dt.isocalendar().week.astype(int)
    return df

if __name__ == "__main__":
    print("Cargando y procesando datos...")
    main_df = load_and_process_data()
    if main_df.empty:
        print("No hay datos para entrenar el modelo.")
        exit(1)

    print("Creando características...")
    main_df = feature_engineering(main_df)

    # Filtrar solo las estaciones con al menos 5 años de datos
    estaciones_validas = [
        'C14-Mindo_Captación',
        'M5182F-Mindo_Guagua_Pichincha',
        'P31-Pichán'
    ]
    main_df = main_df[main_df['station'].isin(estaciones_validas)].copy()

    # Codificar estación
    main_df['station_code'] = main_df['station'].astype('category').cat.codes
    station_mapping = dict(zip(main_df['station'], main_df['station_code']))
    print("Estaciones usadas y sus códigos:", station_mapping)

    # Variable objetivo binaria
    main_df['lluvia_binaria'] = (main_df['precipitacion'] > 0).astype(int)

    X = main_df[['station_code', 'año', 'mes', 'dia', 'dia_del_año', 'dia_de_la_semana', 'semana_del_año']]
    y = main_df['lluvia_binaria']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando el modelo XGBoostClassifier con optimización de hiperparámetros...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }
    xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, use_label_encoder=False)
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("\nMejores hiperparámetros encontrados:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Crear directorio de resultados si no existe
    results_dir = os.path.join(os.path.dirname(__file__), 'model_results')
    os.makedirs(results_dir, exist_ok=True)

    # Guardar el modelo entrenado
    joblib.dump(best_model, os.path.join(results_dir, 'modelo_lluvia.pkl'))
    print(f"Modelo guardado en '{os.path.join(results_dir, 'modelo_lluvia.pkl')}'")

    # Guardar el mapeo de estaciones
    with open(os.path.join(results_dir, 'station_mapping.json'), 'w') as f:
        json.dump(station_mapping, f, indent=4)
    print(f"Mapeo de estaciones guardado en '{os.path.join(results_dir, 'station_mapping.json')}'")

    # Guardar los mejores hiperparámetros
    with open(os.path.join(results_dir, 'mejores_hiperparametros.json'), 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"Hiperparámetros guardados en '{os.path.join(results_dir, 'mejores_hiperparametros.json')}'") 
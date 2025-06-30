import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

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
            station = parts[0]
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

        # Convertir 'station' a una variable categórica numérica
        main_df['station_code'] = main_df['station'].astype('category').cat.codes

        # 3. Preparación para el modelo
        X = main_df[['station_code', 'año', 'mes', 'dia', 'dia_del_año', 'dia_de_la_semana', 'semana_del_año']]
        y = main_df['precipitacion']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("3. Entrenando el modelo XGBoost con optimización de hiperparámetros...")
        
        # 4. Optimización de Hiperparámetros con GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }

        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)
        
        # Nota: GridSearchCV puede tardar bastante. Para una prueba rápida, reduce el param_grid.
        # Por ejemplo: {'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]}
        grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                                   cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
        
        grid_search.fit(X_train, y_train)

        print("\nMejores hiperparámetros encontrados:")
        print(grid_search.best_params_)

        # 5. Evaluación del modelo
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        print(f"\nError Cuadrático Medio (MSE) en el conjunto de prueba: {mse:.4f}")
        print(f"Raíz del Error Cuadrático Medio (RMSE) en el conjunto de prueba: {rmse:.4f}")

        # 6. Ejemplo de predicción
        print("\n--- Ejemplo de Predicción ---")
        # Predicción para una estación y fecha específicas
        # Nota: Debes usar los mismos códigos de estación que el modelo aprendió.
        # Crea un diccionario para mapear nombres de estación a códigos
        station_mapping = dict(zip(main_df['station'], main_df['station_code']))
        print("Mapeo de Estaciones a Códigos:", station_mapping)

        # Ejemplo: predecir para la estación 'C14-Mindo' en una fecha futura
        try:
            sample_station_code = station_mapping['C14-Mindo']
            future_date = pd.to_datetime('2025-10-15')
            
            sample_data = pd.DataFrame({
                'station_code': [sample_station_code],
                'año': [future_date.year],
                'mes': [future_date.month],
                'dia': [future_date.day],
                'dia_del_año': [future_date.dayofyear],
                'dia_de_la_semana': [future_date.dayofweek],
                'semana_del_año': [future_date.isocalendar().week]
            })

            predicted_rain = best_model.predict(sample_data)
            print(f"\nPredicción de lluvia para 'C14-Mindo' el {future_date.date()}: {predicted_rain[0]:.2f} mm")
        except KeyError:
            print("\nNo se pudo hacer la predicción de ejemplo porque la estación 'C14-Mindo' no se encontró en los datos.")
        except Exception as e:
            print(f"\nOcurrió un error en la predicción de ejemplo: {e}")
            
    else:
        print("El script no pudo continuar porque no se cargaron datos.")

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carga y preprocesamiento de datos (igual que en main.py)
def load_and_process_data(data_path="data"):
    all_files = glob.glob(os.path.join(data_path, "*.xlsx"))
    df_list = []
    for f in all_files:
        try:
            filename = os.path.basename(f)
            parts = filename.replace(".xlsx", "").split("_")
            station = parts[0]
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

# 2. Análisis descriptivo y visualizaciones
def eda(df):
    # Crear carpeta plots si no existe
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    print("\n--- Estadísticas Generales ---")
    print(df['precipitacion'].describe())
    print("\n--- Estadísticas por Estación ---")
    print(df.groupby('station')['precipitacion'].describe())
    print("\n--- Estadísticas por Año ---")
    print(df.groupby('año')['precipitacion'].describe())
    print("\n--- Estadísticas por Mes ---")
    print(df.groupby('mes')['precipitacion'].describe())

    # Histograma global
    plt.figure(figsize=(8,4))
    sns.histplot(df['precipitacion'], bins=50, kde=True)
    plt.title('Histograma de Precipitación (mm/día)')
    plt.xlabel('Precipitación (mm)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'histograma_precipitacion.png'))
    plt.close()

    # Boxplot por estación
    plt.figure(figsize=(10,5))
    sns.boxplot(x='station', y='precipitacion', data=df)
    plt.title('Boxplot de Precipitación por Estación')
    plt.xlabel('Estación')
    plt.ylabel('Precipitación (mm)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplot_estacion.png'))
    plt.close()

    # Boxplot por mes
    plt.figure(figsize=(10,5))
    sns.boxplot(x='mes', y='precipitacion', data=df)
    plt.title('Boxplot de Precipitación por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Precipitación (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplot_mes.png'))
    plt.close()

    # Serie temporal de precipitación media mensual
    df_mensual = df.groupby(['año','mes'])['precipitacion'].mean().reset_index()
    # Renombrar columnas para to_datetime
    df_mensual = df_mensual.rename(columns={'año': 'year', 'mes': 'month'})
    df_mensual['fecha'] = pd.to_datetime(df_mensual[['year','month']].assign(day=1))
    plt.figure(figsize=(12,5))
    sns.lineplot(x='fecha', y='precipitacion', data=df_mensual)
    plt.title('Precipitación Media Mensual (todas las estaciones)')
    plt.xlabel('Fecha')
    plt.ylabel('Precipitación media (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'serie_mensual.png'))
    plt.close()

    # Heatmap de precipitación media por mes y año
    tabla_heatmap = df.groupby(['año','mes'])['precipitacion'].mean().unstack()
    plt.figure(figsize=(12,6))
    sns.heatmap(tabla_heatmap, cmap='Blues', annot=True, fmt='.1f')
    plt.title('Mapa de calor: Precipitación media por mes y año')
    plt.xlabel('Mes')
    plt.ylabel('Año')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'heatmap_precipitacion.png'))
    plt.close()

    # Serie temporal por estación (opcional)
    plt.figure(figsize=(12,6))
    for station in df['station'].unique():
        df_station = df[df['station']==station].groupby('fecha')['precipitacion'].mean().reset_index()
        plt.plot(df_station['fecha'], df_station['precipitacion'], label=station)
    plt.title('Serie temporal de precipitación diaria por estación')
    plt.xlabel('Fecha')
    plt.ylabel('Precipitación (mm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'serie_diaria_estaciones.png'))
    plt.close()

    print(f"\nSe han guardado las gráficas como archivos PNG en el directorio '{plots_dir}'.")

if __name__ == "__main__":
    print("Cargando y procesando datos...")
    df = load_and_process_data()
    if not df.empty:
        eda(df)
    else:
        print("No se pudo realizar el análisis por falta de datos.") 
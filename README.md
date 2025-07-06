# Proyecto de Predicción de Lluvia con Machine Learning

## Origen y Descripción de los Datos

Los datos utilizados en este proyecto provienen de la plataforma oficial de la Empresa Pública Metropolitana de Agua Potable y Saneamiento de Quito (EPMAPS): [https://paramh2o.aguaquito.gob.ec/reportes/diario/](https://paramh2o.aguaquito.gob.ec/reportes/diario/)

Se descargaron archivos Excel de precipitaciones diarias para varias estaciones y años. Cada archivo tiene el formato:

```
<estacion>_<año>.xlsx
```

Las columnas representan los meses del año y las filas los días, con la precipitación diaria en milímetros (mm).

## Estructura de Datos
- Carpeta `data/`: contiene todos los archivos `.xlsx` de las estaciones y años.
- Carpeta `plots/`: contiene las gráficas generadas en el análisis exploratorio.
- Carpeta `src/model_results/`: contiene el modelo entrenado (`modelo_lluvia.pkl`), el mapeo de estaciones (`station_mapping.json`) y los mejores hiperparámetros (`mejores_hiperparametros.json`).

## Análisis Exploratorio de Datos (EDA)
- Se realizó un análisis estadístico y visual de las precipitaciones:
  - Estadísticas generales (media, mediana, desviación estándar, máximos, mínimos).
  - Estadísticas por estación, año y mes.
  - Visualizaciones: histogramas, boxplots, series temporales, heatmaps.
- Las gráficas se encuentran en la carpeta `plots/`.
- Se observó que la mayoría de los días tienen poca o ninguna lluvia, pero existen valores extremos (outliers) de lluvias intensas.

## Modelo de Machine Learning
- **Tipo de modelo:** Clasificación binaria (¿lloverá o no?).
- **Algoritmo:** XGBoostClassifier (Gradient Boosting de árboles de decisión).
- **Variables predictoras:** estación, año, mes, día, día del año, día de la semana, semana del año.
- **Variable objetivo:** lluvia binaria (1 si hay precipitación, 0 si no).
- **Optimización:** GridSearchCV para encontrar los mejores hiperparámetros.
- **Entrenamiento:** Solo se usaron estaciones con al menos 5 años de datos (`C14-Mindo_Captación`, `M5182F-Mindo_Guagua_Pichincha`, `P31-Pichán`).
- **Guardado:** El modelo entrenado se guarda en `src/model_results/modelo_lluvia.pkl` junto con el mapeo de estaciones y los hiperparámetros.

## Exposición del Modelo vía API
- Se implementó una API con FastAPI (`src/api.py`).
- La API carga el modelo y permite hacer predicciones de lluvia para cualquier estación y fecha disponible.
- El endpoint principal es `/predict` (POST), que recibe el nombre de la estación y la fecha, y devuelve la probabilidad de lluvia y la predicción binaria.

## Resumen del Flujo
1. **Análisis exploratorio:** Ejecutar `eda.py` para generar estadísticas y gráficas.
2. **Entrenamiento y guardado:** Ejecutar `src/train_and_save_model.py` para entrenar y guardar el modelo.
3. **Exposición como API:** Ejecutar `src/api.py` con Uvicorn para exponer el modelo y hacer predicciones vía HTTP.

---

Para detalles de instalación y uso, consulta el archivo `README_INSTALACION.md`. 
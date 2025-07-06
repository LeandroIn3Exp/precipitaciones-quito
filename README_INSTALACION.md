# Guía de Instalación y Uso de la API de Predicción de Lluvia

## 1. Instalación de dependencias

Asegúrate de tener Python 3.8+ y pip. Luego ejecuta:

```bash
pip install -r requirements.txt
```

## 2. Entrenamiento del modelo (opcional)

Si quieres reentrenar el modelo con nuevos datos, ejecuta:

```bash
python src/train_and_save_model.py
```

Esto generará los archivos necesarios en `src/model_results/`.

## 3. Ejecución de la API

Inicia el servidor FastAPI con Uvicorn:

```bash
uvicorn src.api:app --reload
```

La API estará disponible en: [http://localhost:8000](http://localhost:8000)

## 4. Uso del endpoint `/predict`

- **Método:** POST
- **URL:** `http://localhost:8000/predict`
- **Body (JSON):**

```json
{
  "station": "C14-Mindo_Captación",
  "date": "2025-07-06"
}
```

- **Respuesta esperada (JSON):**

```json
{
  "station": "C14-Mindo_Captación",
  "date": "2025-07-06",
  "probabilidad_lluvia": 0.95,
  "llovera": true
}
```

- Si la estación no existe o la fecha es inválida, la API devolverá un error con un mensaje descriptivo.

## 5. Visualización de resultados

- Las gráficas del análisis exploratorio se encuentran en la carpeta `plots/`.

## 6. Notas adicionales

- Puedes consultar las estaciones disponibles revisando el archivo `src/model_results/station_mapping.json`.
- El modelo solo predice para estaciones con suficiente historial de datos.
- Si quieres modificar el modelo o el flujo, revisa los scripts en la carpeta `src/`. 
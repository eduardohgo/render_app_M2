# Aplicación Flask para Render

Proyecto final de regresión aplicando la metodología CRISP-DM.

## Datos del alumno

- Alumno: Carlos Eduardo Hidalgo Toledo
- Matrícula: 20231335
- Grupo: A
- Profesor: Efrén Juárez Castillo

## Registro del dataset

- Nombre del dataset: Bias correction of numerical prediction model temperature forecast
- Fuente: Kaggle
- URL del dataset: https://www.kaggle.com/datasets/viktorpopov/bias-correction-ucl
- Variable dependiente: `Next_Tmax`
- Descripción de la variable dependiente: temperatura máxima del día siguiente en Seúl, Corea del Sur, medida en grados Celsius.

## Qué hace la aplicación

La aplicación recibe seis variables originales del dataset relacionadas con temperatura y nubosidad.
Después carga el pipeline entrenado y genera la predicción de `Next_Tmax`.

El formulario pide valores entendibles para el usuario. No se solicitan valores escalados, columnas codificadas, variables internas ni componentes principales.

## Modelo desplegado

- Mejor escenario: Características seleccionadas
- Mejor modelo: KNeighborsRegressor
- R² final con train_test_split: 0.886120
- MAE final con train_test_split: 0.735499 °C

## Estructura de archivos

```text
render_app/
├── app.py
├── requirements.txt
├── Procfile
├── .python-version
├── README_RENDER.md
├── URL_Render.txt
├── models/
│   ├── best_model.joblib
│   └── metadata.json
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Prueba local

```bash
pip install -r requirements.txt
python app.py
```

Después abrir:

```text
http://127.0.0.1:5000
```

## Comandos para Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
gunicorn app:app
```

## Valores de prueba

- LDAPS_Tmax_lapse: 29.5
- Present_Tmax: 30.0
- LDAPS_CC3: 0.18
- LDAPS_Tmin_lapse: 23.4
- LDAPS_CC2: 0.25
- LDAPS_CC1: 0.20

Con estos valores, la aplicación debe generar una predicción de temperatura máxima aproximada en grados Celsius.

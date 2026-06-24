from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

# ------------------------------------------------------------
# Proyecto de regresion con CRISP-DM
# Alumno: Carlos Eduardo Hidalgo Toledo
# Objetivo: desplegar en Render el mejor modelo entrenado en la libreta.
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

app = Flask(__name__)

# Se carga el pipeline completo. El modelo ya incluye el preprocesamiento
# necesario y las caracteristicas seleccionadas durante el proyecto.
model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

STUDENT = {
    "name": "Carlos Eduardo Hidalgo Toledo",
    "matricula": "20231335",
    "group": "A",
    "teacher": "Efrén Juárez Castillo",
}

PROJECT_INFO = {
    "dataset_name": metadata.get(
        "dataset_name",
        "Bias correction of numerical prediction model temperature forecast",
    ),
    "dataset_url": metadata.get(
        "kaggle_page_url",
        "https://www.kaggle.com/datasets/viktorpopov/bias-correction-ucl",
    ),
    "target": metadata.get("target", "Next_Tmax"),
    "target_description": metadata.get(
        "target_description",
        "Temperatura máxima del día siguiente en Seúl, Corea del Sur, medida en grados Celsius.",
    ),
    "methodology": "CRISP-DM",
    "best_scenario": metadata.get("best_scenario", "Características seleccionadas"),
    "best_model": metadata.get("best_model", "KNeighborsRegressor"),
}

# Estos son los datos originales que el usuario debe registrar en el formulario.
# No se piden valores escalados ni componentes principales; el usuario escribe
# valores entendibles del clima, y el pipeline se encarga del resto.
FIELDS = [
    {
        "name": "LDAPS_Tmax_lapse",
        "label": "Temperatura máxima estimada por LDAPS",
        "unit": "°C",
        "min": -10,
        "max": 45,
        "step": "0.01",
        "example": "29.5",
        "description": "Pronóstico previo de la temperatura máxima del día siguiente.",
    },
    {
        "name": "Present_Tmax",
        "label": "Temperatura máxima actual",
        "unit": "°C",
        "min": -10,
        "max": 45,
        "step": "0.01",
        "example": "30.0",
        "description": "Temperatura máxima observada en el día actual.",
    },
    {
        "name": "LDAPS_CC3",
        "label": "Nubosidad LDAPS periodo 3",
        "unit": "valor entre 0 y 1",
        "min": 0,
        "max": 1,
        "step": "0.001",
        "example": "0.18",
        "description": "Cobertura de nubes estimada en el tercer periodo del pronóstico.",
    },
    {
        "name": "LDAPS_Tmin_lapse",
        "label": "Temperatura mínima estimada por LDAPS",
        "unit": "°C",
        "min": -10,
        "max": 35,
        "step": "0.01",
        "example": "23.4",
        "description": "Pronóstico previo de la temperatura mínima del día siguiente.",
    },
    {
        "name": "LDAPS_CC2",
        "label": "Nubosidad LDAPS periodo 2",
        "unit": "valor entre 0 y 1",
        "min": 0,
        "max": 1,
        "step": "0.001",
        "example": "0.25",
        "description": "Cobertura de nubes estimada en el segundo periodo del pronóstico.",
    },
    {
        "name": "LDAPS_CC1",
        "label": "Nubosidad LDAPS periodo 1",
        "unit": "valor entre 0 y 1",
        "min": 0,
        "max": 1,
        "step": "0.001",
        "example": "0.20",
        "description": "Cobertura de nubes estimada en el primer periodo del pronóstico.",
    },
]

EXAMPLE_VALUES = {field["name"]: field["example"] for field in FIELDS}


def parse_form(form):
    """Lee, valida y convierte los datos ingresados en el formulario."""
    values = {}
    errors = []

    for field in FIELDS:
        raw_value = form.get(field["name"], "").strip()

        if raw_value == "":
            errors.append(f"Falta capturar el dato: {field['label']}.")
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{field['label']} debe escribirse como número.")
            continue

        if value < field["min"] or value > field["max"]:
            errors.append(
                f"{field['label']} debe estar entre {field['min']} y {field['max']} {field['unit']}."
            )
            continue

        values[field["name"]] = value

    return values, errors


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = []
    submitted = {}

    if request.method == "POST":
        submitted, errors = parse_form(request.form)

        if not errors:
            input_data = pd.DataFrame([submitted], columns=metadata["features"])
            prediction = float(model.predict(input_data)[0])

    return render_template(
        "index.html",
        fields=FIELDS,
        metadata=metadata,
        project=PROJECT_INFO,
        student=STUDENT,
        prediction=prediction,
        errors=errors,
        submitted=submitted,
        example_values=EXAMPLE_VALUES,
    )


@app.route("/health")
def health():
    """Ruta sencilla para comprobar que la aplicacion esta activa en Render."""
    return {"status": "ok", "model": PROJECT_INFO["best_model"]}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

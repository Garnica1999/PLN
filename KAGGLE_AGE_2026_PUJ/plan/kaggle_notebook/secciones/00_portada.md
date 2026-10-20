# Predicción de Rango de Edad en Tuits en Español — Notebook final de entrega

**Competencia:** `tweet-age-2026-puj` (Kaggle, organizada por `NLP_Course_PUJ`)  
**Programa:** Maestría en Inteligencia Artificial — Procesamiento de Lenguaje Natural, Pontificia Universidad Javeriana  
**Equipo:** Grupo 2 - PLN (`carlosgarnica`)  
**Métrica oficial:** macro-F1 sobre 6 rangos de edad ordenados (`13-17`, `18-24`, `25-34`, `35-49`, `50-64`, `65-xx`)  
**Modelo entregado:** **V48 Geometric Product-of-Experts + regla plug-in** — leaderboard público **0.63724** (puesto #1; +0.00196 sobre el segundo lugar al cierre, 0.63528)

---

## Resumen ejecutivo

Este notebook reconstruye, ejecuta y documenta **toda** la experimentación del proyecto: desde los primeros
baselines de TF-IDF (Salton y Buckley, «Term-Weighting Approaches in Automatic Text Retrieval», Information Processing & Management, 1988) hasta el campeón final, que combina seis modelos diversos mediante un consenso bayesiano
multiplicativo en espacio logarítmico (*Product-of-Experts*; Hinton, «Training Products of Experts by Minimizing Contrastive Divergence», Neural Computation, 2002) más una regla de decisión por clase congelada
(*plug-in*; Lipton, Elkan y Narayanaswamy, «Thresholding Classifiers to Maximize F1 Score», arXiv, 2014). El proyecto se desarrolló bajo cinco restricciones inviolables: nada de transformadores ni LLM,
nada de embeddings (ni preentrenados ni entrenados desde cero) y el conjunto de prueba intocable — todo se
entrenó desde cero sobre el corpus oficial.

El resultado final: **macro-F1 0.63724 en el leaderboard público de Kaggle** (primer lugar del torneo) y
**0.6186451 de validación cruzada agrupada** local, con una mejora acumulada de **+0.08489** sobre el primer
baseline (0.55235). El notebook reproduce la cadena completa desde los archivos CSV crudos: no depende de
ningún script, artefacto ni repositorio externo.

| Hito | Versión | CV local | Kaggle |
|---|---|---|---|
| Primer baseline (TF-IDF word+char, LinearSVC) | v01 | 0.5344 | 0.55235 |
| Meta-modelo con 14 miembros y combinadores | v12 | 0.6005 | 0.62182 |
| Refinamiento con Optuna y árboles | V30 | 0.6134471 | 0.63302 |
| Regla plug-in por clase | V30 + plug-in | **0.6159732** | **0.63575** |
| Consenso geométrico de 6 modelos | **V48** | **0.6186451** | **0.63724** |

## Protocolo de reproducibilidad

Este notebook es **independiente y determinista**:

1. **Corpus autónomo.** La primera celda localiza los archivos de la competencia (datos adjuntos de Kaggle,
   copias locales o descarga directa vía API oficial). No se clona ningún repositorio.
2. **Cadena embebida.** Todo el código de reconstrucción (representaciones, 14 miembros, features,
   meta-modelo, V20/V30, regla plug-in y redes neuronales) viaja dentro del notebook; cada módulo es un
   porte fiel del script original del proyecto, con su referencia en el docstring.
3. **Protocolo reproducible.** La partición canónica de validación (5 folds agrupados por clústeres de
   near-duplicados) se genera en la propia ejecución con la regla documentada —`StratifiedGroupKFold`,
   `shuffle=True`, semilla 42— y se verifica por propiedades; los pesos de las reglas de decisión son
   constantes legibles del protocolo. Ejecutar dos veces produce exactamente los mismos valores.
4. **Trazabilidad.** Cada experimento reporta su resultado con la fuente documental del proyecto
   (registro de corridas, actas de decisión e informes), y el notebook verifica que lo reconstruido
   coincide con lo documentado.

**Tiempo de ejecución completo:** del orden de 4 horas (entre 3 y 5) en CPU de 8 hilos + GPU (los
componentes neuronales usan GPU si está disponible; sin ella corren en CPU con diferencias del orden
de la precisión de máquina, documentadas en la sección 6.2).

## Guía de lectura

El notebook sigue la metodología **CRISP-ML** (Studer et al., «Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology», Machine Learning and Knowledge Extraction, 2021):

- **Fase 1 — Comprensión del problema:** la tarea, la métrica y las restricciones.
- **Fase 2 — Comprensión de los datos:** auditoría del corpus, análisis exploratorio y la fuga por
  near-duplicados.
- **Fase 3 — Preparación y representación:** la fase histórica V1–V12 re-ejecutada (representaciones,
  miembros y features; el meta-modelo V12 y V13–V17 quedan documentados).
- **Fase 4 — Modelado:** V18–V29 documentado; V30, la regla plug-in, los fracasos seleccionados y los
  componentes neuronales V38–V43 re-ejecutados; falsaciones V44–V52 documentadas; arco 2026 (V53–V57:
  componentes nuevos, consensos y auditoría matemática) re-ejecutado en la sección 4.6.
- **Fase 5 — Evaluación:** validación, el campeón V48, análisis de error y análisis matemáticos.
- **Fase 6 — Despliegue:** la submission final con sus verificaciones.

Cada experimento se explica en lenguaje claro: qué se hizo, por qué, para qué, qué resultado dio y qué
decisión se tomó, enlazando siempre con lo aprendido antes (el EDA y los experimentos previos).

## 0.1 Entorno de ejecución

```python
# Entorno: semillas fijas, estilo de figuras y utilidades basicas del notebook.
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEMILLA = 42
os.environ["PYTHONHASHSEED"] = str(SEMILLA)
random.seed(SEMILLA)
np.random.seed(SEMILLA)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

CLASSES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]

import sklearn  # noqa: E402  (se importa aqui solo para reportar la version)

print("Entorno listo | Python:", sys.version.split()[0],
      "| numpy:", np.__version__, "| pandas:", pd.__version__,
      "| scikit-learn:", sklearn.__version__)
try:
    import torch
    print("torch:", torch.__version__, "| GPU disponible:", torch.cuda.is_available())
except ImportError:
    print("torch no disponible: los componentes neuronales correran en CPU numpy equivalente")
```

## 0.2 Resolución autónoma del corpus

Esta sección localiza los tres archivos oficiales (`age_train.csv`, `age_test.csv` y la plantilla de envío).
El orden de búsqueda es: datos adjuntos de la competencia en Kaggle, copias locales habituales y, si no
aparecen, descarga directa con la API oficial (requiere credenciales de Kaggle en el entorno). Todo ocurre
dentro del notebook, sin dependencias externas.

```python
# Resolucion autonoma del corpus: Kaggle adjunto -> copias locales -> API oficial.
def _buscar_en(base, nombres):
    """Devuelve la primera ruta existente de una lista de nombres dentro de una base."""
    for nombre in nombres:
        ruta = base / nombre
        if ruta.is_file():
            return ruta
    return None


def _descarga_api(destino):
    """Descarga y descomprime el corpus oficial via la API de Kaggle (solo stdlib)."""
    import urllib.request
    import zipfile

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        credencial = Path.home() / ".kaggle" / "kaggle.json"
        if credencial.is_file():
            datos = json.loads(credencial.read_text(encoding="utf-8"))
            token = str(datos.get("token", "")).strip()
    if not token:
        raise RuntimeError(
            "No hay corpus local ni credenciales de Kaggle (KAGGLE_API_TOKEN o ~/.kaggle/kaggle.json)")
    url = "https://www.kaggle.com/api/v1/competitions/data/download-all/tweet-age-2026-puj"
    peticion = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / "tweet-age-2026-puj.zip"
    with urllib.request.urlopen(peticion, timeout=300) as respuesta:
        archivo.write_bytes(respuesta.read())
    with zipfile.ZipFile(archivo) as comprimido:
        comprimido.extractall(destino)
    return destino


def resolver_corpus():
    """Devuelve (train, test, submission) localizando o descargando el corpus oficial."""
    bases = []
    if Path("/kaggle/input").exists():
        bases.extend(sorted(p for p in Path("/kaggle/input").iterdir() if p.is_dir()))
    bases.extend([Path("data/raw"), Path("../data/raw"), Path.cwd()])
    for base in bases:
        train = _buscar_en(base, ("age_train.csv", "Age_train.csv"))
        test = _buscar_en(base, ("age_test.csv", "Age_test.csv"))
        envio = _buscar_en(base, ("age_submission.csv", "sample_submission.csv"))
        if train and test:
            return train, test, envio
    print("Corpus no encontrado localmente: descargando via API de Kaggle...")
    base = _descarga_api(Path("data/raw"))
    train = _buscar_en(base, ("age_train.csv",))
    test = _buscar_en(base, ("age_test.csv",))
    envio = _buscar_en(base, ("age_submission.csv",))
    if not (train and test):
        raise FileNotFoundError("La descarga no produjo age_train.csv / age_test.csv")
    return train, test, envio


RUTA_TRAIN, RUTA_TEST, RUTA_ENVIO = resolver_corpus()
print("train :", RUTA_TRAIN)
print("test  :", RUTA_TEST)
print("envio :", RUTA_ENVIO if RUTA_ENVIO else "(se generara desde los ids de test)")
```

```python
# Carga del corpus con la convencion exacta del proyecto (separador ';', utf-8-sig, texto limpio).
train = pd.read_csv(RUTA_TRAIN, sep=";", encoding="utf-8-sig")
test = pd.read_csv(RUTA_TEST, sep=";", encoding="utf-8-sig")
train["text"] = train["text"].fillna("").astype(str).str.strip()
test["text"] = test["text"].fillna("").astype(str).str.strip()

assert train.shape == (169336, 3), f"train inesperado: {train.shape}"
assert test.shape == (112892, 2), f"test inesperado: {test.shape}"
assert list(train.columns) == ["id", "text", "age_range"], train.columns

y = train["age_range"].astype(str).to_numpy()
clase = np.array([CLASSES.index(valor) for valor in y])

print(f"train: {len(train):,} filas | test: {len(test):,} filas")
print("distribucion de clases (train):")
for etiqueta, cuenta in train["age_range"].value_counts().reindex(CLASSES).items():
    print(f"  {etiqueta}: {cuenta:,} ({cuenta / len(train) * 100:.2f}%)")
```

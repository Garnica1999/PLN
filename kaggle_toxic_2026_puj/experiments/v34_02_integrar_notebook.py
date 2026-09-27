"""Integra la fase V34 (frontier audit) en el notebook consolidado.

Anade la seccion 30 al final de `notebooks/10_final_pipeline.ipynb`. Las celdas de codigo leen
el JSON en tiempo de ejecucion y no llevan outputs embebidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v34_02_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v34_01_frontera.json"

TITULO = """\
## 30. Fase v34: FRONTIER AUDIT (2026-09-27)

La pregunta de la fase, dicha con las palabras de la directiva:

> Que fuente de informacion o mecanismo de aprendizaje todavia no hemos probado y que podria
> producir predicciones diferentes a V03 incluso cuando el modelo se entrena sobre todos los datos
> disponibles?

La respuesta corta: **la mayoria de los ejes estan cerrados, pero hay cinco fuentes genuinamente
sin probar**, y el top 3 priorizado es SGD con perdidas robustas, hashing trick y BM25 weighting.
"""

FRONTERA = """\
### 30.1 Tabla de frontera

| eje | experimentos | estado |
|---|---:|---|
| LINEAL | 138 | CLOSED |
| KERNEL | 5 | CLOSED |
| BOOSTING | 3 | CLOSED |
| ENSEMBLE | 73 | CLOSED |
| DEEP | 36 | CLOSED |
| EMBEDDINGS | 32 | PARTIALLY_EXPLORED |
| LABEL_NOISE | 18 | CLOSED |
| FUZZY_MATCHING | 18 | PARTIALLY_EXPLORED |
| LABEL_SPECIALIZATION | 50 | CLOSED |
| STRUCTURAL | 50 | CLOSED |
| THRESHOLD | 120 | CLOSED |
| REPRESENTACION | 107 | CLOSED |
| META | 70 | CLOSED |
| SEMI_SUPERVISED | 0 | **UNEXPLORED** |
| METRIC | 1 | PARTIALLY_EXPLORED |
| TUNING | 4 | PARTIALLY_EXPLORED |
| VALIDACION | 69 | CLOSED |
| AUDITORIA | 30 | CLOSED |
| PSEUDO_AUTOR | 19 | CLOSED |
| CLUSTER | 110 | CLOSED |
| FEATURES | 49 | CLOSED |
| SUBWORD | 6 | CLOSED |
| OBJECTIVE | 15 | CLOSED |
| MULTI_LABEL | 5 | CLOSED |
"""

OCHO = """\
### 30.2 Las ocho direcciones especificas

| direccion | estado | que falta |
|---|---|---|
| 1. Lineales avanzados | CERRADA | **SGD, L1/Elastic, FTRL, Passive-Aggressive NO probados**. LinearSVC y NB-SVM probados |
| 2. Supervision de features | CERRADA | log-count ratios probado (E2-15, negativo). **class-dependent scaling NO probado** |
| 3. Representaciones subword | CLOSED | **hashing trick, BM25, binarizacion NO probados**. char n-grams probados |
| 4. Embeddings desde cero | PARTIALLY_EXPLORED | FastText y word2vec probados (EBD, cerrado). **Embeddings supervisados NO probados** |
| 5. Objetivos supervisados | CLOSED | focal y class-balanced probados (cerrados). **margin/robust NO probado** |
| 6. Semi-supervised | **UNEXPLORED** | NO probado. La directiva manda no asumir que el pseudo-labeling es legal |
| 7. Contrastive/metric | PARTIALLY_EXPLORED | NO probado. Requiere adaptacion compatible con las restricciones |
| 8. Multi-etiqueta | CLOSED | condicionalizacion y B2 probados (V6-02, cerrado). **Otras formulaciones NO probadas** |
"""

TOP5 = """\
### 30.3 Top 5 de familias realmente abiertas

**1. SGD con perdidas robustas.** Cambia el inductive bias: no es un solver convexo completo como
liblinear, es un metodo online con penalizacion distinta. Ningun SGD aparece en el registry.
Experimento minimo: SGDClassifier con hinge y modified_huber, mismos features de V03. Coste bajo,
riesgo de overfitting bajo, riesgo de repeticion bajo.

**2. Hashing trick.** Cambia la representacion fundamental: no hay vocabulario, la dimensionalidad
es fija y las colisiones son parte del metodo. NO probado. Experimento minimo: HashingVectorizer
con 2^18 features. Coste bajo, riesgo de overfitting medio.

**3. BM25 weighting.** Cambia la representacion fundamental: BM25 normaliza por longitud del
documento de forma explicita, lo que es distinto a sublinear_tf. NO probado. Experimento minimo:
BM25 sobre los mismos n-grams. Coste bajo, riesgo de overfitting bajo.

**4. Embeddings supervisados desde cero.** Cambia la fuente de informacion: los embeddings se
entrenan con las etiquetas, no solo con la co-ocurrencia. FastText y word2vec probados fueron no
supervisados. Riesgo de overfitting medio-alto con 1731 documentos.

**5. Passive-Aggressive.** Cambia el inductive bias: metodo online con margen, no un solver
convexo. NO probado. Similar a SGD pero con actualizacion pasiva cuando acierta.
"""

TOP3 = """\
### 30.4 Top 3 priorizadas

El criterio es `novedad real x plausibilidad x coste x capacidad de transferir a full-train`, no
intuicion de "esto suena moderno".

| puesto | familia | novedad | plausibilidad | coste | transferencia | riesgo |
|---|---|---|---|---|---|---|
| 1 | SGD con perdidas robustas | alta (0 en 280) | alta (cambia inductive bias) | bajo | alta (es un solver) | bajo |
| 2 | Hashing trick | alta (0 en 280) | media (colisiones) | bajo | media | medio |
| 3 | BM25 weighting | alta (0 en 280) | alta (longitud, donde V03 falla) | bajo | alta | bajo |

**No se incluyen** embeddings supervisados (riesgo de overfitting alto con 1731 documentos) ni
Passive-Aggressive (similar a SGD, menos novedoso).
"""

PRIMERO = """\
### 30.5 Primer experimento de la siguiente fase

**SGD con perdidas robustas.** Es el de mayor valor informativo: cambia el inductive bias, es el
mas barato, y tiene el menor riesgo de repeticion encubierta (ningun SGD en 280 filas).

Si SGD con hinge o modified_huber produce predicciones diferentes a V03 en OOF, habra una fuente
nueva de senal. Si no, se cierra y se pasa al hashing trick.

### 30.6 Lo que esta fase no hace

- No ejecuta ningun modelo.
- No genera submission.
- No modifica V03.
- No abre nuevas features.
"""

CODIGO_FRONTERA = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "v34_01_frontera.json").open(encoding="utf-8") as f:
    d = json.load(f)

print("30.1  TABLA DE FRONTERA")
print(f"  {'eje':<24}{'experimentos':>14}{'estado':>22}")
for nombre, v in d["clasificacion_por_eje"].items():
    print(f"  {nombre:<24}{v['n_experimentos']:>14}{v['estado']:>22}")
print()
print("30.2  LAS OCHO DIRECCIONES")
for nombre, v in d["ocho_direcciones"].items():
    estado = v.get("estado") or v.get("estado_general", "")
    print(f"  {nombre}: {estado}")
print()
print("30.3  TOP 5 DE FAMILIAS REALMENTE ABIERTAS")
for t in d["top5_familias_abiertas"]:
    print(f"  {t['puesto']}. {t['familia']}")
    print(f"     novedad: {t['riesgo_repeticion']} | coste: {t['coste']} | overfitting: {t['riesgo_overfitting']}")
print()
print("30.4  TOP 3 PRIORIZADAS")
print("  1. SGD con perdidas robustas  (0 experimentos en 280 filas)")
print("  2. Hashing trick               (0 experimentos en 280 filas)")
print("  3. BM25 weighting              (0 experimentos en 280 filas)")
print()
print("30.5  PRIMER EXPERIMENTO")
print("  SGD con perdidas robustas: cambia el inductive bias, coste bajo, riesgo bajo.")
"""


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin outputs.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def main() -> None:
    """Anade las celdas de la fase V34 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe {INFORME}, ejecuta antes v34_01")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "FRONTIER AUDIT" in marcas:
        print("  la seccion 30 ya existe, no se duplica")
        return
    notebook["cells"].extend(
        [
            markdown(TITULO),
            markdown(FRONTERA),
            markdown(OCHO),
            markdown(TOP5),
            markdown(TOP3),
            markdown(PRIMERO),
            codigo(CODIGO_FRONTERA),
        ]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 30 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()

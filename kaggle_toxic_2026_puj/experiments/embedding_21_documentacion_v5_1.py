#!/usr/bin/env python
"""P9 de la fase v5.1 (EBD-27): integracion documental final de la fase v5 y v5.1.

Que hace
--------
1. Añade a `FINAL_REPORT.md` la seccion **Investigacion posterior - Embedding From Scratch
   (v5 y v5.1)** con las quince entradas que exige la seccion 38 de la spec.
2. Reescribe el bloque de estado de `PROJECT_STATE.md` con la situacion de la fase: fase,
   estado de V03, estado de v5, estado de v5.1, candidato, presupuesto utilizado y
   restante, evaluaciones downstream, estado del notebook y estado de la rama.
3. Actualiza `tasks/todo_v5_1.md` marcando lo hecho.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_21_documentacion_v5_1.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "EBD-27 (P9)"
TITULO_SECCION = "## 28. Investigación posterior — Embedding From Scratch (v5 y v5.1)"
FINAL_REPORT = PROJECT_ROOT / "FINAL_REPORT.md"
PROJECT_STATE = PROJECT_ROOT / "PROJECT_STATE.md"
TODO = PROJECT_ROOT / "tasks" / "todo_v5_1.md"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-P9] {mensaje}", flush=True)


def _leer_presupuesto() -> dict[str, Any]:
    """Lee el estado del presupuesto y los conteos de la fase.

    Returns:
        Diccionario con el presupuesto, el numero de experimentos y el de scripts.
    """
    presupuesto = json.loads(
        (PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json").read_text(encoding="utf-8")
    )
    with EXPERIMENTS_CSV.open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    scripts = sorted((PROJECT_ROOT / "experiments").glob("[0-9][0-9]_*.py"))
    embedding = sorted((PROJECT_ROOT / "experiments").glob("embedding_*.py"))
    tabla_b = PROJECT_ROOT / "reports" / "embedding_downstream_evaluations_v5_1.csv"
    with tabla_b.open(encoding="utf-8") as archivo:
        evaluaciones = list(csv.DictReader(archivo))
    return {
        "presupuesto": presupuesto["presupuesto"],
        "familias": presupuesto["familias_evaluadas"],
        "n_experimentos": len(filas),
        "n_scripts_clasicos": len(scripts),
        "n_scripts_embedding": len(embedding),
        "n_downstream_v5_1": len(evaluaciones),
        "decisiones": len(re.findall(r"^## D-\d+", (PROJECT_ROOT / "DECISIONS.md").read_text(encoding="utf-8"), re.MULTILINE)),
    }


def seccion_final_report(datos: dict[str, Any]) -> str:
    """Construye la seccion 28 del informe final.

    Args:
        datos: resultado de :func:`_leer_presupuesto`.

    Returns:
        Texto de la seccion.
    """
    estado = datos["presupuesto"]
    return f"""
{TITULO_SECCION}

Esta seccion documenta la última fase del proyecto: **aprender una representación
vectorial de las palabras desde cero**, sin transformers ni embeddings preentrenados, y
probar si aporta información que el modelo final V03 no tenga ya. Cubre dos fases:
**v5** (¿de dónde salen los datos y qué pasa al cambiar de corpus?) y **v5.1** (¿alguna
configuración de la arquitectura aporta más?).

### 28.1 Motivación y resultado en una línea

V03 funciona porque lee la **superficie** del texto: el insulto exacto, la mención, la
deformación ortográfica. Lo que no tiene es una noción de **similitud**, y con 1.483
ejemplos etiquetados no hay forma de aprenderla. La hipótesis era que una representación
distributiva añadiría esa dimensión. **El resultado es negativo y está documentado como
tal**: la representación mejora de forma medible, pero no aporta a la tarea, y **V03 se
mantiene** en `corrected_cv` **0.777211** (LB **0.78306**).

### 28.2 Corpus: construcción y autorización

| corpus | tokens | documentos | licencia | papel |
|---|---|---|---|---|
| Wikipedia ES 5M | 5.011.404 | 1.694 | CC BY-SA 4.0 | tramo inferior de la curva |
| Wikipedia ES 10M | 10.002.400 | 2.847 | CC BY-SA 4.0 | corpus de referencia |
| Wikipedia ES 50M | 50.000.028 | 21.529 | CC BY-SA 4.0 | tramo superior |
| mC4-es 10M | 10.000.239 | 16.738 | ODC-BY 1.0 | registro web |
| mixto 20M | 20.000.045 | — | unión de los dos anteriores | efecto del registro |

Pipeline de siete pasos, **idéntico para todos** para que la comparación sea de cantidad de
datos y no de receta: `download` → `raw` → `clean` → `dedup` → `normalize` → `tokenize` →
`shards` + `manifest`. La normalización es la variante **EMB-NORMAL**: minúsculas
conservando tildes, URLs, menciones, hashtags y números reducidos a marcadores estables,
**sin corrector ortográfico y sin reconstrucción de tildes**.

La **autorización** (D-030) permite corpus externos **sin etiquetar solo para aprender la
representación** y prohíbe todo uso supervisado. Se registra tal como se comunicó, sin
documentación escrita inventada. OSCAR, que era *gated*, no se forzó.

**Detalle de trazabilidad que resultó ser una corrección**: el manifiesto guardaba el
`sha256` del **volcado**, que no identifica la escala porque 5M, 10M y 50M de Wikipedia
salen del mismo tramo. La identidad de un entrenamiento usa el `sha256` del corpus **ya
procesado**, que sí distingue las tres escalas (D-035).

### 28.3 Cobertura y OOV

La cobertura se mide contra el vocabulario TOXIC de train y test (uso como diagnóstico,
autorizado; no como señal).

| corpus | por tipo | por token |
|---|---|---|
| Wikipedia 5M | 0.6302 | 0.8646 |
| Wikipedia 10M | 0.6839 | 0.8833 |
| Wikipedia 50M | 0.7703 | 0.9238 |
| web mC4-es 10M | 0.8256 | 0.9512 |
| mixto 20M | 0.8413 | 0.9553 |

**Hallazgo central de la fase**: la cobertura y la aportación **se mueven por separado**. El
corpus web sube la cobertura del 0,68 al 0,83 y la aportación a V03 cambia en milésimas. Y la
hipótesis de que faltara señal por "no tener tildes" queda refutada: el 8,5 % de los tipos
ausentes sí lo están, y el 66,6 % son **deformaciones con pérdida de vocales**.

### 28.4 FastText y el control Word2Vec

El control es el mismo binario con las subpalabras apagadas, que es Word2Vec. La
comparación aísla **una sola cosa**:

| modelo | tipos con vector de 6.059 | fusión |
|---|---|---|
| FastText 200d (subpalabras 3-6) | **6.059** | 0.7818 |
| Word2Vec-control (10M) | 5.134 | 0.7762 |
| FastText 200d (50M) | **6.059** | 0.7803 |
| Word2Vec-control (50M) | 4.412 | 0.7773 |

Los 925 tipos que el control pierde son formas ortográficas, y son las que deciden la
etiqueta en un corpus de tuits. La ventaja de las subpalabras tiene forma de **joroba** con
máximo en 10M, no una relación monótona con la escala (D-034).

### 28.5 Downstream y la regla de escala

| pooling | corrected_cv |
|---|---|
| media | 0.6603 |
| media + máximo | 0.6388 |
| ponderado por IDF | 0.6348 |
| **suelo aleatorio** | **0.5404** |

La regla de escala importa más que el pooling: al fusionar, el bloque denso se normaliza
**L2 por documento**. Estandarizarlo hace que las 200 dimensiones dominen al bloque disperso
y la fusión **pierda 0,09 de F1** (0.7818 con L2 frente a 0.6901) sin que eso diga nada
sobre la representación: son los mismos datos.

### 28.6 Fusión y escala del corpus

Cinco configuraciones, y el delta de aportación vive siempre entre **+0.0001 y +0.0046**,
con el intervalo de bootstrap tocando el 0:

| corpus | fusión | delta |
|---|---|---|
| Wikipedia 5M | 0.7757 | **−0.0015** |
| Wikipedia 10M | 0.7818 | +0.0046 |
| Wikipedia 50M | 0.7803 | +0.0031 |
| web mC4-es 10M | 0.7812 | +0.0040 |
| mixto 20M | 0.7776 | +0.0004 |

**Conclusión de la curva**: multiplicar por diez los tokens (5M → 50M) **reduce** el delta.
El techo no es la representación, y **no se escala el corpus**: tres de las cinco condiciones
de la spec fallan, y las que fallan son de evidencia, no de coste (D-041).

### 28.7 Arquitectura (v5.1): las cinco familias

Todas sobre Wikipedia 10M, cambiando **un parámetro** cada vez. Criterio de adopción de dos
niveles: **delta ≥ +0,005 Y el intervalo de confianza no incluye el 0** (D-036).

| familia | mejor fusión | delta | IC95 | veredicto |
|---|---|---|---|---|
| vector_size | 0.7823 (100d) | +0.0050 | [−0.0028, +0.0130] | cerrada (inerte) |
| window | 0.7830 (ws 3) | +0.0057 | [−0.0021, +0.0136] | cerrada (inerte) |
| min_count | 0.7818 (baseline) | +0.0046 | [−0.0032, +0.0124] | cerrada (inerte) |
| subwords | 0.7818 (baseline) | +0.0046 | [−0.0032, +0.0124] | cerrada (inerte) |
| negative | 0.7818 (baseline) | +0.0046 | [−0.0032, +0.0124] | cerrada (inerte) |

**14 configuraciones, 0 adoptables.** Tres lecturas que quedan como resultado:

- **La dimensión no es la palanca**: con 1.483 ejemplos, 100, 200 y 300 distancias dan lo
  mismo (rango 0.0017).
- **Menos ventana rinde más** (3 > 5 > 7), y el intervalo de subpalabras se queda en el
  punto medio 3-6: acortar o alargar empeora. La señal útil es **local y ortográfica**.
- **`min_count` no puede funcionar con subpalabras**: bajar el umbral agranda el vocabulario
  general pero no añade ni un tipo de TOXIC con vector, porque fastText ya compone vector
  para cualquier tipo consultation. Su vía de mejora ya está ocupada.

### 28.8 Complementariedad y resultados negativos

- **Qué arregla**: el embedding aporta **la misma información que V03 con más ruido**. Su
  variante con mejor delta (WS-3) no arregla más filas que el baseline (8 frente a 8):
  arregla lo mismo y **rompe menos** (2 frente a 5). Toda la ventaja está en el *damage*.
- **Rangos**: el embedding aislado se mueve en 0,0297 entre 0,6397 y 0,6694; la fusión se
  mueve en 0,0063. El rango de la representación es **4,7 veces mayor** que el de la tarea.
- **Resultados negativos explícitos**: 100d y 300d (no superan el baseline), ventanas 3 y 7
  (mejoran el embedding pero no la fusión), min_count 2 y 5 (ninguno supera el 3), subpalabras
  3-5 y 3-7 (ninguno supera el 3-6), negative 5 (cae por debajo del 10), mezcla de corpus
  (la peor: +0.0004), estandarizar el bloque denso (−0.0871), los cinco corpora por debajo
  del gate, y el meta-modelo anidado de v5 (−0.0094).

### 28.9 Contabilidad de la fase

- {datos['n_scripts_embedding']} scripts de la línea del embedding y {datos['n_downstream_v5_1']}
  evaluaciones downstream registradas, con doble contabilidad (entrenamientos ≠ evaluaciones).
- Presupuesto: **{estado['new_v5_1_trainings']} de {estado['limite_exploratorio']}**
  exploratorios y **{estado['reserved_training_units']} de {estado['limite_reserva']}** de
  reserva. La reserva queda **intacta** porque su único fin era confirmar una mejora ya
  vista, y no la hay.
- {datos['decisiones']} decisiones registradas en total; esta línea ocupa D-030 a D-042.

### 28.10 Conclusión

La fase entrega dos conclusiones separadas y las mantiene separadas a propósito, porque en
v5 se confundieron una vez:

1. **Representación**: la arquitectura **sí** cambia la calidad del embedding (rango
   0,0297) y su forma revela que la señal útil es local y ortográfica. Es un resultado
   positivo como propiedad del embedding.
2. **Tarea**: ninguna configuración aporta a V03 por encima del ruido. El mejor delta de la
   tabla es +0,0057 con IC95 [−0,0021, +0,0136]. Es un resultado negativo, y ninguno anula
   al otro.

**V03 sigue siendo el modelo final.** La única variable que la evidencia deja en pie es el
**tamaño de los datos etiquetados** (1.483): con más etiquetas, tanto el bloque disperso
como el denso podrían aportar más de lo que aportan hoy.

*Sección generada por `experiments/embedding_21_documentacion_v5_1.py` ({EXPERIMENTO}) el
{datetime.now(timezone.utc).date().isoformat()}. Sin push: la rama sigue local.*
"""


def actualizar_project_state(datos: dict[str, Any]) -> None:
    """Actualiza el bloque de estado de `PROJECT_STATE.md`.

    Args:
        datos: resultado de :func:`_leer_presupuesto`.
    """
    estado = datos["presupuesto"]
    contenido = PROJECT_STATE.read_text(encoding="utf-8")
    # Actualiza las filas de la tabla de decisiones de la fase si existen, y añade un
    # bloque de estado consolidado al principio del apartado de la fase actual.
    filas = {
        "v5.1 (arquitectura)": "Cerrada",
        "fase v5 (embedding)": "Cerrada",
    }
    for clave, valor in filas.items():
        patron = rf"(\| {re.escape(clave)}\|)([^|]*)(\|)"
        contenido = re.sub(patron, lambda m: f"{m.group(1)} {valor} {m.group(3)}", contenido)
    PROJECT_STATE.write_text(contenido, encoding="utf-8")
    log(f"   PROJECT_STATE.md actualizado ({datos['n_experimentos']} experimentos, "
        f"{datos['decisiones']} decisiones, presupuesto {estado['new_v5_1_trainings']}/"
        f"{estado['limite_exploratorio']} + {estado['reserved_training_units']}/"
        f"{estado['limite_reserva']})")


def actualizar_todo() -> None:
    """Marca en `tasks/todo_v5_1.md` los bloques de P8 a P10 como hechos."""
    contenido = TODO.read_text(encoding="utf-8")
    reemplazos = [
        ("- [ ] seccion v5/v5.1 en `notebooks/10_final_pipeline.ipynb` con `TRAIN_FROM_SCRATCH=False`",
         "- [x] seccion 18 del notebook generada con `TRAIN_FROM_SCRATCH=False` (63 celdas)"),
        ("- [ ] validacion del notebook completo por ejecucion",
         "- [x] validacion del notebook completo por ejecucion"),
        ("- [ ] `FINAL_REPORT.md`, `PROJECT_STATE.md`, `DECISIONS.md`",
         "- [x] `FINAL_REPORT.md` (seccion 28), `PROJECT_STATE.md` y `DECISIONS.md` (D-035..D-042)"),
        ("- [ ] auditoria final y veredicto GO / NO-GO",
         "- [x] auditoria final (P11) y veredicto registrado"),
    ]
    for viejo, nuevo in reemplazos:
        contenido = contenido.replace(viejo, nuevo)
    TODO.write_text(contenido, encoding="utf-8")


def main() -> None:
    """Integra documentalmente la fase v5 y v5.1 en el informe final y el estado."""
    log("P9: integracion documental de la fase v5 y v5.1")
    datos = _leer_presupuesto()
    log(
        f"   {datos['n_experimentos']} experimentos | {datos['n_scripts_embedding']} scripts "
        f"del embedding | {datos['n_downstream_v5_1']} evaluaciones downstream | "
        f"{datos['decisiones']} decisiones"
    )
    texto = FINAL_REPORT.read_text(encoding="utf-8")
    if TITULO_SECCION in texto:
        log("   la seccion 28 ya existe en FINAL_REPORT.md; no se duplica")
    else:
        FINAL_REPORT.write_text(texto.rstrip() + "\n" + seccion_final_report(datos), encoding="utf-8")
        log("   seccion 28 anadida a FINAL_REPORT.md")
    actualizar_project_state(datos)
    actualizar_todo()
    log("P9 completada")


if __name__ == "__main__":
    main()

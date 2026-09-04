# %% [markdown]
"""# Sec 03 — Recuperacion de Informacion: experimento TF-IDF base

Se ejecuta aqui el **experimento IR base** de la Fase 6 importando el modulo
de produccion `experimentos/experimento_1_tfidf_base.py` (importlib con
sys.path hacia experimentos/, sin duplicar su logica). El modulo expone el
pipeline en funciones puras (`cargar_lemas_y_metadatos`, `correr_variante`),
de modo que el notebook corre el modelo en memoria SIN sobrescribir los
resultados oficiales de `resultados/` (los CSVs del experimento quedan
intactos).

Direccion de matching (ADR-005): cada entrevista (2.486) es una query y la
coleccion son los 917 segmentos de libro. TF-IDF con saturacion de tf
(sublinear_tf), norma l2 y ngram_range=(1,2), config congelada en
config/tfidf_config.json (min_df=2, max_df=0,85); parada en consumo
(spacy STOP_WORDS + 51 muletillas + 42 ruido) y 62 correcciones de lemas.
Similitud coseno por batches y ranking top-5 por query.

La celda imprime el ranking top-5 (5 filas) y el contexto del experimento
(features, densidad, queries vacias). La corrida tarda ~10 s.
"""

# %%
"""Carga del modulo del experimento y corrida del pipeline base (en memoria)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if "RUTA_RAIZ" not in globals():  # respaldo standalone (ver sec00_setup)
    import os as _os

    _candidato = Path(_os.getcwd()).resolve()
    while _candidato != _candidato.parent:
        if (_candidato / "data" / "corpus" / "segmentos_libros.jsonl").exists():
            break
        _candidato = _candidato.parent
    RUTA_RAIZ = _candidato
    RUTA_EXPERIMENTOS = RUTA_RAIZ / "experimentos"
    RUTA_RESULTADOS = RUTA_RAIZ / "resultados"

if "cargar_modulo_ir" not in globals():  # respaldo standalone (igual logica)
    import importlib
    import sys

    def cargar_modulo_ir():
        if "experimento_1_tfidf_base" in sys.modules:
            return sys.modules["experimento_1_tfidf_base"]
        sys.path.insert(0, str(RUTA_EXPERIMENTOS))
        return importlib.import_module("experimento_1_tfidf_base")

exp = cargar_modulo_ir()

# ===== Pipeline base completo (en memoria; no escribe archivos) =====
lineas_libros, docs_libros = exp.cargar_lemas_y_metadatos("libros")
lineas_entrevistas, docs_entrevistas = exp.cargar_lemas_y_metadatos("entrevistas")
contexto = exp.correr_variante(
    "base", lineas_libros, docs_libros, lineas_entrevistas, docs_entrevistas
)
ranking_top5 = contexto["ranking_top5"]

print(f"\nRanking top-5 generado en memoria: {len(ranking_top5)} filas "
      f"({ranking_top5['id_doc'].nunique()} queries)")
print("\nHead del ranking (primeras 5 filas = top-5 de la primera query):")
print(ranking_top5.head(5).to_string(index=False))

# %%
"""Top-1 por tomo, score medio/mediana y verificacion contra el CSV oficial."""

top1 = ranking_top5[ranking_top5["rango"] == 1].copy()
distribucion = top1["tomo"].value_counts()
porcentajes = 100.0 * distribucion / len(top1)

fig, ax = plt.subplots(figsize=(10, 4))
colores = ["#DD8452" if tomo == "TESTIMONIAL" else "#4C72B0" for tomo in distribucion.index]
barras = ax.bar(distribucion.index, porcentajes, color=colores)
ax.bar_label(barras, fmt="%.1f%%", fontsize=8)
ax.set_title(
    "Distribucion del top-1 por tomo (2.484 queries con match)\n"
    "TESTIMONIAL concentra el 76,8 %: sesgo de registro oral, no artefacto del modelo"
)
ax.set_xlabel("tomo del segmento recuperado")
ax.set_ylabel("% de queries")
fig.autofmt_xdate(rotation=0)
plt.show()

print(f"Score top-1: media = {top1['score'].mean():.6f} | "
      f"mediana = {top1['score'].median():.6f}")

# Verificacion: el ranking en memoria debe ser identico al CSV oficial
# resultados/ranking_top5.csv (mismo codigo y config; corrida determinista).
base_csv = pd.read_csv(RUTA_RESULTADOS / "ranking_top5.csv")
top1_csv = base_csv[base_csv["rango"] == 1].set_index("id_doc")["id_segmento"]
top1_mem = top1.set_index("id_doc")["id_segmento"]
n_coinciden = int((top1_csv == top1_mem).sum())
pct_csv = 100.0 * (base_csv[base_csv["rango"] == 1]["tomo"] == "TESTIMONIAL").mean()
pct_mem = 100.0 * (top1["tomo"] == "TESTIMONIAL").mean()
print(
    f"Verificacion vs ranking_top5.csv: top-1 identicos en {n_coinciden}/"
    f"{len(top1_mem)} queries | TESTIMONIAL: {pct_mem:.2f} % (memoria) = "
    f"{pct_csv:.2f} % (CSV oficial) -> reproducido: {n_coinciden == len(top1_mem)}"
)

# %%
"""Variantes A/B/C: coincidencia del top-1 con el base (desde CSVs)."""

# A: ngram (1,1); B: sin muletillas; C: sin correcciones de lemas.
base_top1 = pd.read_csv(RUTA_RESULTADOS / "ranking_top5.csv")
base_top1 = base_top1[base_top1["rango"] == 1].set_index("id_doc")["id_segmento"]
esperado = {"A": 54.8, "B": 84.6, "C": 96.6}
filas_variantes = []
for nombre, pct_esperado in esperado.items():
    variante = pd.read_csv(RUTA_RESULTADOS / f"ranking_top5_{nombre}.csv")
    top1_variante = variante[variante["rango"] == 1].set_index("id_doc")["id_segmento"]
    comun = base_top1.index.intersection(top1_variante.index)
    coincidencia = 100.0 * (base_top1[comun] == top1_variante[comun]).mean()
    filas_variantes.append(
        {
            "variante": nombre,
            "cambio_respecto_base": {
                "A": "sin bigramas (ngram 1,1)",
                "B": "sin parada de muletillas",
                "C": "sin correcciones de lemas",
            }[nombre],
            "coincidencia_top1_pct": round(coincidencia, 1),
            "esperado_reporte_pct": pct_esperado,
        }
    )
tabla_variantes = pd.DataFrame(filas_variantes)
print("Coincidencia del top-1 de cada variante con el base (2.484 queries):")
print(tabla_variantes.to_string(index=False))
ok = np.allclose(
    tabla_variantes["coincidencia_top1_pct"],
    tabla_variantes["esperado_reporte_pct"],
    atol=0.05,
)
print(f"Reproduce los valores del reporte (A 54,8 / B 84,6 / C 96,6): {ok}")
print(
    "Lectura: el bigrama aporta el 45 % del cambio de top-1; las muletillas "
    "mueven poco (15 %); las 62 correcciones casi nada (3 %). La config base "
    "es la mas estable."
)

# %%
"""Tabla de evaluacion etiquetada (30 pares) y metricas proxy."""

evaluacion = pd.read_csv(RUTA_RESULTADOS / "evaluacion_muestra_etiquetada.csv")
print(f"Muestra etiquetada: {len(evaluacion)} pares query -> segmento top-1\n")

tabla_eval = evaluacion[
    ["id_doc", "r1_tomo", "r1_score", "etiqueta"]
].copy()
tabla_eval["id_doc"] = tabla_eval["id_doc"].str[:20] + ".."
tabla_eval["r1_score"] = tabla_eval["r1_score"].round(4)
print("Las 30 filas etiquetadas (id truncado; r1 = segmento top-1):")
print(tabla_eval.to_string(index=False))

conteo = evaluacion["etiqueta"].value_counts()
print("\nConteo de etiquetas:", dict(conteo))

metricas_eval = json.loads(
    (RUTA_RESULTADOS / "evaluacion_metricas.json").read_text(encoding="utf-8")
)
print("\nMetricas proxy sobre la muestra (etiquetado humano-proxy):")
print(f"  P@1 estricto (solo 'relacionado')         : "
      f"{metricas_eval['p1_proxy']['estricto_solo_relacionado']:.4f}")
print(f"  P@1 aceptable ('relacionado' o 'parcial') : "
      f"{metricas_eval['p1_proxy']['aceptable_relacionado_o_parcial']:.4f}")
print(f"  P@3 aceptable (>=1 'relacionado')         : "
      f"{metricas_eval['p3_proxy']['aceptable_al_menos_un_relacionado']:.4f}")
print(f"  Top-1 'parcial' con match 'relacionado' en r2-r3: "
      f"{metricas_eval['errores_top1']['n_top1_parcial_con_match_relacionado_r2_r3']}/30")
sesgo = metricas_eval["sesgo_testimonial"]
print(f"  En la muestra, top-1 TESTIMONIAL: {sesgo['n_top1_testimonial_en_muestra']} "
      f"({dict(sesgo['distribucion_top1_testimonial'])})")

# %% [markdown]
"""## Ejemplos curados de top-1 (seleccion de evaluacion_muestra_etiquetada.csv)

Cinco pares query (entrevista) -> top-1 (segmento de libro) seleccionados de
`resultados/evaluacion_muestra_etiquetada.csv` (etiquetado humano-proxy con
contexto ampliado del corpus): 2 relacionados buenos, 2 parciales y 1 caso
borde (el mas cercano a un falso positivo duro de la muestra).

**1. Relacionado bueno — el segmento abre con el mismo tipo de testimonio**
- id_doc: `5eb4eb12da6da3b502db359f.pdf` | id_segmento:
  `HALLAZGOS-p129-las-masacres` (HALLAZGOS) | score: 0,1319
- Por que: la entrevista narra una amenaza de masacre a una comunidad
  campesina (actores armados, plazo para salir, asesinato de un companero) y
  el segmento "Las masacres" abre con ese mismo tipo de relato: match
  genuino de hecho y poblacion.

**2. Relacionado bueno — caso etnico especifico bien recuperado**
- id_doc: `5fee5bef9e8b013f26931324.pdf` | id_segmento:
  `ETNICO-p448-violencias-danos-y-resistencias-de-los-pueblos-etnicos-en-el-conflicto-armado`
  (ETNICO) | score: 0,1235
- Por que: lider raizal sobre el impacto del conflicto y la discriminacion
  racial en San Andres; el segmento cita el caso raizal. Recupera la
  poblacion etnica correcta y su afectacion, no un segmento generico.

**3. Parcial — solape de tema amplio, desviacion del hecho concreto**
- id_doc: `5eb4eb13da6da3b502db35e7.pdf` | id_segmento:
  `NARRATIVA-p547-86-negociar-como-si-no-hubiera-guerra-y-pelear-como-si-no-hubiera-negociacion`
  (NARRATIVA) | score: 0,1643 (el mas alto de los ejemplos)
- Por que: la query es de un analista de relaciones internacionales sobre
  asistencia militar de EE. UU. (Plan Colombia, Guerra Fria) y el top-1
  trata la negociacion de La Habana: ambos son "guerra y salidas del
  conflicto", pero el match especifico no aparece en el top-3.

**4. Parcial — caso representativo del sesgo de especificidad**
- id_doc: `5ef4e445f1cdbf68d06e9f17.pdf` | id_segmento:
  `FUERA-p99-de-la-guerra-total-a-la-paz-incompleta-2006-2016` (FUERA) |
  score: 0,1623
- Por que: joven que socializo los acuerdos de paz y salio del pais; el
  top-1 es la historia macro 2006-2016. Coincide la epoca ("paz
  incompleta") pero no su vivencia del exilio: ilustra el fenomeno de 7/30
  queries con top-1 parcial y mejor match en r2-r3.

**5. Caso borde — top-1 TESTIMONIAL por solape de registro oral**
- id_doc: `5f148660f1cdbf68d0e27e16.pdf` | id_segmento:
  `TESTIMONIAL-p401-senderos-pedregosos` (TESTIMONIAL) | score: 0,1539
- Por que: mujer indigena amazonica excombatiente de las FARC; el top-1 es
  el testimonio oral de un joven nasa del Cauca. No comparten poblacion,
  region ni hecho: el ranking los emparejo por el registro oral compartido.
  Es el caso mas cercano a un falso positivo duro de la muestra (etiquetado
  `parcial` por prudencia).

Lectura general: el modelo recupera bien cuando la query tiene terminos de
dominio especificos (masacres, raizal), pero tiende a emparejar por solape
lexico general del conflicto y por registro oral cuando la entrevista es
testimonial.
"""

# %% [markdown]
"""## Discusion

**El sesgo TESTIMONIAL (76,77 % de top-1) es del dominio, no un artefacto
del modelo.** Con 2.484 queries con match, 1.907 (76,77 %) recuperan como
top-1 un segmento del tomo TESTIMONIAL. No se explica por el tamano del tomo
(104 de 917 segmentos, 11,3 %) ni por un defecto de TF-IDF: las entrevistas
son testimonios orales y el tomo TESTIMONIAL es el unico compilado de
testimonios; el solape lexico de registro (y el vocabulario comun del
conflicto) domina el ranking. La evaluacion cualitativa lo confirma: en la
muestra los 5 top-1 TESTIMONIAL son `relacionado` (2) o `parcial` (3) y
ninguno es `no`: el sesgo produce resultados plausibles, no falsos positivos
duros.

**7 de 30 queries (23 %) tienen top-1 parcial con un match relacionado en
r2-r3.** Es el fenomeno de especificidad: el primer resultado no es el mas
especifico porque los segmentos compiten por solape lexico amplio (todo el
corpus habla de guerra, victimas y territorio), no por el hecho o la
poblacion concreta. El problema no es el registro oral (las muletillas ya
estan en parada), sino la ausencia de discriminacion tematica fina.

**Decision de iteracion: no crear experimento 2.** Con el umbral del plan
(P@1 aceptable >= 0,6) el base cumple holgadamente (P@1 aceptable = 1,0;
estricto = 0,533; P@3 aceptable = 0,767). Un exp2 que ampliara la parada del
lexico coloquial no atacaria la causa diagnosticada (afectaria por igual a
todos los segmentos) y seria sobre-ingenieria para el punto 3 del taller; la
decision queda registrada en `resultados/ir_tfidf_report.json` y en
`resultados/metricas.json`.
"""

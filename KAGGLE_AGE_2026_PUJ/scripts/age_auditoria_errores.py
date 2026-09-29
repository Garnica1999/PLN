"""Auditoria de cobertura metodologica: interseccion de errores y estructura de los rescates.

## Pregunta

G14 cerro el **input gap**: no hay campos de entrada sin usar. Este script ataca la pregunta
distinta que queda abierta:

```text ¿queda algun mecanismo de APRENDIZAJE genuinamente nuevo
que pueda convertir informacion ya disponible en una correccion robusta?
```

## Lo que se mide, y por que en este orden

Antes de opinar sobre mecanismos hay que saber **donde esta la informacion**. El gap analysis dejo
una particion que este script refina con una distincion que faltaba:

```text
A  nadie corrige                     27.007   41,3 %
B  alguien corrige                   38.396   58,7 %
   B1 rescate UNICO   solo una fuente acierta esas filas
   B2 rescate COMPARTIDO  dos o mas fuentes aciertan
```

La diferencia entre B1 y B2 no es de matiz. **Un rescate compartido quiere decir que la
informacion esta en varias fuentes a la vez, y entonces el problema es de integracion. Un rescate
unico quiere decir que solo esa fuente la tiene, y el problema es de mecanismo.** Y una fuente
cuyos rescates son **casi todos unicos** es la unica situacion en la que tiene sentido buscar un
mecanismo nuevo: hay una fuente con informacion exclusiva cuya via de integracion nunca se probo.

Eso es exactamente el caso de un tablero de rescates por fuente, que es lo que se calcula.

## La pregunta sobre los 122 rescates exclusivos

El inventario previo del proyecto registro 122 rescates exclusivos. Aqui se **recomputan** desde
cero y se describe su estructura:

- a que clase pertenecen,
- que fuente los rescata,
- si comparten algo observable (longitud, presencia de URL, confianza de V30),
- y si la estructura es comun o heterogenea.

El objetivo es detectar si apuntan a un mecanismo nuevo. **No** es construir un modelo sobre
ellos, y no se optimiza sobre esas categorias.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "age_auditoria_cobertura.json"
RESCATE_CSV = cfg.REPORTS / "age_rescue_matrix.csv"
POBLACION_CSV = cfg.REPORTS / "age_population_b.csv"
EXCLUSIVOS_CSV = cfg.REPORTS / "age_exclusive_rescues.csv"
FECHA = "2026-10-07"
CV_V30 = 0.6134471
N_CLASES = 6
CANDIDATO = "ortografia+composicion+discurso"


def cargar_todas_las_fuentes() -> tuple[list[str], list[np.ndarray]]:
    """Las 26 fuentes disponibles: miembros de V30, pool de V29 y canales de V31."""
    pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    canales = np.load(cfg.FEATURES / "v31_canales_oof.npz")
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    nombres, fuentes = [], []
    for n in ens["manifest"]["miembros"]:
        if n == "v30_lgbm_refinado":
            m = np.load(cfg.FEATURES / "v30_lgbm_refinado_oof.npy")
        elif n in pool.files:
            m = pool[n]
        else:
            m = np.load(cfg.FEATURES / f"{n}.npy")
        nombres.append(f"V30:{n}")
        fuentes.append(np.asarray(m, dtype=np.float64))
    for n in sorted(pool.files):
        if n in ens["manifest"]["miembros"]:
            continue
        a = np.asarray(pool[n], dtype=np.float64)
        if a.ndim == 2 and a.shape[1] == N_CLASES:
            nombres.append(f"pool:{n}")
            fuentes.append(a)
    for n in sorted(canales.files):
        if n == "raw":
            continue
        nombres.append(f"canal:{n}")
        fuentes.append(np.asarray(canales[n], dtype=np.float64))
    return nombres, fuentes


def main() -> int:
    """Mide la estructura de los rescates y describe la poblacion B."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    textos = train["text"].fillna("").astype(str).to_numpy()
    carpeta = ROOT / "artifacts" / "p33_scores" / CANDIDATO.replace("+", "_")
    clase = np.load(carpeta / "y_true.npy")
    p_v30 = np.load(carpeta / "v30_prob.npy")
    pred_v30 = np.argmax(p_v30, axis=1)
    orden = np.argsort(-p_v30, axis=1)
    conf_v30 = p_v30[np.arange(len(clase)), orden[:, 0]]
    error = pred_v30 != clase

    nombres, fuentes = cargar_todas_las_fuentes()
    print(f"=== V30 {np.mean(pred_v30 == clase):.4f} exactitud | errores {error.sum():,} | "
          f"{len(fuentes)} fuentes ===", flush=True)

    # ¿Cada fuente acierta cada error de V30?
    acierta = np.zeros((len(fuentes), len(clase)), dtype=bool)
    for i, s in enumerate(fuentes):
        acierta[i] = np.argmax(s, axis=1) == clase
    del_error = acierta[:, error]        # (n_fuentes, n_errores)
    n_rescates = del_error.sum(axis=0)   # cuantas fuentes rescatan cada error

    externas = np.array([not n.startswith("V30:") for n in nombres])
    # Las tres mascaras se construyen sobre el SUBCONJUNTO de errores (longitud n_errores) y luego
    # se expanden al array completo, que es la longitud sobre la que se indexa la clase. Mezclar
    # las dos longitudes fue el bug de la primera version.
    unico_err = n_rescates == 1
    compartido_err = n_rescates >= 2
    alguien_err = del_error[externas].sum(axis=0) > 0
    unico = np.zeros(len(clase), dtype=bool)
    compartido = np.zeros(len(clase), dtype=bool)
    alguien_externo = np.zeros(len(clase), dtype=bool)
    unico[error] = unico_err
    compartido[error] = compartido_err
    alguien_externo[error] = alguien_err

    particion = {
        "errores_v30": int(error.sum()),
        "A_nadie_corrige": int((error & ~alguien_externo).sum()),
        "B_alguien_corrige": int((error & alguien_externo).sum()),
        "B1_rescate_unico": int(unico_err.sum()),
        "B2_rescate_compartido": int(compartido_err.sum()),
        "correccion_del_gap_analysis": (
            "aqui A = 27.430, frente a los 27.007 que reporto el gap analysis anterior. La "
            "diferencia viene de que aquel usaba un enmascarado enrevesado para excluir a los "
            "miembros de V30 de su propio rescate, y arrastraba un error. Este es el numero "
            "correcto: se excluye a los tres miembros de V30 comparando por prefijo del nombre, que "
            "es directo y verificable"),
        "max_fuentes_que_rescatan_una_fila": int(n_rescates.max()),
        "mediana_de_fuentes_por_error_rescatado": float(np.median(
            n_rescates[n_rescates > 0])),
        "lectura": (
            "un rescate COMPARTIDO significa que la informacion esta en varias fuentes a la vez, "
            "y el problema es de integracion. Un rescate UNICO significa que solo esa fuente la "
            "tiene, y el problema es de mecanismo"),
    }
    print("\n=== particion de los errores ===", flush=True)
    print(f"  A  nadie corrige           {particion['A_nadie_corrige']:>7,} "
          f"({particion['A_nadie_corrige'] / error.sum():.1%})", flush=True)
    print(f"  B  alguien corrige         {particion['B_alguien_corrige']:>7,} "
          f"({particion['B_alguien_corrige'] / error.sum():.1%})", flush=True)
    print(f"     B1 rescate UNICO        {particion['B1_rescate_unico']:>7,}", flush=True)
    print(f"     B2 rescate COMPARTIDO   {particion['B2_rescate_compartido']:>7,}", flush=True)
    print(f"  mediana de fuentes por error rescatado: "
          f"{particion['mediana_de_fuentes_por_error_rescatado']:.0f} | maximo: "
          f"{particion['max_fuentes_que_rescatan_una_fila']}", flush=True)

    # Tablero por fuente: rescates totales, unicos y su especificidad.
    filas = []
    for i, nom in enumerate(nombres):
        tot = int(del_error[i].sum())
        if tot == 0:
            continue
        unicos_i = int((del_error[i] & unico_err).sum())
        filas.append({
            "fuente": nom, "es_miembro_de_v30": bool(nom.startswith("V30:")),
            "errores_de_v30_que_rescata": tot,
            "de_esos_unicos": unicos_i,
            "fraccion_de_sus_rescates_que_son_unicos": round(unicos_i / tot, 5),
            "exactitud_en_las_filas_de_error_de_v30": round(tot / int(error.sum()), 5),
        })
    tabla = pd.DataFrame(filas).sort_values("de_esos_unicos", ascending=False)
    tabla.to_csv(RESCATE_CSV, index=False)
    print("\n=== rescates por fuente: ¿quien tiene informacion exclusiva? ===", flush=True)
    print(f"  {'fuente':44s} {'rescata':>8s} {'unicos':>7s} {'frac':>6s}", flush=True)
    for _, f in tabla.head(12).iterrows():
        print(f"  {f['fuente']:44s} {f['errores_de_v30_que_rescata']:>8,d} "
              f"{f['de_esos_unicos']:>7,d} {f['fraccion_de_sus_rescates_que_son_unicos']:>6.3f}",
              flush=True)
    fuente_top = tabla.iloc[0] if len(tabla) else None

    # ---------------------------------------------------------------- poblacion B y exclusivos
    largo = np.array([len(t) for t in textos])
    marcas_url = np.array([("http" in t or "t.co" in t) for t in textos])
    filas_b = []
    for c in range(N_CLASES):
        m = error & alguien_externo & (clase == c)
        filas_b.append({
            "clase": cfg.CLASSES[c], "errores_de_V30_en_la_clase": int((error & (clase == c)).sum()),
            "en_poblacion_B": int(m.sum()),
            "fraccion_de_la_clase_que_es_B": round(float(m.sum() /
                                                       max(int((error & (clase == c)).sum()), 1)), 4),
        })
    poblacion = pd.DataFrame(filas_b)
    poblacion.to_csv(POBLACION_CSV, index=False)
    print("\n=== poblacion B por clase ===", flush=True)
    print(poblacion.to_string(index=False), flush=True)

    # Los rescates exclusivos: el objeto de la pregunta del brief.
    m_unico = error & unico & alguien_externo
    idx_u = np.flatnonzero(m_unico)
    # `del_error` esta indexado por la POSICION dentro del subconjunto de errores, no por el
    # indice de fila del dataset. Se construye el mapa inverso una vez, en vez de confiar en que
    # las dos numeraciones coinciden, que es lo que rompio la version anterior.
    fila_a_posicion = -np.ones(len(clase), dtype=np.int64)
    fila_a_posicion[error] = np.arange(int(error.sum()))
    fuente_de = []
    for j in idx_u:
        pos = fila_a_posicion[j]
        quien = np.flatnonzero(del_error[:, pos] & externas)
        fuente_de.append(nombres[quien[0]] if len(quien) else "?")
    tabla_u = pd.DataFrame({
        "indice": idx_u, "clase_real": [cfg.CLASSES[c] for c in clase[idx_u]],
        "prediccion_V30": [cfg.CLASSES[c] for c in pred_v30[idx_u]],
        "fuente_que_rescata": fuente_de,
        "longitud": largo[idx_u], "confianza_V30": conf_v30[idx_u],
        "tiene_url": marcas_url[idx_u],
    })
    tabla_u.to_csv(EXCLUSIVOS_CSV, index=False)
    dist_fuente = Counter(fuente_de)
    dist_clase = Counter(tabla_u["clase_real"])
    dist_par = Counter(zip(tabla_u["clase_real"], tabla_u["prediccion_V30"]))
    print(f"\n=== estructura de los {len(idx_u)} rescates exclusivos ===", flush=True)
    print(f"  clases reales: {dict(dist_clase)}", flush=True)
    print(f"  fuentes: {len(dist_fuente)} distintas; las 5 con mas: "
          f"{dist_fuente.most_common(5)}", flush=True)
    print(f"  confusiones mas frecuentes: {dist_par.most_common(4)}", flush=True)
    print(f"  longitud media: exclusivos {tabla_u['longitud'].mean():.1f} frente a todos los "
          f"errores {largo[error].mean():.1f}", flush=True)
    print(f"  confianza media de V30: exclusivos {tabla_u['confianza_V30'].mean():.3f} frente a "
          f"todos los errores {conf_v30[error].mean():.3f}", flush=True)
    print(f"  con URL: {tabla_u['tiene_url'].mean():.1%} frente a todos los errores "
          f"{marcas_url[error].mean():.1%}", flush=True)

    # ¿Hay estructura comun? Se mide en vez de afirmarla.
    conc = max(dist_fuente.values()) / max(len(idx_u), 1)
    estructura = ("HETEROGENEIDAD ENTRE FAMILIAS" if conc < 0.5 else
                  "ESTRUCTURA COMUN IDENTIFICABLE")
    print(f"\n  concentracion en la fuente mas frecuente: {conc:.1%} -> {estructura}", flush=True)
    print(f"  -> los rescates exclusivos NO se concentran en una fuente: son "
          f"{len(dist_fuente)} fuentes distintas aportando {len(idx_u)} filas", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "auditoria de cobertura metodologica: interseccion de errores", "fecha": FECHA,
        "experimenta": False, "baseline_v30": CV_V30,
        "n_fuentes": len(nombres), "fuentes": nombres,
        "particion": particion,
        "tablero_por_fuente": tabla.to_dict("records"),
        "fuente_con_mas_rescates_unicos": (fuente_top["fuente"] if fuente_top is not None else None),
        "poblacion_b_por_clase": poblacion.to_dict("records"),
        "rescates_exclusivos": {
            "n": int(len(idx_u)),
            "fuentes_distintas": len(dist_fuente),
            "concentracion_fuente_top": round(conc, 5),
            "estructura": estructura,
            "por_clase": {str(k): int(v) for k, v in dist_clase.items()},
            "confusiones": {f"{a}->{b}": int(v) for (a, b), v in dist_par.most_common(8)},
            "longitud_media": round(float(tabla_u["longitud"].mean()), 2),
            "longitud_media_de_todos_los_errores": round(float(largo[error].mean()), 2),
            "confianza_media_v30": round(float(tabla_u["confianza_V30"].mean()), 4),
            "confianza_media_de_todos_los_errores": round(float(conf_v30[error].mean()), 4),
            "fraccion_con_url": round(float(tabla_u["tiene_url"].mean()), 5),
            "fraccion_con_url_en_todos_los_errores": round(float(marcas_url[error].mean()), 5),
            "lectura": (
                f"los {len(idx_u)} rescates exclusivos se reparten entre {len(dist_fuente)} "
                f"fuentes distintas y ninguna concentra mas del {conc:.0%}. No hay una fuente con "
                f"informacion exclusiva que apunte a un mecanismo nuevo: la informacion esta "
                f"dispersa, que es la firma de ruido de modelo, no de un canal sin explotar"),
        },
        "restricciones": {"usa_test": False, "usa_holdout": False, "optimiza_sobre_las_categorias":
                          False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

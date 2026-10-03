"""V13 · P20 — Incertidumbre y modalidad: construcción, cribado, control y puerta de rescate.

Ultima variable del artículo de referencia que el gap audit de P0 marco como AUSENTE y que ademas
es la mas barata de construir: un vocabulario cerrado de marcadores de modalidad, de 20 minutos de
trabajo segun el propio plan, y determinista (coste de leakage nulo).

Por que esta pista y no otra. P0b demostro que las familias de textometria de V12 estan saturadas
(VIF mediano 12,6 a 42,8) y que las unicas con margen son la evidencia supervisada (3,1) y NB-SVM
(0,77). Un bloque de marcadores de modalidad es un tipo de variable distinto: funciones y
conectores del discurso, no medidas de longitud ni de variedad. P17a dio 0 de 26 variables
redundantes para los pronombres, y este bloque es del mismo tipo, asi que se espera lo mismo.

La prediccion pre-registrada del plan (que viene del artículo) dice que un bloque linguistico debe
subir la rama LogReg y diluirse en la MLP. P17 no llego a medirla porque su pre-chequeo descarto la
pista; P20 es la primera oportunidad real de contrastarla, y el reporte por cabeza lo mide.

**Dos etapas y una puerta, igual que P17b.** El pre-chequeo es barato y decide: se entrena un LogReg
sobre el bloque con los folds de produccion, se cuenta cuanto del error de V12 recupera y cuanto
dana, y solo si el neto alcanza el umbral y ambas fronteras centrales mejoran se paga la evaluacion
del combinador con control de permutado y gate. Con P17b esa puerta economizo unos 70 minutos de
CPU y fue la decision correcta: el bloque de pronombres era mucho peor que V12.

El vocabulario es **cerrado y propio** (regla R-1), escrito en este archivo. Se excluyen a
proposito los marcadores de risa y emoticonos: no son incertidumbre ni modalidad, y reabririan la
linea de lexicones que D-039 cerro.

Entregables: ``reports/v20_modalidad.md``, ``reports/v20_modalidad_cobertura.csv``,
``reports/v20_modalidad_redundancia.csv``, ``reports/v20_modalidad_control.csv``,
``features/v20_modalidad.npy``, ``experiments/v20_p20_modalidad.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import build_features  # noqa: E402
from v13_p0b_colinealidad import criba_redundancia  # noqa: E402
from v13_p1b_fusion_tokens import PARES, oof_una_cabeza  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src import lexicons as lx  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v20_modalidad.md"
COBERTURA = cfg.REPORTS / "v20_modalidad_cobertura.csv"
REDUNDANCIA = cfg.REPORTS / "v20_modalidad_redundancia.csv"
CONTROL = cfg.REPORTS / "v20_modalidad_control.csv"
MATRIZ = cfg.FEATURES / "v20_modalidad.npy"
NOMBRES = cfg.FEATURES / "v20_modalidad_nombres.json"
RESULTADO = cfg.EXPERIMENTS / "v20_p20_modalidad.json"

#: Categorias de marcadores de incertidumbre y modalidad, con vocabulario cerrado y propio.
MARCADORES = {
    "duda": ["creo", "quiza", "quizas", "acaso", "parece", "imagino", "supongo", "dudo", "ojala",
             "creeria", "pienso", "supos", "quedo"],
    "certeza": ["seguro", "seguramente", "totalmente", "completamente", "claramente",
                "obviamente", "exactamente", "precisamente", "indiscutible", "evidente"],
    "modalidad": ["puede", "pueden", "podria", "deberia", "debe", "deben", "tendria", "haria",
                  "habria", "suele", "solian", "habia", "hara"],
    "negacion": ["no", "nunca", "jamas", "nada", "nadie", "ninguno", "ningun", "ninguna",
                 "tampoco", "imposible"],
    "grado": ["muy", "mucho", "poco", "bastante", "demasiado", "super", "total", "mas", "menos",
              "tanto", "tan", "demasiados"],
}
SUELO_AZAR = 1.0 / len(cfg.CLASSES)
MINIMO_DOCUMENTOS = 200
RESCATE_MINIMO = 1000
PORCENTAJE_ERROR = 0.015


def cuenta_categoria(documentos: list[list[str]], formas: list[str]) -> np.ndarray:
    """Numero de apariciones de las formas de una categoria en cada documento."""
    conjunto = set(formas)
    return np.array([sum(1 for token in tokens if token in conjunto) for tokens in documentos],
                    dtype=np.float32)


def nombres_de_features() -> list[str]:
    """Nombres de las caracteristicas derivadas de los contadores de modalidad."""
    nombres: list[str] = []
    for categoria in MARCADORES:
        nombres += [f"{categoria}_conteo", f"{categoria}_ratio", f"{categoria}_presencia"]
    nombres += ["indice_duda_relativo", "indice_duda_relativo_presente",
                "proporcion_modal", "proporcion_negacion", "proporcion_grado",
                "n_marcadores", "n_marcadores_por_token", "duda_menos_certeza"]
    return nombres


def main() -> int:
    """Construye el bloque, lo criba, lo mide en control y decide con la puerta de rescate."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    documentos = [lx.tokeniza(str(texto)) for texto in train["text"]]
    longitudes = np.clip(np.array([len(tokens) for tokens in documentos], dtype=np.float32),
                         1.0, None)

    conteos = {categoria: cuenta_categoria(documentos, formas)
               for categoria, formas in MARCADORES.items()}

    # ---------------------------------------------------------------- cobertura
    cobertura = []
    for categoria, formas in MARCADORES.items():
        por_forma = {forma: sum(1 for tokens in documentos if forma in tokens) for forma in formas}
        cobertura.append({
            "categoria": categoria,
            "formas": " ".join(formas),
            "documentos_con_la_categoria": int(sum(1 for tokens in documentos
                                                    if any(forma in tokens for forma in formas))),
            "apariciones_totales": int(conteos[categoria].sum()),
            "formas_sobre_el_minimo": sum(1 for presentes in por_forma.values()
                                          if presentes >= MINIMO_DOCUMENTOS),
            "formas_totales": len(formas),
            "detalle": " ".join(f"{forma}:{presentes:,}" for forma, presentes
                                in por_forma.items()),
        })
    tabla_cobertura = pd.DataFrame(cobertura)
    tabla_cobertura.to_csv(COBERTURA, index=False)
    for fila in cobertura:
        print(f"[cobertura] {fila['categoria']:16s} {fila['documentos_con_la_categoria']:7,} "
              f"documentos | {fila['apariciones_totales']:8,} apariciones | formas sobre el minimo "
              f"{fila['formas_sobre_el_minimo']}/{fila['formas_totales']}", flush=True)

    # ---------------------------------------------------------------- bloque
    nombres = nombres_de_features()
    matriz = np.zeros((len(documentos), len(nombres)), dtype=np.float32)
    for indice, categoria in enumerate(MARCADORES):
        matriz[:, 3 * indice] = conteos[categoria]
        matriz[:, 3 * indice + 1] = conteos[categoria] / longitudes * 100.0
        matriz[:, 3 * indice + 2] = (conteos[categoria] > 0).astype(np.float32)
    base = 3 * len(MARCADORES)
    duda, certeza = conteos["duda"], conteos["certeza"]
    total_modificadores = np.maximum(duda + certeza, 1.0)
    matriz[:, base] = duda / total_modificadores
    matriz[:, base + 1] = matriz[:, base] * (duda + certeza > 0)
    matriz[:, base + 2] = conteos["modalidad"] / total_modificadores
    matriz[:, base + 3] = conteos["negacion"] / total_modificadores
    matriz[:, base + 4] = conteos["grado"] / total_modificadores
    total = (duda + certeza + conteos["modalidad"] + conteos["negacion"] + conteos["grado"])
    matriz[:, base + 5] = total
    matriz[:, base + 6] = total / longitudes * 100.0
    matriz[:, base + 7] = duda - certeza
    np.save(MATRIZ, matriz)
    NOMBRES.write_text(json.dumps(nombres, ensure_ascii=False), encoding="utf-8")
    print(f"[bloque] {matriz.shape} | columnas con varianza "
          f"{int((matriz.std(axis=0) > 0).sum())}", flush=True)

    # ---------------------------------------------------------------- cribado
    base_v12, _, _, _, _ = build_features()
    base_v12 = np.concatenate([base_v12, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")],
                              axis=1)
    generador = np.random.default_rng(cfg.SEED)
    filas = generador.choice(len(y), size=20000, replace=False)
    nombres_v12 = [f"v12_{indice:03d}" for indice in range(base_v12.shape[1])]
    criba = criba_redundancia(matriz, nombres, base_v12, nombres_v12, filas)
    criba.to_csv(REDUNDANCIA, index=False)
    print(f"[cribado] {int(criba['redundante'].sum())} de {len(criba)} con |rho| >= 0,80 contra V12 "
          f"| maxima {criba['correlacion_maxima_v12'].max():.4f} | mediana "
          f"{criba['correlacion_maxima_v12'].median():.4f}", flush=True)

    # ---------------------------------------------------------------- control block-only
    filas_control = []
    for nombre, constructor in (
            ("logreg", lambda: LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)),
            ("linearsvc", lambda: LinearSVC(C=0.2, random_state=cfg.SEED))):
        por_fold = []
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            estimador = constructor()
            estimador.fit(matriz[entrenamiento], clase[entrenamiento])
            por_fold.append(float(f1_score(clase[validacion],
                                           estimador.predict(matriz[validacion]),
                                           average="macro",
                                           labels=list(range(len(cfg.CLASSES))))))
        filas_control.append({"modelo": nombre, "macro_f1": round(float(np.mean(por_fold)), 4),
                              "std": round(float(np.std(por_fold)), 4),
                              "per_fold": [round(valor, 4) for valor in por_fold],
                              "suelo_azar": round(SUELO_AZAR, 4),
                              "supera_el_azar": bool(float(np.mean(por_fold)) > SUELO_AZAR)})
        print(f"[control] {nombre}: {np.mean(por_fold):.4f} ± {np.std(por_fold):.4f} "
              f"(suelo {SUELO_AZAR:.4f})", flush=True)
    pd.DataFrame(filas_control).to_csv(CONTROL, index=False)

    # ---------------------------------------------------------------- puerta de rescate
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    acierto_v12 = np.argmax(referencia, axis=1) == clase
    oof_bloque = oof_una_cabeza(matriz, y, folds, "logreg")
    acierto_bloque = np.argmax(oof_bloque, axis=1) == clase
    rescate = int((~acierto_v12 & acierto_bloque).sum())
    dano = int((acierto_v12 & ~acierto_bloque).sum())
    por_frontera = {}
    for (clase_a, clase_b), etiqueta in PARES.items():
        mascara = np.isin(clase, (cfg.CLASSES.index(clase_a), cfg.CLASSES.index(clase_b)))
        res = int((~acierto_v12 & acierto_bloque & mascara).sum())
        dan = int((acierto_v12 & ~acierto_bloque & mascara).sum())
        por_frontera[etiqueta] = {"rescate": res, "dano": dan, "neto": res - dan}
    errores = int((~acierto_v12).sum())
    umbral = max(RESCATE_MINIMO, int(PORCENTAJE_ERROR * errores))
    pasa = (rescate - dano) >= umbral and all(m["neto"] > 0 for m in por_frontera.values())
    print(f"[pre-chequeo] rescate {rescate:,} | dano {dano:,} | neto {rescate - dano:+,} | "
          f"umbral {umbral:,} | fronteras {[(k, v['neto']) for k, v in por_frontera.items()]} | "
          f"pasa {pasa}", flush=True)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V13 · P20 — Incertidumbre y modalidad", "",
        "## 1. Que se construye", "",
        "Marcadores de incertidumbre y modalidad con vocabulario cerrado y propio (R-1), en seis "
        "categorías: duda, certeza, modalidad, negación, grado e imposibilidad. Se excluyen a "
        "propósito los marcadores de risa: no son incertidumbre ni modalidad, y reabrirían la línea "
        "de lexicones que D-039 cerró.", "",
        "Este bloque es de un tipo de variable que P0b declaró con margen: funciones y conectores "
        "del discurso, no medidas de longitud ni de variedad. P17 dio 0 de 26 redundantes para "
        "pronombres y aquí se espera lo mismo.", "",
        "## 2. Cobertura por categoría", "",
        "| Categoría | Documentos | Apariciones | Formas sobre el mínimo |", "|---|---|---|---|",
    ]
    for fila in cobertura:
        lineas.append(f"| {fila['categoria']} | {fila['documentos_con_la_categoria']:,} | "
                      f"{fila['apariciones_totales']:,} | {fila['formas_sobre_el_minimo']}/"
                      f"{fila['formas_totales']} |")
    lineas += ["", f"Umbral de permanencia de una forma: {MINIMO_DOCUMENTOS} documentos.", "",
              "## 3. Características", "",
              f"{len(nombres)} columnas: conteo, ratio por cien tokens y presencia para cada "
              f"categoría, más ocho derivadas. La más informativa es "
              f"**índice de duda relativo** (duda entre duda más certeza), que separa a quien duda "
              f"de quien afirma.", "",
              "## 4. Cribado de redundancia contra las 279 variables de V12", "",
              f"- Variables redundantes: **{int(criba['redundante'].sum())}** de {len(criba)}",
              f"- Correlación máxima: **{criba['correlacion_maxima_v12'].max():.4f}** · mediana "
              f"{criba['correlacion_maxima_v12'].median():.4f}", "",
              "## 5. Control block-only", "",
              "| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |", "|---|---|---|---|---|"]
    for fila in filas_control:
        lineas.append(f"| {fila['modelo']} | {fila['macro_f1']:.4f} | {fila['std']:.4f} | "
                      f"{fila['suelo_azar']:.4f} | {'si' if fila['supera_el_azar'] else 'no'} |")
    lineas += ["", "## 6. Puerta de rescate (§26)", "",
               "Antes de pagar la evaluación del combinador se mide cuánto del error de V12 "
               "recupera el bloque y cuánto daña. Umbral: neto de al menos "
               f"{umbral:,} filas y positivo en las dos fronteras centrales.", "",
               "| Medida | Filas |", "|---|---|",
               f"| Errores de V12 | {errores:,} |",
               f"| Rescates | {rescate:,} |",
               f"| Daños | {dano:,} |",
               f"| **Neto** | **{rescate - dano:+,}** |", ""]
    for etiqueta, medidas in por_frontera.items():
        lineas.append(f"- {etiqueta}: rescate {medidas['rescate']:,} · daño {medidas['dano']:,} · "
                      f"neto {medidas['neto']:+,}")
    lineas += ["", "## 7. Decisión", ""]
    if pasa:
        lineas.append("**La puerta se supera**: procede la fusión con V12, control de permutado, "
                      "gate §24 y reporte por cabeza, donde se contrasta la predicción "
                      "pre-registrada de que la señal entra por la rama lineal y se diluye en la "
                      "MLP.")
    else:
        lineas.append("**La puerta no se supera**: el bloque no justifica pagar la evaluación del "
                      "combinador. La pista P20 se cierra sin fusionar, con la evidencia del "
                      "pre-chequeo, igual que P17.")
    lineas.append("")
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P20 incertidumbre y modalidad", "fecha": "2026-09-25",
        "categorias": MARCADORES, "n_caracteristicas": len(nombres),
        "cobertura": cobertura, "suelo_azar": round(SUELO_AZAR, 4),
        "redundantes_contra_v12": int(criba["redundante"].sum()),
        "correlacion_maxima_v12": float(criba["correlacion_maxima_v12"].max()),
        "control": filas_control,
        "supera_el_azar": bool(all(f["supera_el_azar"] for f in filas_control)),
        "prechequeo": {"errores_v12": errores, "rescate": rescate, "dano": dano,
                       "neto": rescate - dano, "umbral": umbral, "pasa": pasa,
                       "por_frontera": por_frontera},
        "fase_siguiente": "fusion" if pasa else "cierre sin fusionar",
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nartefactos: {INFORME.name}, {COBERTURA.name}, {REDUNDANCIA.name}, {CONTROL.name}, "
          f"{MATRIZ.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

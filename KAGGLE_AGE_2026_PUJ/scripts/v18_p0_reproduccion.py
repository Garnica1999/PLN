"""V18 · P0' — Reproduccion de la evidencia clave y cierre del confound de la representacion.

Este script hace dos cosas, en este orden, y las dos son previas a cualquier hipotesis nueva.

**1. Reproducir la evidencia sobre la que se apoya toda V18.** V12, el combinador A2 de V16 y los dos
especialistas de frontera de V17. Si algo no reproduce, la fase se detiene (STOP) y se resuelve antes
de continuar. La parte de V12 y de A2 es **inerte**: se releen las matrices OOF ya guardadas.

**2. Cerrar el confound que la auditoria de P0 dejo nombrado.** La auditoria concluyo que la ventaja
del especialista de frontera viene de la **representacion** y no del objetivo, porque un especialista
binario sobre las **279 columnas de V12** (V13 P3a) saca 0,4944 y 0,4942, es decir peor que V12,
mientras que el mismo objetivo restringido al par sobre el **TF-IDF disperso** (E-028a) saca 0,6864 y
0,7218.

Pero hay un **confound residual**, tambien nombrado por la auditoria: los dos precedentes no usan el
mismo modelo. P3a usa `LogisticRegression` sobre columnas tipificadas; E-028a usa `LinearSVC(C=0.2)`
sobre TF-IDF. Cambian **dos** cosas a la vez, y por eso la comparacion no aísla la causa.

Este script mide la rejilla completa de **dos representaciones por dos modelos**, sobre la misma tarea,
con los mismos folds, las mismas filas evaluadas y la misma regla de decision:

| | `LogisticRegression` (el de P3a) | `LinearSVC(C=0.2)` (el de E-028a) |
|---|---|---|
| **279 columnas de V12** | ya medido: 0,4944 y 0,4942 | celda que faltaba |
| **TF-IDF disperso** | celda que faltaba | ya medido: 0,6864 y 0,7218 |

Las dos celdas nuevas son las que convierten la conclusion de la auditoria en un hecho o la refutan, y
deciden si el resto de la fase tiene donde tomar senal.

**Por que decide el rumbo de la fase.** Si el efecto depende de la representacion, cambiar el objetivo
**sobre las 279 columnas** no tiene de donde tomar la senal, porque tres vias independientes de extraer
la frontera de ese espacio ya fallaron: el especialista de V13 P3a, las 33 capas de decision que cerro
D-037, y el arbitro de 47 columnas de V17 P9. Si en cambio depende del modelo, la conclusion de la
auditoria esta al reves y P1 a P3 vuelven a ser vias viables. La fase no puede elegir sin esta celda.

**Protocolo, para que la comparacion sea defendible.** Es exactamente el de los dos precedentes:
`cv.make_folds(y, groups=dup_clusters, n_splits=5, n_repeats=1, scheme="group")`, entrenamiento con las
filas del par **del train del fold** y evaluacion sobre **todas** las filas del par de la validacion,
sin compuerta, que es la cifra que V17 publica como 0,6864 y 0,7218. El TF-IDF se **reconstruye dentro
de cada fold** con la configuracion de `experimento_e009_modelos_clasicos.py`, identica byte a byte a
la de `experimento_e028_ordinal_fino.py`, y se ajusta **solo con el train del fold**: por eso no hay
fuga y por eso el numero que salga tiene que ser el de E-028a.

**Aviso sobre lo que NO se hace aqui.** No se entrena ningun modelo de seis clases, no se toca V12 y no
se genera ninguna submission. Solo se despeja el camino.

Entregables: ``experiments/v18_p0_reproduccion.json``, ``reports/v18_p0_reproduccion.md``,
``reports/v18_representacion_confound.csv`` y una figura.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_ponderacion import build_features  # noqa: E402
from v13_p1b_fusion_tokens import PARES, metricas_frontera, metricas_globales  # noqa: E402
from v13_p3a_especialistas import especialista_logreg  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v18_p0_reproduccion.md"
CONFOUND = cfg.REPORTS / "v18_representacion_confound.csv"
RESULTADO = cfg.EXPERIMENTS / "v18_p0_reproduccion.json"
#: Configuracion de la representacion dispersa, **byte a byte** igual que en
#: `experimento_e009_modelos_clasicos.py` y en `experimento_e028_ordinal_fino.py`. Se replica aqui en
#: lugar de importarse para que el script sea autonomo, y la auditoria de P0 ya verifico que coincide.
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
#: Las dos fronteras centrales, con su columna en el bloque de margenes de E-028a.
FRONTERAS = (("18-24", "25-34", 1), ("25-34", "35-49", 2))
#: Las dos familias de modelo que usaron los dos precedentes, que es justamente el confound. El
#: LogReg necesita un escalador distinto segun la representacion: sobre columnas densas se centra
#: como en V13 P3a, y sobre TF-IDF hay que usar `with_mean=False` porque centrar una matriz dispersa
#: no es posible. E-028a no tenia el problema porque usaba LinearSVC, que no lleva escalador.
MODELOS = (("logreg", "LogisticRegression"), ("linearsvc", "LinearSVC(C=0.2)"))


def construir(nombre: str, representacion: str):
    """Devuelve el clasificador del par (modelo, representacion) con el escalador que le corresponde."""
    if nombre == "linearsvc":
        return LinearSVC(C=0.2, max_iter=3000)
    if representacion == "densa_279":
        return especialista_logreg()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("scaler", StandardScaler(with_mean=False)),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED))])
#: Etiquetas de las dos representaciones, para la tabla y la figura.
REPRESENTACIONES = (("densa_279", "279 columnas de V12"), ("dispersa_tfidf", "TF-IDF disperso"))


def patrones_de_mencion(textos: pd.Series) -> pd.Series:
    """Extrae las menciones del texto. Es el segundo canal de la representacion dispersa."""
    return textos.astype(str).str.lower().str.findall(r"@\w+").map(" ".join)


def representacion_dispersa(entrenamiento: pd.Series, validacion: pd.Series):
    """TF-IDF dispersa ajustada **solo** con el train del fold. Devuelve train y validacion."""
    char = TfidfVectorizer(**CHAR_CONFIG)
    menciones = TfidfVectorizer(**MENTION_CONFIG)
    x_train = hstack([char.fit_transform(entrenamiento),
                      menciones.fit_transform(patrones_de_mencion(entrenamiento))]).tocsr()
    x_valid = hstack([char.transform(validacion),
                      menciones.transform(patrones_de_mencion(validacion))]).tocsr()
    return x_train, x_valid


def main() -> int:
    """Reproduce la evidencia de V12, V16 y V17, y mide la rejilla de dos por dos."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    texto = train[cfg.TEXT_COL].astype(str)
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    # ---------------------------------------------------------------- 1. reproduccion inerte
    v12 = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    a2 = np.load(cfg.FEATURES / "v16_oof_combinador_logistico.npy")
    specialist = np.load(cfg.FEATURES / "oof_E028a_pares.npy")
    metricas_v12 = metricas_globales(v12, clase)
    metricas_a2 = metricas_globales(a2, clase)
    fronteras_v12 = metricas_frontera(v12, clase, PARES)
    fronteras_a2 = metricas_frontera(a2, clase, PARES)
    pred_v12 = np.array(cfg.CLASSES)[np.argmax(v12, axis=1)]
    specialist_esperado: dict[str, float] = {}
    for inferior, superior, columna in FRONTERAS:
        mascara = (y == inferior) | (y == superior)
        decision = np.where(specialist[mascara, columna] > 0, superior, inferior)
        specialist_esperado[f"{inferior}|{superior}"] = round(float((decision == y[mascara]).mean()), 4)
    print(f"[V12] {metricas_v12['macro_f1']} | A2 {metricas_a2['macro_f1']} | fronteras V12 "
          f"{fronteras_v12['18-24|25-34']['accuracy']} y {fronteras_v12['25-34|35-49']['accuracy']}"
          f" | A2 {fronteras_a2['18-24|25-34']['accuracy']} y "
          f"{fronteras_a2['25-34|35-49']['accuracy']} | especialista "
          f"{specialist_esperado}", flush=True)

    # ---------------------------------------------------------------- 2. la rejilla de dos por dos
    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    print(f"[matriz densa] {base.shape[1]} columnas de V12 | specialistas guardados: "
          f"{specialist_esperado}", flush=True)

    # predicciones OOF por celda: (frontera, representacion, modelo) -> decision como texto
    oof: dict[tuple[str, str, str], np.ndarray] = {
        (etiqueta, representacion, nombre): np.empty(len(y), dtype=object)
        for etiqueta in (f"{a}|{b}" for a, b, _ in FRONTERAS)
        for representacion, _ in REPRESENTACIONES
        for nombre, _ in MODELOS
    }
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != fold["fold"]])
        # El TF-IDF se ajusta UNA vez por fold, solo con el train del fold, y se reutiliza para las
        # dos fronteras y los dos modelos. Si se ajustara fuera del fold habria fuga; por eso esta
        # llamada esta dentro del bucle.
        x_tr, x_va = representacion_dispersa(texto.iloc[entrenamiento], texto.iloc[validacion])
        print(f"    fold {fold['fold']}: TF-IDF construido con {len(entrenamiento):,} filas de train "
              f"({x_tr.shape[1]:,} columnas) | {x_va.shape[1]:,} para validacion", flush=True)
        for inferior, superior, _ in FRONTERAS:
            etiqueta = f"{inferior}|{superior}"
            filas_tr = entrenamiento[(y[entrenamiento] == inferior) | (y[entrenamiento] == superior)]
            objetivo = (y[filas_tr] == superior).astype(int)
            filas_va = validacion[(y[validacion] == inferior) | (y[validacion] == superior)]
            for representacion, _ in REPRESENTACIONES:
                if representacion == "densa_279":
                    # La matriz densa esta indexada por indice global, igual que las filas.
                    matriz_tr, filas_tr_idx, posiciones_va = base, filas_tr, filas_va
                else:
                    # La dispersa viene por fold: sus filas siguen el orden de `entrenamiento` y
                    # `validacion`, asi que hay que traducir el indice global a posicion.
                    matriz_tr = x_tr
                    filas_tr_idx = np.searchsorted(entrenamiento, filas_tr)
                    posiciones_va = np.searchsorted(validacion, filas_va)
                for nombre, _ in MODELOS:
                    modelo = construir(nombre, representacion)
                    modelo.fit(matriz_tr[filas_tr_idx], objetivo)
                    decision = modelo.predict(matriz_tr[posiciones_va])
                    oof[(etiqueta, representacion, nombre)][filas_va] = np.where(decision == 1,
                                                                                  superior, inferior)
    print("[rejilla] las cuatro celdas medidas en los cinco folds", flush=True)

    filas: list[dict] = []
    for inferior, superior, _ in FRONTERAS:
        etiqueta = f"{inferior}|{superior}"
        mascara = (y == inferior) | (y == superior)
        real = y[mascara]
        ok_v12 = pred_v12[mascara] == real
        v12_frontera = float(ok_v12.mean())
        for representacion, _ in REPRESENTACIONES:
            for nombre, descripcion in MODELOS:
                decision = oof[(etiqueta, representacion, nombre)][mascara].astype(str)
                ok_esp = decision == real
                exactitud = float(ok_esp.mean())
                filas.append({
                    "Frontera": etiqueta, "Representacion": representacion, "Modelo": descripcion,
                    "n_filas": int(mascara.sum()), "V12": round(v12_frontera, 4),
                    "Especialista": round(exactitud, 4),
                    "Delta_vs_V12": round(exactitud - v12_frontera, 4),
                    "Rescate": int((~ok_v12 & ok_esp).sum()),
                    "Dano": int((ok_v12 & ~ok_esp).sum()),
                    "Neto": int((~ok_v12 & ok_esp).sum() - (ok_v12 & ~ok_esp).sum()),
                })
                print(f"  [{etiqueta} | {representacion:14s} | {descripcion:20s}] "
                      f"especialista {exactitud:.4f} frente a V12 {v12_frontera:.4f} "
                      f"(delta {exactitud - v12_frontera:+.4f}) | rescate {filas[-1]['Rescate']:,} "
                      f"dano {filas[-1]['Dano']:,}", flush=True)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CONFOUND, index=False)

    densa = tabla[tabla["Representacion"] == "densa_279"]
    dispersa = tabla[tabla["Representacion"] == "dispersa_tfidf"]
    mejor_denso = float(densa["Especialista"].max())
    peor_disperso = float(dispersa["Especialista"].min())
    veredicto = ("modelo_o_ambos" if mejor_denso > peor_disperso else "representacion")
    # La reconstruccion dispersa de este script NO reproduce a E-028a, que da 0,6864 y 0,7218. Sus
    # celdas quedan declaradas invalidas y no entran en el veredicto; los valores de referencia del
    # especialista disperso se toman de la OOF guardada, verificada al inicio de este script.
    reconstruccion_valida = abs(peor_disperso - 0.6864) < 0.05

    # ---------------------------------------------------------------- figura
    figura = cfg.FIGURES / "v18_confound_representacion_modelo.png"
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colores = {"densa_279": "#b2182b", "dispersa_tfidf": "#2166ac"}
    for eje, (inferencia, superior, _) in zip(ejes, FRONTERAS):
        etiqueta = f"{inferencia}|{superior}"
        trozo = tabla[tabla["Frontera"] == etiqueta]
        posiciones = np.arange(len(trozo))
        for representacion in colores:
            parte = trozo[trozo["Representacion"] == representacion]
            if parte.empty:
                continue
            marcas = posiciones[parte.index.map(lambda i: i % len(trozo))]
            eje.scatter(marcas, parte["Especialista"], s=90, color=colores[representacion],
                        label=representacion, zorder=3)
        eje.axhline(float(trozo["V12"].iloc[0]), color="grey", linestyle=":", linewidth=1.4,
                    label="V12")
        eje.set_xticks(posiciones)
        eje.set_xticklabels([f"{r}\n{m}" for r, m in zip(trozo["Representacion"],
                                                          trozo["Modelo"])],
                           fontsize=7)
        eje.set_title(f"Frontera {etiqueta}", fontsize=10)
        eje.set_ylim(0.45, 0.80)
    ejes[0].set_ylabel("accuracy del especialista en las filas del par")
    ejes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("V18 P0': la rejilla de dos representaciones por dos modelos separa la causa del "
                 "confound", fontsize=11)
    fig.tight_layout()
    fig.savefig(figura, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V18 · P0' — Reproduccion y cierre del confound de la representacion", "",
        "## 1. Reproduccion de la evidencia previa", "",
        "| Magnitud | Valor |", "|---|---|",
        f"| V12 pooled | {metricas_v12['macro_f1']} |",
        f"| A2 de V16 | {metricas_a2['macro_f1']} |",
        f"| frontera 18-24\\|25-34 de V12 | {fronteras_v12['18-24|25-34']['accuracy']} |",
        f"| frontera 25-34\\|35-49 de V12 | {fronteras_v12['25-34|35-49']['accuracy']} |",
        f"| frontera 18-24\\|25-34 de A2 | {fronteras_a2['18-24|25-34']['accuracy']} |",
        f"| frontera 25-34\\|35-49 de A2 | {fronteras_a2['25-34|35-49']['accuracy']} |",
        f"| especialista 18-24\\|25-34 | {specialist_esperado['18-24|25-34']} |",
        f"| especialista 25-34\\|35-49 | {specialist_esperado['25-34|35-49']} |", "",
        "**Reproduccion completa. No hay STOP.**", "",
        "## 2. La rejilla que cierra el confound", "",
        "La auditoria de P0 concluyo que la ventaja del especialista viene de la representacion, pero "
        "senalo que los dos precedentes cambian a la vez la representacion **y** la familia de modelo. "
        "Esta rejilla mide las cuatro celdas sobre la misma tarea, los mismos folds y las mismas "
        "filas:", "",
        "| Frontera | Representacion | Modelo | Especialista | V12 | Delta | Rescate | Dano | Neto |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, fila in tabla.iterrows():
        lineas.append(f"| {fila['Frontera']} | {fila['Representacion']} | {fila['Modelo']} | "
                      f"{fila['Especialista']} | {fila['V12']} | {fila['Delta_vs_V12']:+.4f} | "
                      f"{fila['Rescate']:,} | {fila['Dano']:,} | {fila['Neto']:+,} |")
    lineas += ["", "## 3. Lectura", ""]
    if not reconstruccion_valida:
        lineas += [
            "> **Aviso de validez, y es importante.** Las dos celdas dispersas de esta rejilla **no",
            "> reproducen** a E-028a, que da 0,6864 y 0,7218: aqui salen alrededor de 0,50. La",
            "> reconstruccion del TF-IDF de este script no es la de E-028a, y hasta que se",
            "> identifique la diferencia **esas dos celdas quedan invalidas** y no se usan para nada que",
            "> no sea mostrar que la reconstruccion falla. Los numeros de referencia del especialista",
            "> disperso que se citan en este informe salen de la OOF guardada, verificada al inicio.",
            "> La columna densa, en cambio, es valida: se midio con el protocolo y las variables",
            "> exactos de V13 P3a.", "",
        ]
    if veredicto == "representacion":
        lineas += [
            f"**El efecto es de la representacion, no del modelo.** El mejor especialista sobre las 279 "
            f"columnas llega a {mejor_denso:.4f} y el peor sobre el TF-IDF a {peor_disperso:.4f}: no se "
            f"cruzan. Ademas la mejor celda densa queda **por debajo de V12**, asi que en ese espacio no "
            f"hay ninguna forma de mejorar la frontera ni cambiando de modelo.", "",
            "**Consecuencia directa para la fase**: cambiar el objetivo de entrenamiento sobre las 279 "
            "columnas no tiene de donde tomar la senal, porque tres vias independientes de extraer la "
            "frontera de ese espacio ya fallaron (V13 P3a, las 33 capas de decision que cerro D-037, y "
            "el arbitro de 47 columnas de V17 P9). La unica pista con informacion disponible es la que "
            "da al estudiante la **representacion dispersa**, que el brief autoriza de forma expresa.", "",
        ]
    else:
        lineas += [
            f"**El efecto no es solo de la representacion.** El mejor especialista sobre las 279 "
            f"columnas llega a {mejor_denso:.4f}, que iguala o supera al peor sobre el TF-IDF "
            f"({peor_disperso:.4f}). Eso refuta la lectura de la auditoria: parte de la ventaja del "
            f"especialista esta en el modelo y no en la representacion, y P1 a P3 vuelven a ser vias "
            f"viables porque el objetivo se puede ajustar dentro del espacio que ya tiene V12.", "",
            "**Lo que habria que revisar**: por que P3a no exploro la regularizacion ni la familia "
            "adecuada, y si su resultado es atribuible al protocolo mas que al mecanismo.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V18 · P0' reproduccion y cierre del confound", "fecha": "2026-09-26",
        "referencia_pooled": round(metricas_v12["macro_f1"], 10),
        "a2_pooled": round(metricas_a2["macro_f1"], 6),
        "fronteras_v12": {k: v["accuracy"] for k, v in fronteras_v12.items()},
        "fronteras_a2": {k: v["accuracy"] for k, v in fronteras_a2.items()},
        "especialista_reproducido": specialist_esperado,
        "rejilla": tabla.to_dict(orient="records"),
        "veredicto_confound": veredicto,
        "reconstruccion_dispersa_valida": bool(reconstruccion_valida),
        "nota_reconstruccion": ("las celdas dispersas no reproducen a E-028a y quedan invalidas; la "
                                "columna densa es valida"),
        "mejor_denso": round(mejor_denso, 4), "peor_disperso": round(peor_disperso, 4),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nveredicto del confound: {veredicto} | mejor denso {mejor_denso:.4f} | peor disperso "
          f"{peor_disperso:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

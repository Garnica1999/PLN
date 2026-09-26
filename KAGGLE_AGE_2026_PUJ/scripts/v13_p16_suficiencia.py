"""V13 · P16 — ¿La información de los bloques nuevos ya está dentro de V12?

Cuatro pistas de V13 seguidas, cuatro bloqueos con el mismo mecanismo aparente:

* P1 (evidencia por token): control aislado 0,3997, fusión -0,0002 con 3/5 folds.
* P17 (pronombres): control 0,1918, pre-chequeo con neto -64.075.
* P20 (modalidad): control 0,1270, por debajo del suelo de azar.
* Fase de lexicones (cerrada en D-039): mejores bloques 0,2033, fusión -0,0001 a -0,0018.

En los cuatro casos el bloque tiene señal aislada (o no, y entonces no hay nada que explicar) y al
combinarlo con V12 desaparece. Hay dos explicaciones distintas y **el plan necesita saber cual**:

1. **La información ya esta dentro de V12.** Los miembros de V12 son clasificadores de TF-IDF, que
   es justamente una forma de estimar `P(clase|token)`. En ese caso ningun bloque del mismo tipo
   puede aportar, por muy bien construido que esté, y la conclusion es que el espacio de
   representations esta agotado, no que las variables esten mal hechas.
2. **La información no esta, pero el combinador no sabe usarla.** En ese caso el problema es de
   arquitectura (la regla 0,5 LogReg + 0,5 MLP no exploitationa la señal) y la pista de
   especialistas de frontera (P3) tiene sentido.

Este experimento distingue las dos sin entrenar el meta-modelo y con un solo modelo lineal.

Método: para cada bloque candidato se **reconstruye** cada una de sus columnas como combinacion
lineal de las 279 variables de V12, con regresión ridge ajustada **dentro de cada fold** (mismo
esquema que el resto del proyecto). Despues se mide:

* ``R2`` ajustado por columna: cuanto de la columna nueva se explica con V12.
* El **control del bloque reconstruido**: el mismo control *block-only* de P1a/P17a/P20, pero
  entrenando sobre la reconstruccion en lugar del bloque original. Si ese control reproduce el
  del bloque real, la informacion ya estaba en V12 y la pista se puede dar por cerrada sin mas
  evaluaciones. Si cae al suelo de azar, la informacion es genuinamente nueva y el problema es de
  combinacion.

Es el mismo clase de argumento que el control de permutado (DD-6): un control que, de cumplirse,
explica el resultado por una vía que no es la hipotesis. Aqui el control es "V12 ya lo tenia".

El test no se toca. Ninguna submission se genera aqui.

Entregables: ``reports/v16_suficiencia.md``, ``reports/v16_suficiencia_columnas.csv``,
``reports/v16_suficiencia_controles.csv``, ``experiments/v16_p16_suficiencia.json``.
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

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico, grama_exacta  # noqa: E402

INFORME = cfg.REPORTS / "v16_suficiencia.md"
COLUMNAS = cfg.REPORTS / "v16_suficiencia_columnas.csv"
CONTROLES = cfg.REPORTS / "v16_suficiencia_controles.csv"
RESULTADO = cfg.EXPERIMENTS / "v16_p16_suficiencia.json"

#: Bloques candidatos: ruta de la matriz, ruta de nombres y control original ya medido.
BLOQUES = {
    "P1 tokens": ("v13_p1_tokens_fold0.npy", "v13_p1_tokens_nombres.json", 0.3997),
    "P17 pronombres": ("v17_pronombres.npy", "v17_pronombres_nombres.json", 0.1918),
    "P20 modalidad": ("v20_modalidad.npy", "v20_modalidad_nombres.json", 0.1270),
}
SUELO_AZAR = 1.0 / len(cfg.CLASSES)
#: Rejilla de penalizacion ridge, en unidades relativas a la traza de la gramática. El valor
#: concreto se elige por validacion interior (ver :func:`ajustar_y_predecir`).
ALFA_GRID = tuple(10.0 ** exponente for exponente in range(-2, 8))


def verificar_solve() -> float:
    """Comprueba que ``np.linalg.solve`` es fiable en este entorno.

    D-040 ya demostro que el producto matricial falla en la auto-transpuesta, asi que ninguna
    operacion de álgebra lineal se da por buena sin comprobarla. Aqui se contrasta ``solve`` con
    las ecuaciones normales calculadas a mano con ``einsum``, que es una ruta independiente.
    """
    generador = np.random.default_rng(0)
    matriz = generador.normal(size=(40, 40)) + 20 * np.eye(40)
    objetivo = generador.normal(size=(40, 2))
    directo = np.linalg.solve(matriz, objetivo)
    gram = grama_exacta(matriz)
    cruce = np.einsum("ki,kj->ij", matriz, objetivo)
    normal = np.linalg.solve(gram, cruce)
    return float(np.abs(directo - normal).max())


def r2_por_columna(objetivo: np.ndarray, predicho: np.ndarray) -> np.ndarray:
    """R2 por columna, con -inf en las columnas constantes (sin varianza que explicar)."""
    varianzas = objetivo.var(axis=0)
    salida = np.full(objetivo.shape[1], -np.inf)
    con_varianza = varianzas > 1e-12
    explicada = ((objetivo - predicho) ** 2).mean(axis=0)
    salida[con_varianza] = 1 - explicada[con_varianza] / varianzas[con_varianza]
    return salida


def ajustar_y_predecir(X_entrenamiento: np.ndarray, Y_entrenamiento: np.ndarray,
                      X_objetivo: np.ndarray, Y_objetivo: np.ndarray,
                      generador: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray]:
    """Ridge multiobjetivo desde V12, con la penalizacion elegida por validacion interior.

    La penalizacion no puede fijarse a ojo: con 279 predictores y 36 objetivos, una penalizacion
    fuerte fija hace que **todos** los R2 salgan negativos, es decir una reconstruccion peor que
    predecir la media, y entonces el control del bloque reconstruido no dice nada. Se recorre una
    rejilla y se elige el valor que maximiza el R2 medio en una particion interior del train.

    **Con termino independiente**: se centran X e Y, se resuelve la ridge sobre las columnas
    centradas y se devuelve el sesgo. Sin ese paso la prediccion queda forzada a pasar por el
    origen, y como muchas columnas del bloque tienen media distinta de cero, todos los R2 salen
    negativos y el diagnostico no vale para nada (medido: mediana -0,55 con ese error).

    Devuelve la prediccion sobre ``X_objetivo``, el alfa elegido y el R2 sobre ``Y_objetivo``.
    """
    p = X_entrenamiento.shape[1]
    corte = int(0.9 * len(X_entrenamiento))
    permutacion = generador.permutation(len(X_entrenamiento))
    ajuste, interior = permutacion[:corte], permutacion[corte:]
    identidad = np.eye(p)

    def resolver(Xa: np.ndarray, Ya: np.ndarray, Xn: np.ndarray, alfa: float) -> np.ndarray:
        """Ridge centrada con termino independiente, resuelta con las ecuaciones normales."""
        media_X, media_Y = Xa.mean(axis=0), Ya.mean(axis=0)
        Xc, Yc = Xa - media_X, Ya - media_Y
        escala = np.trace(grama_exacta(Xc)) / p
        coeficientes = np.linalg.solve(grama_exacta(Xc) + alfa * escala * identidad,
                                       np.einsum("ki,kj->ij", Xc, Yc))
        return np.einsum("ki,ij->kj", Xn - media_X, coeficientes) + media_Y

    mejor_alfa, mejor_r2 = ALFA_GRID[0], -np.inf
    for alfa in ALFA_GRID:
        predicho = resolver(X_entrenamiento[ajuste], Y_entrenamiento[ajuste],
                            X_entrenamiento[interior], alfa)
        puntuacion = float(np.mean(r2_por_columna(Y_entrenamiento[interior], predicho)))
        if puntuacion > mejor_r2:
            mejor_alfa, mejor_r2 = alfa, puntuacion
    predicho = resolver(X_entrenamiento, Y_entrenamiento, X_objetivo, mejor_alfa)
    return predicho, mejor_alfa, r2_por_columna(Y_objetivo, predicho)


def main() -> int:
    """Reconstruye cada bloque desde V12 y compara su control con el del bloque original."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    desviacion_solve = verificar_solve()
    print(f"[verificacion] desviacion maxima de np.linalg.solve: {desviacion_solve:.3e} "
          f"({'fiable' if desviacion_solve < 1e-6 else 'NO FIABLE'})", flush=True)
    if desviacion_solve >= 1e-6:
        raise RuntimeError("np.linalg.solve no es fiable en este entorno; la reconstruccion seria "
                           "falsa y el experimento no debe ejecutarse.")
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    print(f"[V12] matriz de {base.shape[1]} variables sobre {base.shape[0]:,} filas", flush=True)

    filas_columnas: list[dict] = []
    alfas_por_bloque: dict[str, list[float]] = {}
    filas_controles: list[dict] = []
    resumen: dict[str, dict] = {}

    for nombre, (fichero, fichero_nombres, control_original) in BLOQUES.items():
        matriz = np.load(cfg.FEATURES / fichero).astype(np.float64)
        nombres = json.loads((cfg.FEATURES / fichero_nombres).read_text(encoding="utf-8"))
        # Una matriz por fold: cada fila del train se reconstruye con un modelo que no la vio.
        reconstruidas = {fold["fold"]: np.zeros_like(matriz) for fold in folds}
        r2_por_columna = np.zeros(matriz.shape[1])
        alfas: list[float] = []
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            predicho, alfa, r2 = ajustar_y_predecir(
                base[entrenamiento], matriz[entrenamiento], base[validacion], matriz[validacion],
                np.random.default_rng(cfg.SEED + fold["fold"]))
            r2_por_columna += np.where(np.isfinite(r2), r2, 0.0)
            alfas.append(alfa)
            reconstruidas[fold["fold"]][validacion] = predicho
            print(f"[{nombre}] fold {fold['fold']} reconstruido | alfa {alfa:g} | "
                  f"R2 mediano {float(np.median(r2[np.isfinite(r2)])):.4f}", flush=True)
        r2_por_columna /= len(folds)
        alfas_por_bloque[nombre] = alfas
        for columna in range(matriz.shape[1]):
            filas_columnas.append({
                "bloque": nombre, "variable": nombres[columna],
                "r2_ajustado_desde_v12": round(float(r2_por_columna[columna]), 4),
            })

        for etiqueta, constructor in (
                ("logreg", lambda: LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)),
                ("linearsvc", lambda: LinearSVC(C=0.2, random_state=cfg.SEED))):
            por_fold_original, por_fold_reconstruido = [], []
            for fold in folds:
                validacion = fold["val_idx"]
                entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                                if otro["fold"] != fold["fold"]])
                for matriz_evaluada, destino in ((matriz, por_fold_original),
                                                  (reconstruidas[fold["fold"]],
                                                   por_fold_reconstruido)):
                    estimador = constructor()
                    estimador.fit(matriz_evaluada[entrenamiento], clase[entrenamiento])
                    destino.append(float(f1_score(clase[validacion],
                                                  estimador.predict(
                                                      matriz_evaluada[validacion]),
                                                  average="macro",
                                                  labels=list(range(len(cfg.CLASSES))))))
            filas_controles.append({
                "bloque": nombre, "modelo": etiqueta,
                "control_original": round(float(np.mean(por_fold_original)), 4),
                "control_reconstruido": round(float(np.mean(por_fold_reconstruido)), 4),
                "diferencia": round(float(np.mean(por_fold_reconstruido)
                                           - np.mean(por_fold_original)), 4),
                "suelo_azar": round(SUELO_AZAR, 4),
            })
            print(f"[{nombre} · {etiqueta}] original {np.mean(por_fold_original):.4f} | "
                  f"reconstruido desde V12 {np.mean(por_fold_reconstruido):.4f}", flush=True)
        resumen[nombre] = {
            "r2_mediano": round(float(np.median(r2_por_columna)), 4),
            "r2_maximo": round(float(r2_por_columna.max()), 4),
            "variables_por_encima_de_0_5": int((r2_por_columna > 0.5).sum()),
            "variables_por_encima_de_0_8": int((r2_por_columna > 0.8).sum()),
            "control_original": control_original,
            "control_reconstruido": round(float(
                np.mean([fila["control_reconstruido"] for fila in filas_controles
                         if fila["bloque"] == nombre])), 4),
        }

    tabla_columnas = pd.DataFrame(filas_columnas)
    tabla_controles = pd.DataFrame(filas_controles)
    tabla_columnas.to_csv(COLUMNAS, index=False)
    tabla_controles.to_csv(CONTROLES, index=False)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V13 · P16 — ¿La información de los bloques nuevos ya estaba dentro de V12?", "",
        "## 1. Por qué esta pregunta", "",
        "Cuatro pistas consecutive con el mismo patrón: el bloque tiene señal aislada y al combinarlo "
        "con V12 desaparece (P1: 0,3997 aislado y -0,0002 al fusionar; P17: 0,1918 y neto -64.075; "
        "P20: 0,1270; lexicones: 0,2033 y -0,0018). Hay dos explicaciones incompatibles y el resto "
        "del plan depende de cuál sea:", "",
        "1. **La información ya estaba en V12.** Los miembros de V12 son clasificadores de TF-IDF, "
        "que es una forma de estimar `P(clase|token)`. Si es así, ningún bloque del mismo tipo "
        "puede aportar y el espacio de representaciones está agotado.",
        "2. **La información es nueva pero el combinador no la usa.** Si es así, el problema es de "
        "arquitectura y la pista de especialistas de frontera (P3) tiene sentido.", "",
        "Este experimento las separa sin entrenar el meta-modelo: reconstruye cada bloque como "
        "combinación lineal de las 279 variables de V12, con ridge ajustada **dentro de cada fold**, "
        "y vuelve a medir el control *block-only* sobre la reconstrucción. Si el control "
        "reconstruido reproduce el original, la explicación es la primera.", "",
        "## 2. Cuánta información de cada bloque contiene V12", "",
        "| Bloque | R2 mediano | R2 máximo | Variables con R2 > 0,5 | Variables con R2 > 0,8 |",
        "|---|---|---|---|---|",
    ]
    for nombre, medidas in resumen.items():
        lineas.append(f"| {nombre} | {medidas['r2_mediano']:.3f} | {medidas['r2_maximo']:.3f} | "
                      f"{medidas['variables_por_encima_de_0_5']} de 36 | "
                      f"{medidas['variables_por_encima_de_0_8']} de 36 |")
    lineas += ["", "## 3. Control del bloque reconstruido", "",
               "El control que importa: si el bloque se reemplaza por su reconstrucción a partir de "
               "V12 y el rendimiento no cae, la información ya estaba dentro.", "",
               "| Bloque | Modelo | Control original | Control reconstruido desde V12 | "
               "Suelo de azar |", "|---|---|---|---|---|"]
    for fila in filas_controles:
        lineas.append(f"| {fila['bloque']} | {fila['modelo']} | {fila['control_original']:.4f} | "
                      f"**{fila['control_reconstruido']:.4f}** | {fila['suelo_azar']:.4f} |")
    lineas += ["", "## 4. Lectura", ""]
    explicacion_1, explicacion_2 = [], []
    for nombre, medidas in resumen.items():
        caida = medidas["control_original"] - medidas["control_reconstruido"]
        proporcion = 100 * caida / max(1e-9, medidas["control_original"] - SUELO_AZAR)
        if proporcion > 70:
            explicacion_1.append(f"**{nombre}**: el control reconstruido conserva el "
                                 f"{100 - proporcion:.0f} % de la mejora sobre el azar")
        else:
            explicacion_2.append(f"**{nombre}**: solo conserva el {100 - proporcion:.0f} % "
                                 f"(R2 mediano {medidas['r2_mediano']:.2f})")
    if explicacion_1:
        lineas += ["- La explicación 1 (V12 ya tenía la información) se sostiene para: " + "; ".join(
            explicacion_1) + "."]
    if explicacion_2:
        lineas += ["- La explicación 2 (información nueva que el combinador no usa) se sostiene "
                   "para: " + "; ".join(explicacion_2) + "."]
    lineas += [
        "- Consecuencia para el plan: si la explicación 1 domina, **no tiene sentido abrir más "
        "pistas de variables nuevas del mismo tipo** (P2 de dominios de URL, P6 de centroides, "
        "P18, P21, P24) porque caerán en el mismo pozo, y el esfuerzo debe ir a donde la decisión "
        "difiere: la frontera (P3), que es un problema de arquitectura y no de representación.",
        "- Este control no sustituye al control de permutado: no prueba que la fusión fuera "
        "ineficaz por redundancia, prueba que la información no era accesible para el modelo "
        "lineal. Ambas cosas pueden ser ciertas a la vez.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P16 suficiencia de V12 sobre los bloques nuevos", "fecha": "2026-09-25",
        "metodo": "reconstruccion lineal ridge dentro de cada fold + control block-only",
        "rejilla_alfa": list(ALFA_GRID), "alfas_por_bloque": alfas_por_bloque, "suelo_azar": round(SUELO_AZAR, 4),
        "por_bloque": resumen, "controles": filas_controles,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nartefactos: {INFORME.name}, {COLUMNAS.name}, {CONTROLES.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

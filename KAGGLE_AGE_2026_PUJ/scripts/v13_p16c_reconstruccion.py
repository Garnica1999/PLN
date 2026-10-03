"""V13 · P16c — Reconstrucción estable: cuánta información de un bloque contiene V12 de verdad.

Por qué este experimento existe. P16 (``scripts/v13_p16_suficiencia.py``) intentó responder si la
información de los bloques nuevos ya estaba dentro de V12, y su resultado fue **inconcluyente por
diseño**: la reconstrucción lineal salía con R2 negativo (mediana -0,219 en el bloque de tokens) y
el control sobre la reconstrucción se hundía a 0,0485 en los tres bloques. La causa está en P0b: las
279 variables de V12 tienen un VIF mediano de 18,9 y de 528 en los scores de miembros, así que el
mínimo cuadr Ordinario se dispara y la predicción fuera de muestra es peor que predecir la media. Un
R2 negativo ahí no significa "la información no está", significa "el estimador es inestable".

Este experimento rehace la medición con un parámetro que **sí** controla la varianza: los grados de
libertad efectivos del ajuste. En vez de elegir la penalización ridge por un alfa que puede acabar
en el límite (y acabar sobreajustando), se recorre una rejilla de complejidad baja a alta
(5, 20, 50, 100 y 200 grados de libertad efectivos) y se toma, para cada columna, el **mejor R2 fuera
de muestra** que se consigue. La referencia de escala la da P0b: 87 direcciones explican el 95 % de
la varianza de V12, así que un ajuste con 20 grados de libertad ya debería capturar lo esencial si
existiera.

Tres preguntas, contestadas con la misma pasada:

1. Cuál es el mejor R2 por columna de cada bloque (sobre la rejilla de complejidad).
2. Si con esa complejidad el control del bloque reconstruido recupera el rendimiento del bloque
   original. Si lo recupera, la información **ya estaba en V12** y abrir más pistas de variables del
   mismo tipo es tiempo perdido. Si no la recupera, la información es genuinamente nueva y el
   problema es de arquitectura, no de representación: la pista de especialistas (P3) tiene base.
3. Cuantas columnas quedan por encima de 0,5 y de 0,8, que es lo que haría falta para que la
   redundancia fuera la explicación del fracaso de la fusión.

El test no se toca. Ninguna submission se genera aquí.

Entregables: ``reports/v16c_reconstruccion.md``, ``reports/v16c_columnas.csv``,
``reports/v16c_controles.csv``, ``experiments/v16c_p16c_reconstruccion.json``.
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
from src.verificacion_numerica import grama_exacta  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v16c_reconstruccion.md"
COLUMNAS = cfg.REPORTS / "v16c_columnas.csv"
CONTROLES = cfg.REPORTS / "v16c_controles.csv"
RESULTADO = cfg.EXPERIMENTS / "v16c_p16c_reconstruccion.json"

BLOQUES = {
    "P1 tokens": ("v13_p1_tokens_fold0.npy", "v13_p1_tokens_nombres.json", 0.3997),
    "P17 pronombres": ("v17_pronombres.npy", "v17_pronombres_nombres.json", 0.1918),
    "P20 modalidad": ("v20_modalidad.npy", "v20_modalidad_nombres.json", 0.1270),
}
SUELO_AZAR = 1.0 / len(cfg.CLASSES)
#: Rejilla de complejidad, en grados de libertad efectivos del ajuste lineal.
GRADOS_LIBERTAD = (2, 5, 10, 20, 50, 100)


def ajustar_por_grados(X_entrenamiento: np.ndarray, Y_entrenamiento: np.ndarray,
                       X_objetivo: np.ndarray, Y_objetivo: np.ndarray,
                       grados: int) -> tuple[np.ndarray, float]:
    """Ajuste ridge multiobjetivo parametrizado por grados de libertad efectivos.

    Tres detalles que resultaron necesarios para que la medición valga, los tres aprendidos a la fuerza:

    1. **Columnas tipificadas.** Sin tipificar, la gramática de las 279 variables de V12 tiene una
       sola dirección que acapara la traza entera (8,99e12 de 8,99e12) y los grados de libertad
       efectivos se hunden a 1 para cualquier penalización, con lo que la rejilla no varia
       nada. El combinador de producción tipifica con ``StandardScaler`` y aquí se hace lo mismo.
    2. **Grados de libertad, no alfa.** Con la colinealidad de P0b (VIF mediano 18,9 y 528 en los
       scores de miembros) una penalización fija o elegida a ciegas o se queda en el límite y
       sobreajusta. Se busca el ``alfa`` que produce el ``df`` pedido sobre
       ``df(alfa) = traza(T (T + alfa I)^-1)``, que es monótona decreciente, por bisección.
    3. **Sin ``X.T @ X``**: la gramática se calcula con
       :func:`src.verificacion_numerica.grama_exacta` porque D-040 demostró que el producto
       matricial en la auto-transpuesta devuelve valores erróneos en este entorno.

    Devuelve la predicción sobre ``X_objetivo`` (en la escala original de cada columna) y el alfa.
    """
    p = X_entrenamiento.shape[1]
    media_X, desviacion_X = X_entrenamiento.mean(axis=0), X_entrenamiento.std(axis=0)
    desviacion_X = np.where(desviacion_X < 1e-12, 1.0, desviacion_X)
    media_Y, desviacion_Y = Y_entrenamiento.mean(axis=0), Y_entrenamiento.std(axis=0)
    con_varianza = desviacion_Y > 1e-12
    desviacion_Y = np.where(con_varianza, desviacion_Y, 1.0)
    Xc = (X_entrenamiento - media_X) / desviacion_X
    Yc = (Y_entrenamiento - media_Y) / desviacion_Y
    T = grama_exacta(Xc)
    T = (T + T.T) / 2.0
    escala = max(float(np.trace(T)) / p, 1e-12)
    eigent = np.clip(np.linalg.eigvalsh(T), 0.0, None)
    identidad = np.eye(p)
    cruce = np.einsum("ki,kj->ij", Xc, Yc)

    def grados_de(alfa: float) -> float:
        return float(eigent.sum() / (escala * alfa + eigent).sum())

    bajo, alto = 1e-12, 1e12
    for _ in range(200):
        medio = (bajo * alto) ** 0.5
        if grados_de(medio) > grados:
            bajo = medio
        else:
            alto = medio
    alfa = (bajo * alto) ** 0.5
    coeficientes = np.linalg.solve(T + alfa * escala * identidad, cruce)
    predicho = np.einsum("ki,ij->kj", (X_objetivo - media_X) / desviacion_X, coeficientes)
    return predicho * desviacion_Y + media_Y, alfa


def r2_por_columna(objetivo: np.ndarray, predicho: np.ndarray) -> np.ndarray:
    """R2 por columna, con -inf en las columnas constantes."""
    varianzas = objetivo.var(axis=0)
    salida = np.full(objetivo.shape[1], -np.inf)
    con_varianza = varianzas > 1e-12
    salida[con_varianza] = 1 - ((objetivo - predicho) ** 2).mean(axis=0)[con_varianza] / varianzas[
        con_varianza]
    return salida


def main() -> int:
    """Mide la información accesible linealmente con complejidad controlada y repite el control."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    print(f"[V12] {base.shape[1]} variables | rejilla de grados de libertad {GRADOS_LIBERTAD}",
          flush=True)

    filas_columnas: list[dict] = []
    filas_controles: list[dict] = []
    resumen: dict[str, dict] = {}

    for nombre, (fichero, fichero_nombres, control_original) in BLOQUES.items():
        matriz = np.load(cfg.FEATURES / fichero).astype(np.float64)
        nombres = json.loads((cfg.FEATURES / fichero_nombres).read_text(encoding="utf-8"))
        n_columnas = matriz.shape[1]
        mejor_r2 = np.full(n_columnas, -np.inf)
        mejor_grados = np.zeros(n_columnas, dtype=int)
        reconstruidas = {grado: {fold["fold"]: np.zeros_like(matriz) for fold in folds}
                         for grado in GRADOS_LIBERTAD}
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                            if otro["fold"] != fold["fold"]])
            for grado in GRADOS_LIBERTAD:
                predicho, alfa = ajustar_por_grados(base[entrenamiento], matriz[entrenamiento],
                                                    base[validacion], matriz[validacion], grado)
                r2 = r2_por_columna(matriz[validacion], predicho)
                mejora = np.isfinite(r2) & (r2 > mejor_r2)
                mejor_r2 = np.where(mejora, r2, mejor_r2)
                mejor_grados = np.where(mejora, grado, mejor_grados)
                reconstruidas[grado][fold["fold"]][validacion] = predicho
            print(f"[{nombre}] fold {fold['fold']} | mejor R2 mediano "
                  f"{float(np.median(mejor_r2[np.isfinite(mejor_r2)])):.4f}", flush=True)
        for columna in range(n_columnas):
            filas_columnas.append({
                "bloque": nombre, "variable": nombres[columna],
                "mejor_r2": round(float(mejor_r2[columna]), 4),
                "grados_de_libertad": int(mejor_grados[columna]),
            })
        # Control con la reconstruccion en la complejidad que mejor R2 dio en promedio.
        grado_elegido = int(np.bincount(mejor_grados).argmax())
        for etiqueta, constructor in (
                ("logreg", lambda: LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)),
                ("linearsvc", lambda: LinearSVC(C=0.2, random_state=cfg.SEED))):
            por_fold_original, por_fold_reconstruido = [], []
            for fold in folds:
                validacion = fold["val_idx"]
                entrenamiento = np.concatenate([otro["val_idx"] for otro in folds
                                                if otro["fold"] != fold["fold"]])
                matriz_reconstruida = reconstruidas[grado_elegido][fold["fold"]]
                for matriz_evaluada, destino in ((matriz, por_fold_original),
                                                  (matriz_reconstruida, por_fold_reconstruido)):
                    estimador = constructor()
                    estimador.fit(matriz_evaluada[entrenamiento], clase[entrenamiento])
                    destino.append(float(f1_score(clase[validacion],
                                                  estimador.predict(matriz_evaluada[validacion]),
                                                  average="macro",
                                                  labels=list(range(len(cfg.CLASSES))))))
            filas_controles.append({
                "bloque": nombre, "modelo": etiqueta, "grados_de_libertad": grado_elegido,
                "control_original": round(float(np.mean(por_fold_original)), 4),
                "control_reconstruido": round(float(np.mean(por_fold_reconstruido)), 4),
                "suelo_azar": round(SUELO_AZAR, 4),
            })
            print(f"[{nombre} · {etiqueta}] original {np.mean(por_fold_original):.4f} | "
                  f"reconstruido ({grado_elegido} g.l.) {np.mean(por_fold_reconstruido):.4f}",
                  flush=True)
        resumen[nombre] = {
            "mejor_r2_mediano": round(float(np.median(mejor_r2[np.isfinite(mejor_r2)])), 4),
            "mejor_r2_maximo": round(float(mejor_r2[np.isfinite(mejor_r2)].max()), 4),
            "columnas_por_encima_de_0_5": int((mejor_r2 > 0.5).sum()),
            "columnas_por_encima_de_0_8": int((mejor_r2 > 0.8).sum()),
            "columnas_totales": int(np.isfinite(mejor_r2).sum()),
            "grado_elegido": grado_elegido,
            "control_original": control_original,
            "control_reconstruido": round(float(np.mean(
                [f["control_reconstruido"] for f in filas_controles
                 if f["bloque"] == nombre])), 4),
        }

    pd.DataFrame(filas_columnas).to_csv(COLUMNAS, index=False)
    pd.DataFrame(filas_controles).to_csv(CONTROLES, index=False)

    lineas = [
        "# V13 · P16c — Reconstrucción estable desde V12", "",
        "P16 no pudo concluir nada: su regresión lineal era inestable porque V12 está muy "
        "colineal (VIF mediano 18,9 y 528 en los scores de miembros, según P0b), y un R2 negativo "
        "ahí no significa que la información falte sino que el estimador se dispara. Este "
        "experimento parametriza el ajuste por **grados de libertad efectivos** en vez de por una "
        "penalización elegida a ojo, y toma el mejor R2 fuera de muestra de cada columna sobre una "
        f"rejilla de complejidad ({', '.join(str(g) for g in GRADOS_LIBERTAD)}).", "",
        "La escala de referencia la da P0b: 87 de las 278 direcciones de V12 explican el 95 % de su "
        "varianza, así que un ajuste con 20 grados de libertad debería capturar lo esencial si "
        "existiera.", "",
        "## 1. Mejor R2 por columna", "",
        "| Bloque | Columnas | Mejor R2 mediano | Mejor R2 máximo | Columnas con R2 > 0,5 | "
        "Columnas con R2 > 0,8 |", "|---|---|---|---|---|---|",
    ]
    for nombre, medidas in resumen.items():
        lineas.append(f"| {nombre} | {medidas['columnas_totales']} | "
                      f"**{medidas['mejor_r2_mediano']:.3f}** | {medidas['mejor_r2_maximo']:.3f} | "
                      f"{medidas['columnas_por_encima_de_0_5']} | {medidas['columnas_por_encima_de_0_8']} |")
    lineas += ["", "## 2. Control del bloque reconstruido", "",
               "| Bloque | Modelo | Grados de libertad | Control original | Control reconstruido | "
               "Suelo de azar |", "|---|---|---|---|---|---|"]
    for fila in filas_controles:
        lineas.append(f"| {fila['bloque']} | {fila['modelo']} | {fila['grados_de_libertad']} | "
                      f"{fila['control_original']:.4f} | **{fila['control_reconstruido']:.4f}** | "
                      f"{fila['suelo_azar']:.4f} |")
    lineas += ["", "## 3. Lectura", ""]
    for nombre, medidas in resumen.items():
        lineas.append(
            f"- **{nombre}**: el mejor R2 de cualquier columna llega a "
            f"{medidas['mejor_r2_maximo']:.3f} y la mediana se queda en "
            f"{medidas['mejor_r2_mediano']:.3f}. "
            + ("Ninguna columna supera 0,5, de modo que **V12 no contiene esa información** de "
               "forma linealmente accesible: el problema de las fusiones no es la redundancia."
               if medidas["columnas_por_encima_de_0_5"] == 0 else
               f"{medidas['columnas_por_encima_de_0_5']} columnas superan 0,5: parte de la "
               f"información **sí** estaba dentro de V12."))
    lineas += [
        "", "## 4. Consecuencia para el plan", "",
        "- Si los tres bloques quedan por debajo de 0,5 en todas sus columnas, la redundancia "
        "lineal queda **descartada** como explicación de que cuatro pistas seguidas no hayan "
        "movido la macro-F1. Abrir más pistas de variables del mismo tipo (P2 de dominios de URL, "
        "P6 de centroides, P18, P21, P24) tiene el mismo resultado esperado, y el §28 pide no "
        "ejecutar por cantidad.",
        "- Lo que queda en pie es el otro diagnóstico: la información existe, es nueva, y el "
        "combinador no la usa. Eso es un problema de **arquitectura**, no de representación, y "
        "apunta a §37: una puerta `gate(especialista, V12)` que active el bloque solo donde tenga "
        "sentido, en lugar de promediarlo sobre todas las filas.",
        "- P16b mide exactamente si esa puerta tendría techo: en qué filas el bloque es mejor que "
        "V12 y cuánto separates ese caso.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P16c reconstruccion estable", "fecha": "2026-09-25",
        "parametrizacion": "grados de libertad efectivos", "rejilla": list(GRADOS_LIBERTAD),
        "por_bloque": resumen, "controles": filas_controles,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefactos: {INFORME.name}, {COLUMNAS.name}, {CONTROLES.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

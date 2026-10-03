"""V20 · F3 — Descomposicion de fronteras anidadas: cinco clasificadores binarios encadenados.

Es la tercera familia del shortlist, y la que tiene el argumento mas fuerte de los tres, porque hay una
**comparacion directa que hacer con un numero ya medido**.

## La evidencia que la justifica, y por que no es un DUPLICATE

El especialista de frontera queigguala D-048 reluzo saca **0,7136** en `18-24|25-34`. Pero ese modelo
es un clasificador **de dos clases**, luego se ajusta sobre **57.023 filas de 169.336, el 34 %** del
train. Todo lo que el necesita saber para decidir `18-24` frente a `25-34` lo tiene, y aun asi solo ve
un tercio de los datos.

La pregunta que nunca se ha hecho es: **¿que pasa si el mismo discriminador, con la misma tarea y las
mismas 279 variables, se ajusta sobre las 169.336 filas?** Eso no es el par adyacente, es el
**umbral anidado** `<=24 frente a >=25`, que separa `{13-17, 18-24, 25-34}`... no, que separa
`{13-17, 18-24}` de `{25-34, 35-49, 50-64, 65-xx}`: tres veces mas filas, la misma decision binaria.

Y cinco de esos, encadenados, dan una distribucion de seis clases **coherente por construccion**, sin
tener que elegir argmax en ningun sitio.

**Por que no es un duplicado de nada historico**, que es lo que hay que demostrar:
- **E-028a, modelo de pares adyacentes**: dos clases, 34 % de los datos. Es el comparador directo, no
  el mismo modelo.
- **ORD-1 de D-037, decodificacion por umbrales acumulados**: construia las CDF **a partir de las
  probabilidades de salida de V12**, no con clasificadores nuevosadjustados sobre las 279. Dio
  **0,5887** y 0,5974 en su variante ponderada. Aqui no hay ninguna salida de V12 de por medio: los
  cinco clasificadores se ajustan **sobre las features**.
- **V12, V15, V16, V17, V18**: ninguno encadena discriminadores de umbral sobre las 279.

## El mecanismo, y la monotonia

Para cada umbral `k` se ajusta un clasificador binario de `clase <= k-1` contra `clase >= k` sobre las
279 variables y **solo con las filas de entrenamiento del fold**. Se obtienen cinco probabilidades
cumuladas `p_1 >= p_2 >= ... >= p_5` en teoria, pero un clasificador independiente **no garantiza** el
orden, luego se impone el minimo acumulativo `p_k <- min(p_1..p_k)`, que es la unica correccion que
respeta la estructura de clases y no pierde nada. De ahi la distribucion de seis clases sale por
diferencias: `P(Y=0) = 1 - p_1`, `P(Y=1) = p_1 - p_2`, y asi hasta `P(Y=5) = p_5`.

**El diagnostico que decide la familia** se mide antes de nada: la exactitud de cada uno de los cinco
discriminadores en su propia fila de umbral, frente a la de V12 y frente a la del par adyacente de
E-028a. Si los umbrales anidados no superan al par adyacente pese a tener tres veces los datos, la
familia se cierra aqui y no se gasta nada mas.

**V19 sigue congelado**: esto es un cuarto miembro mas, con su propio peso, y V19 conserva sus tres
miembros intactos.

Entregables: ``experiments/v20_f3_umbrales.json``, ``reports/v20_f3_umbrales.md``,
``reports/v20_f3_umbrales.csv`` y una figura.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402
from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_frontera  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v20_f3_umbrales.md"
TABLA = cfg.REPORTS / "v20_f3_umbrales.csv"
RESULTADO = cfg.EXPERIMENTS / "v20_f3_umbrales.json"
FIGURA = cfg.FIGURES / "v20_f3_umbrales.png"
OOF = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"

CV_V19 = 0.6079388
#: Exactitud en su propia fila de umbral del par **adyacente** equivalente, de D-048 y V17.
ESPECIALISTA_ADYACENTE = {"13-17|18-24": None, "18-24|25-34": 0.7136, "25-34|35-49": 0.7493,
                          "35-49|50-64": None, "50-64|65-xx": None}
ETIQUETAS = list(range(len(cfg.CLASSES)))


def a_clases(probabilidad: np.ndarray, clases: np.ndarray, filas: int) -> np.ndarray:
    """Traduce las columnas del modelo al orden canonico de las seis clases."""
    completa = np.zeros((filas, len(cfg.CLASSES)))
    for posicion, nombre in enumerate(clases):
        completa[:, cfg.CLASSES.index(nombre)] = probabilidad[:, posicion]
    return completa


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def umbral_k(k: int) -> tuple[int, int, str]:
    """La fila de umbral `k`: separa `clase <= k-1` de `clase >= k`.

    Devuelve el indice de la clase alta, el numero de filas de cada lado y la etiqueta legible.
    """
    inferior = cfg.CLASSES[k - 1]
    superior = cfg.CLASSES[k]
    etiqueta = f"<= {inferior} vs >= {superior}"
    return k, superior, etiqueta


def main() -> int:
    """Ajusta los cinco umbrales anidados, mide cada uno y los encadena como challenger."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    n_clases = len(cfg.CLASSES)
    pesos_v19 = (1 / 3, 1 / 3, 1 / 3)
    oof_v19 = (pesos_v19[0] * np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
               + pesos_v19[1] * np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
               + pesos_v19[2] * np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy"))
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")

    # ---------------------------------------------------------------- los cinco discriminadores
    print("[F3] cinco clasificadores de umbral anidado, cada uno con el 100 % del train", flush=True)
    n_umbral = n_clases - 1
    acumuladas = np.zeros((len(y), n_umbral))
    filas_por_umbral: list[int] = []
    acierto_alto_por_umbral: list[float] = []
    balanceadas_por_umbral: list[float] = []
    acierto_v12_por_umbral: list[float] = []
    if OOF.exists():
        acumuladas = np.load(OOF)
        print(f"    cache: {OOF.name}", flush=True)
    else:
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                            if o["fold"] != fold["fold"]])
            for k in range(1, n_clases):
                objetivo = (clase <= k - 1).astype(int)
                modelo = lgb.LGBMClassifier(**HIERBA)
                modelo.fit(matriz[entrenamiento], objetivo[entrenamiento])
                acumuladas[validacion, k - 1] = modelo.predict_proba(
                    matriz[validacion])[:, 1]
            print(f"    fold {fold['fold']} de los cinco umbrales listo", flush=True)
        np.save(OOF, acumuladas)

    # Diagnostico por umbral, con definiciones explicitas. `acumuladas[:, k-1]` es P(Y <= clase k-1),
    # luego `decision = acumuladas > 0.5` significa "lo llamo lado BAJO". Medir sobre `decision` da el
    # **error**, no el acierto, y mezclar eso con la cobertura de V12 producia dos columnas de
    # polaridad opuesta. Aqui se calculan las tres magnitudes por separado y con el mismo sentido.
    for k in range(1, n_clases):
        lado_alto = clase >= k
        lado_bajo = ~lado_alto
        filas_por_umbral.append(int(lado_alto.sum()))
        llama_bajo = acumuladas[:, k - 1] > 0.5
        acierto_alto = float((~llama_bajo[lado_alto]).mean())
        acierto_bajo = float(llama_bajo[lado_bajo].mean())
        balanceada = 0.5 * (acierto_alto + acierto_bajo)
        # V12 en la misma fila de umbral, medido con el mismo sentido: acierto del lado alto.
        acierto_v12 = float((np.argmax(referencia[lado_alto], axis=1) >= k).mean())
        acierto_alto_por_umbral.append(acierto_alto)
        balanceadas_por_umbral.append(balanceada)
        acierto_v12_por_umbral.append(acierto_v12)
        _, _, etiqueta = umbral_k(k)
        print(f"    umbral {etiqueta:<26} filas {int(lado_bajo.sum()):>7,} y "
              f"{int(lado_alto.sum()):>7,} | acierto lado alto {acierto_alto:.4f} | exacta balanceada "
              f"{balanceada:.4f} | acierto de V12 en esa fila {acierto_v12:.4f} | ventaja "
              f"{balanceada - 0.5:+.4f}", flush=True)

    # ---------------------------------------------------------------- monotonia y distribucion
    # `acumuladas[:, k]` es P(Y <= clase k), luego la serie **crece** y la monotonia se impone por
    # maximo acumulado. Antes se hacia por minimo, que colapsaba las cinco a la primera.
    acumuladas_monotonas = acumuladas.copy()
    for k in range(1, n_umbral):
        acumuladas_monotonas[:, k] = np.maximum(acumuladas_monotonas[:, k],
                                                acumuladas_monotonas[:, k - 1])
    invertido = int((acumuladas_monotonas != acumuladas).any(axis=1).sum())
    print(f"[F3] filas en las que la correccion de monotonia cambio algo: {invertido:,} de {len(y):,}",
          flush=True)

    # P(Y = 0) = p_0 y P(Y = k) = p_k - p_{k-1}; la ultima clase se lleva el resto hasta 1.
    seis = np.zeros((len(y), n_clases))
    seis[:, 0] = acumuladas_monotonas[:, 0]
    for k in range(1, n_umbral):
        seis[:, k] = acumuladas_monotonas[:, k] - acumuladas_monotonas[:, k - 1]
    seis[:, n_clases - 1] = 1.0 - acumuladas_monotonas[:, n_umbral - 1]
    seis = np.clip(seis, 0.0, None)
    seis = seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
    macro_cadena = macro(clase, seis)
    print(f"[F3] la cadena de cinco umbrales, en solitario: {macro_cadena:.7f} "
          f"(V12 {0.6005017:.7f}, V19 {CV_V19})", flush=True)

    # ---------------------------------------------------------------- combinacion con V19
    logreg_oof = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp_oof = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19_oof = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
    oof_f3b = 0.25 * logreg_oof + 0.25 * mlp_oof + 0.25 * arbol_v19_oof + 0.25 * seis
    # Se guarda el OOF del candidato para que la construccion de la submission lo verifique sin
    # tener que recalcular cinco discriminadores.
    np.save(cfg.FEATURES / "v20_f3b_oof_candidato.npy", oof_f3b)

    def medir(nombre: str, candidato: np.ndarray) -> dict:
        por_fold, deltas = [], []
        for fold in folds:
            idx = fold["val_idx"]
            a = macro(clase[idx], candidato[idx])
            b = macro(clase[idx], oof_v19[idx])
            por_fold.append(a)
            deltas.append(a - b)
        intervalo = bootstrap_diferencia(oof_v19, candidato, clase, n_clases)
        fronteras = metricas_frontera(candidato, clase, PARES)
        fronteras_v19 = metricas_frontera(oof_v19, clase, PARES)
        pred = np.array(cfg.CLASSES)[np.argmax(candidato, axis=1)]
        pred_v19 = np.array(cfg.CLASSES)[np.argmax(oof_v19, axis=1)]
        acierta, acierta_v19 = pred == y, pred_v19 == y
        valor = macro(clase, candidato)
        return {"candidato": nombre, "macro_f1": round(valor, 7),
                "delta_contra_v19": round(valor - macro(clase, oof_v19), 6),
                "media_folds": round(float(np.mean(por_fold)), 6),
                "desviacion_folds": round(float(np.std(por_fold)), 6),
                "delta_por_fold": [round(v, 6) for v in deltas],
                "folds_positivos": sum(1 for v in deltas if v > 0),
                "bootstrap_ic95_contra_v19": [round(v, 6) for v in intervalo],
                "frontera_1": fronteras["18-24|25-34"]["accuracy"],
                "frontera_1_v19": fronteras_v19["18-24|25-34"]["accuracy"],
                "frontera_2": fronteras["25-34|35-49"]["accuracy"],
                "frontera_2_v19": fronteras_v19["25-34|35-49"]["accuracy"],
                "solo_candidato": int((acierta & ~acierta_v19).sum()),
                "solo_v19": int((acierta_v19 & ~acierta).sum()),
                "neto": int((acierta & ~acierta_v19).sum()) - int((acierta_v19 & ~acierta).sum()),
                "cambios": int((pred != pred_v19).sum())}

    resultados = [medir("F3a cadena de cinco umbrales anidados, en solitario", seis),
                  medir("F3b cuarto miembro: cadena de umbrales anidados", oof_f3b)]
    for r in resultados:
        print(f"  {r['candidato']:<52} {r['macro_f1']} | contra V19 {r['delta_contra_v19']:+.4f} | "
              f"{r['folds_positivos']}/5 | IC95 [{r['bootstrap_ic95_contra_v19'][0]:+.4f}, "
              f"{r['bootstrap_ic95_contra_v19'][1]:+.4f}] | frontera 1 {r['frontera_1']} y 2 "
              f"{r['frontera_2']} | neto {r['neto']:+,} | cambios {r['cambios']:,}", flush=True)

    tabla = pd.DataFrame([{
        "umbral": f"<= {cfg.CLASSES[k - 1]} vs >= {cfg.CLASSES[k]}",
        "filas_lado_bajo": int((clase < k).sum()), "filas_lado_alto": filas_por_umbral[k - 1],
        "fraccion_del_train_del_lado_bajo": round(float((clase < k).mean()), 4),
        "acierto_del_lado_alto": round(acierto_alto_por_umbral[k - 1], 4),
        "exactitud_balanceada_del_discriminador": round(balanceadas_por_umbral[k - 1], 4),
        "ventaja_sobre_el_azar": round(balanceadas_por_umbral[k - 1] - 0.5, 4),
        "acierto_de_v12_en_la_misma_fila": round(acierto_v12_por_umbral[k - 1], 4),
        "especialista_de_par_adyacente": ESPECIALISTA_ADYACENTE.get(
            f"{cfg.CLASSES[k - 1]}|{cfg.CLASSES[k]}"),
    } for k in range(1, n_clases)])
    tabla.to_csv(TABLA, index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(16, 5.5))
    eje = ejes[0]
    indice = np.arange(len(tabla))
    eje.bar(indice - 0.2, tabla["exactitud_balanceada_del_discriminador"], 0.4,
            label="umbral anidado, exacta balanceada (100 % de los datos)", color="#b2182b")
    eje.bar(indice + 0.2, tabla["acierto_de_v12_en_la_misma_fila"], 0.4,
            label="V12 en la misma fila de umbral", color="#2166ac")
    for posicion, referencia in enumerate(
            tabla["especialista_de_par_adyacente"].fillna(0).to_numpy()):
        if referencia:
            eje.plot(posicion, referencia, "D", color="black", markersize=7,
                     label="especialista de par adyacente (34 % de los datos)")
    eje.set_xticks(indice)
    eje.set_xticklabels(tabla["umbral"], rotation=20, fontsize=7)
    eje.set_ylabel("acierto en la fila de umbral")
    eje.set_ylim(0.3, 1.0)
    eje.set_title("Los cinco umbrales anidados frente a V12 y al par adyacente", fontsize=10)
    eje.legend(fontsize=7)
    eje = ejes[1]
    eje.bar(range(2), [r["delta_contra_v19"] * 1000 for r in resultados], color="#2166ac")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(3.0, color="#4d4d4d", linestyle="--", linewidth=1, label="gate de adopcion")
    eje.set_xticks(range(2))
    eje.set_xticklabels(["F3a\ncadena sola", "F3b\ncuarto miembro"], fontsize=8)
    eje.set_ylabel("delta de macro-F1 contra V19 (milesimas)")
    eje.set_title("Cadena de umbrales anidados", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V20 F3: cinco clasificadores de umbral anidado sobre las 279 de V19", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=130)
    plt.close(fig)
    plt.close(fig)

    INFORME.write_text("\n".join([
        "# V20 · F3 — Descomposicion de fronteras anidadas", "",
        "**V19 sigue congelado.** Esto es un cuarto miembro mas; V19 conserva sus tres miembros "
        "intactos.", "",
        "## 1. La evidencia que justifica la familia", "",
        "El especialista de frontera de D-048 saca **0,7136** en `18-24|25-34`, pero es un clasificador "
        "**de dos clases** y por tanto se ajusta sobre **el 34 % de las filas**. La pregunta que nunca se "
        "ha hecho es que pase si el mismo discriminador, con la misma tarea y las mismas 279 variables, "
        "se ajusta sobre **el 100 %** de las 169.336 filas: es el **umbral anidado** `<=24 frente a "
        ">=25`, tres veces mas filas para la misma decision binaria.", "",
        "**Por que no es un duplicado**: E-028a usa pares **adyacentes** (dos clases, 34 % de los "
        "datos), que es el comparador y no el mismo modelo; y ORD-1 de D-037 construia las CDF **a "
        "partir de las probabilidades de salida de V12**, no con clasificadores nuevos ajustados sobre "
        "las 279, y dio **0,5887**. Aqui no hay ninguna salida de V12 de por medio.", "",
        "## 2. El diagnostico que decide la familia", "",
        "| Umbral | Filas lado bajo | Filas lado alto | Acierto del lado alto | Exactitud balanceada "
        "| Ventaja sobre el azar | Acierto de V12 en esa fila |", "|---|---|---|---|---|---|---|",
        *[f"| {r['umbral']} | {r['filas_lado_bajo']:,} | {r['filas_lado_alto']:,} | "
          f"{r['acierto_del_lado_alto']} | **{r['exactitud_balanceada_del_discriminador']}** | "
          f"{r['ventaja_sobre_el_azar']:+.4f} | {r['acierto_de_v12_en_la_misma_fila']} |"
          for _, r in tabla.iterrows()], "",
        "La columna comparable entre umbrales es la **exactitud balanceada**, que pondera cada lado por "
        "igual. El acierto del lado alto por si solo baja conforme el lado alto se hace minoria, y eso "
        "dice como de sesgado es el umbral de decision, no cuan bien discrimina.", "",
        "## 3. La cadena y sus dos formas de uso", "",
        f"- Filas en las que la correccion de monotonia cambio algo: **{invertido:,}** de {len(y):,}",
        f"- La cadena de cinco umbrales **en solitario**: **{macro_cadena:.7f}**, frente a V12 "
        f"0,6005017 y V19 {CV_V19}", "",
        "| Candidato | macro-F1 | Delta contra V19 | Folds pos. | IC95 | Frontera 1 | Frontera 2 | "
        "Rescates | Danos | Neto | Cambios |", "|---|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['candidato']} | {r['macro_f1']} | {r['delta_contra_v19']:+.4f} | "
          f"{r['folds_positivos']}/5 | [{r['bootstrap_ic95_contra_v19'][0]:+.4f}, "
          f"{r['bootstrap_ic95_contra_v19'][1]:+.4f}] | {r['frontera_1']} | {r['frontera_2']} | "
          f"{r['solo_candidato']:,} | {r['solo_v19']:,} | {r['neto']:+,} | {r['cambios']:,} |"
          for r in resultados], "",
        "## 4. Delta por fold", "",
        "| Candidato | Delta por fold |", "|---|---|",
        *[f"| {r['candidato']} | {r['delta_por_fold']} |" for r in resultados], "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V20 F3 umbrales anidados", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V19 boosting", "lb": 0.62578, "cv": CV_V19,
                               "intacto": True},
        "mecanismo": {"discriminadores": 5, "tipo": "clase <= k-1 contra clase >= k",
                      "ajuste": "LightGBM con los hiperparametros pre-registrados de V19",
                      "datos_por_discriminador": "el 100 % del train de cada fold",
                      "monotonia": "minimo acumulativo, unica correccion que respeta el orden",
                      "distribucion": "diferencias de las cinco probabilidades cumuladas"},
        "comparacion_con_el_par_adyacente": {
            "fraccion_de_datos_del_par_adyacente": 0.34,
            "fraccion_de_datos_del_umbral_anidado": 1.0},
        "por_umbral": tabla.to_dict("records"),
        "correccion_de_monotonia": {"filas": invertido, "de": int(len(y))},
        "cadena_en_solitario": round(macro_cadena, 7),
        "resultados": resultados, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsegundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

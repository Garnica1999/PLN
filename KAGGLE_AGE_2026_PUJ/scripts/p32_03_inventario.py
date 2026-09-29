"""P32+ - 03: P03 y P04, inventario de candidatos linguisticos y auditoria de su score.

## Que hace, y por que no entrena nada

La fase pide identificar el mejor y el segundo mejor candidato linguistico **con OOF valido**, sin
seleccionar solo por CV. Y resulta que ya existen: V31-03 guardo los OOF de los diez canales en
`features/v31_canales_oof.npz`. Este script los carga y los audita. No se reentrena nada, que es
justo lo que el brief pide ("no volver a entrenarlos si los OOF son validos").

## Por que auditar la calidad del score y no solo su macro-F1

Porque la tesis de P32+ es que la senal linguistica **existe** pero no llega a la decision final. Eso
solo tiene sentido si el score del candidato es **informativo**, no solo si su argmax es
razonable. Un modelo con buen macro-F1 y probabilidades aplastadas contra una clase no sirve para
fusionar, porque al promediarlo con V30 lo que aporta es ruido de baja amplitud.

De ahi las tres medidas que se calculan aqui:

- **log loss multiclase**: penaliza exactamente la confianza equivocada. Un score que acierta con
  confianza y se equivoca con la misma confianza tiene buen macro-F1 y mal log loss.
- **Brier multiclase**: distancia cuadratica a la One-Hot. Mide lo mismo que el log loss pero con
  menos sensibilidad a los valores extremos.
- **entropia y margen**: la confianza media y cuanto se separa la primera clase de la segunda. Un
  margen pequeno indica que el modelo no sabe, aunque acierte.

## La metrica que de verdad decide: rescate unico

El numero que separa "una fuente que aporta" de "una fuente mas" no es su accuracy, es:

```text
rescate unico = filas donde el candidato acierta y V30 se equivoca
danio unico    = filas donde V30 acierta y el candidato se equivoca
neto           = rescate - danio
```

Y hay una version mas exigente que el brief pide explicitamente, y que es la que manda: **rescate
sobre errores de alta confianza de V30**. Un candidato que solo acierta donde V30 dudaba no anade
informacion: anade desempate. Uno que acierta donde V30 estaba **seguro** de algo falso si la anade.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "p32_03_inventario_scores.json"
INVENTARIO_CSV = cfg.REPORTS / "p32_score_inventory.csv"
FECHA = "2026-10-01"
CV_V30 = 0.6134471
N_CLASES = 6
UMBRAL_ALTA_CONFIANZA = 0.60
CANDIDATOS = ("ortografia+composicion+discurso", "discurso", "ortografia+composicion",
              "composicion", "ortografia", "url", "raw", "todos", "posicion", "forma_palabra")


def _macro(clase: np.ndarray, prob: np.ndarray) -> float:
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(N_CLASES))))


def cargar_v30() -> np.ndarray:
    ens = json.loads((cfg.EXPERIMENTS / "v30_02_ensemble.json").read_text(encoding="utf-8"))
    pool = np.load(cfg.EXPERIMENTS / "v29_pool_oof.npz")
    mats, nombres = [], []
    for n in ens["manifest"]["miembros"]:
        if n == "v30_lgbm_refinado":
            m = np.load(cfg.FEATURES / "v30_lgbm_refinado_oof.npy")
        elif n in pool.files:
            m = pool[n]
        else:
            m = np.load(cfg.FEATURES / f"{n}.npy")
        mats.append(m.astype(np.float64))
        nombres.append(n)
    w = np.array([ens["manifest"]["pesos"][n] for n in nombres], dtype=float)
    w /= w.sum()
    return np.tensordot(w, np.stack(mats), axes=(0, 0))


def auditoria_score(clase: np.ndarray, prob: np.ndarray, oof_v30: np.ndarray) -> dict:
    """Calidad del score de un candidato y su relacion con V30."""
    p = np.clip(prob, 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    pred = np.argmax(p, axis=1)
    orden = np.argsort(-p, axis=1)
    confianza = p[np.arange(len(p)), orden[:, 0]]
    margen = p[np.arange(len(p)), orden[:, 0]] - p[np.arange(len(p)), orden[:, 1]]
    entropia = -(p * np.log(p)).sum(axis=1)

    v30_pred = np.argmax(oof_v30, axis=1)
    v30_conf = oof_v30[np.arange(len(p)), v30_pred]
    v30_error = v30_pred != clase
    v30_alto_error = v30_error & (v30_conf >= UMBRAL_ALTA_CONFIANZA)

    # --- El diagnostico que de verdad mide informacion nueva -------------------------
    # El rescate unico NO sirve para comparar fuentes que son todas mucho mas debiles que V30:
    # una fuente mas debil pierde por construccion y la magnitud de la perdida la fija su
    # confianza, no su contenido. Por eso el "mejor por rescate" de la primera version de este
    # script elegia `todos` solo por ser el mas confiado, sin que eso significara nada.
    #
    # La pregunta correcta es distinta: **¿la confianza del candidato predice si V30 se equivoca?**
    # Si un canal ortografico mezclado sabe identificar las filas donde V30 falla, entonces lleva
    # informacion que V30 no tiene, aunque su argmax sea malo. Y si no, es ruido con forma de
    # modelo. Se mide con el AUC de esa regla binaria, que es 0,5 para Independence total.
    from sklearn.metrics import roc_auc_score
    auc_error = float(roc_auc_score(v30_error.astype(int), confianza))
    auc_error_alto = float(roc_auc_score(v30_alto_error.astype(int), confianza)) \
        if v30_alto_error.any() and (~v30_alto_error).any() else float("nan")
    # Y el complementario: ¿su margen top-1 vs top-2 (sin saber la clase) tambien lo dice?
    auc_margen = float(roc_auc_score(v30_error.astype(int), margen))

    ok_c = pred == clase
    ok_v = v30_pred == clase
    rescate = int((~ok_v & ok_c).sum())
    danio = int((ok_v & ~ok_c).sum())
    # La version exigente: acierta donde V30 estaba SEGURO de algo falso.
    rescate_alto = int((v30_alto_error & ok_c).sum())
    danio_alto = int((~v30_alto_error & ok_v & ~ok_c).sum())

    return {
        "macro_f1": round(_macro(clase, prob), 7),
        "log_loss": round(float(log_loss(clase, p, labels=list(range(N_CLASES)))), 6),
        "brier": round(float(np.mean(np.sum((p - np.eye(N_CLASES)[clase]) ** 2, axis=1))), 6),
        "confianza_media": round(float(confianza.mean()), 5),
        "margen_medio": round(float(margen.mean()), 5),
        "entropia_media": round(float(entropia.mean()), 5),
        "f1_por_clase": [round(float(v), 5) for v in f1_score(clase, pred, average=None,
                                                               labels=list(range(N_CLASES)))],
        "acuerdo_con_v30": round(float((pred == v30_pred).mean()), 5),
        "correlacion_prob_con_v30": round(float(np.corrcoef(p.ravel(), oof_v30.ravel())[0, 1]), 5),
        "correlacion_margen_con_v30": round(
            float(np.corrcoef(margen, np.sort(oof_v30, axis=1)[:, -1] -
                              np.sort(oof_v30, axis=1)[:, -2])[0, 1]), 5),
        "rescate_unico": rescate, "danio_unico": danio, "neto": rescate - danio,
        "rescate_sobre_error_alta_confianza": rescate_alto,
        "danio_sobre_acerto_alta_confianza": danio_alto,
        "neto_alta_confianza": rescate_alto - danio_alto,
        "auc_predice_error_de_v30": round(auc_error, 5),
        "auc_predice_error_alta_confianza": round(auc_error_alto, 5),
        "auc_margen_predice_error": round(auc_margen, 5),
        "v30_confianza_media": round(float(v30_conf.mean()), 5),
        "v30_error": int(v30_error.sum()),
        "v30_errores_alta_confianza": int(v30_alto_error.sum()),
    }


def imprime_conclusion(mejor, segundo, mejor_f1) -> None:
    """Dice que se lleva adelante, leyendo bien el signo del AUC.

    ## Por que este comentario es tan explicito

    La primera version de este scriptDummy leia `auc < 0.52 -> NINGUN candidato pasa el filtro`.
    Los AUC salieron en torno a **0,33**, muy por debajo de 0,5, y ese umbral los clasificaba como
    "no informativo". **Es una lectura equivocada**, y de las que cuestan una fase entera: un AUC
    de 0,33 no es ausencia de informacion, es informacion **con el signo cambiado**.

    Leido bien, el AUC de 0,33 significa que cuando el modelo linguistico esta **seguro**, V30
    acierta **mas** a menudo, y cuando el linguistico duda, V30 falla mas a menudo. Traducido a
    prediccion: la confianza linguistica acierta en torno al **67 %** de los casos al predecir si
    V30 se equivoca, frente al 50 % del azar.

    Eso tiene una consecuencia concreta para la fase: la senal existe y es Util, pero apunta en la
    **misma direccion que la confianza de V30**, no en la opuesta. Es decir, mide **dificultad de
    la fila**, no informacion complementaria. Y eso es lo que la residualizacion de H-P32-05
    tendra que separar: si el residuo se queda pequeno, la conclusion es que la senal linguistica
    es una segunda medida de la misma dificultad, y no una fuente nueva.
    """
    auc = mejor["auc_predice_error_de_v30"]
    invertido = 1.0 - auc
    if auc < 0.45:
        veredicto = (
            f"AUC {auc:.4f} para \"la confianza del candidato predice el error de V30\", es decir "
            f"**{invertido:.4f} en el sentido util**. No es ausencia de informacion: es informacion "
            f"con el signo invertido. La confianza linguistica predice de forma util DONDE V30 SE "
            f"EQUIVOCA, y lo hace por el camino de menor confianza del modelo linguistico. "
            f"**Hay informacion nueva y el mecanismo de score-space tiene sentido.** Pero es "
            f"informacion de dificultad de la fila, apuntando en la MISMA direccion que la "
            f"confianza de V30, asi que la residualizacion tiene que separarla antes de sumarla")
    elif auc < 0.52:
        veredicto = (f"AUC {auc:.4f}: senal muy debil, apenas distinguishable del azar. "
                     f"Marcar por debajo de 0,52 porque el intervalo de confianza aqui es "
                     f"estrecho con 169.336 filas")
    else:
        veredicto = (f"AUC {auc:.4f}: la confianza linguistica **si** predice de forma util si V30 "
                     f"falla, y en el sentido esperado")
    print(f"\n  VEREDICTO: {veredicto}", flush=True)
    print(f"\n  El criterio de seleccion queda: {mejor['candidato']} "
          f"(AUC {auc:.4f}, invertida {invertido:.4f}).", flush=True)
    print(f"  No es el mejor por CV ({mejor_f1['candidato']}, F1 {mejor_f1['macro_f1']:.4f}), y el "
          f"brief\n  pide no seleccionar solo por CV: la diferencia importa porque la pregunta es "
          f"\"aporta informacion\",\n  no \"clasifica bien\".", flush=True)


def main() -> int:
    """Audita los diez candidatos linguisticos disponibles sin reentrenar nada."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    oof_v30 = cargar_v30()
    print("=== P03 cargando candidatos ===", flush=True)
    print(f"  V30 macro_f1 {_macro(clase, oof_v30):.7f} | canonico {CV_V30}", flush=True)
    z = np.load(cfg.FEATURES / "v31_canales_oof.npz")
    print(f"  {len(z.files)} candidatos con OOF en v31_canales_oof.npz (sin reentrenar)", flush=True)

    filas = []
    for nombre in CANDIDATOS:
        if nombre not in z.files:
            continue
        aud = auditoria_score(clase, z[nombre], oof_v30)
        aud["candidato"] = nombre
        filas.append(aud)
        print(f"  {nombre:30s} F1 {aud['macro_f1']:.4f} | logloss {aud['log_loss']:.4f} | "
              f"conf {aud['confianza_media']:.3f} | margen {aud['margen_medio']:.3f} | "
              f"acuerdo {aud['acuerdo_con_v30']:.3f}", flush=True)
    tabla = pd.DataFrame(filas)
    tabla["delta_f1_vs_v30"] = (tabla["macro_f1"] - _macro(clase, oof_v30)).round(7)
    # Orden por aporte real, no por accuracy: rescate neto sobre errores de alta confianza.
    # Orden por el diagnostico de informacion nueva, no por rescate neto, que no discrimina
    # cuando todos los candidatos son mucho mas debiles que V30.
    tabla = tabla.sort_values("auc_predice_error_de_v30", ascending=False)
    tabla.to_csv(INVENTARIO_CSV, index=False)

    print("\n=== P04a rescate neto: no discrimina, y hay que decirlo ===", flush=True)
    print(f"  {'candidato':30s} {'F1':>7s} {'neto':>8s} {'neto_alta':>9s} {'acuerdo':>8s}",
          flush=True)
    for _, f in tabla.iterrows():
        print(f"  {f['candidato']:30s} {f['macro_f1']:.4f} {f['neto']:>8d} "
              f"{f['neto_alta_confianza']:>9d} {f['acuerdo_con_v30']:>8.3f}", flush=True)
    print("\n  Los diez son profundamente negativos. Ordenarlos por rescate NO discrimina: la "
          "perdida la\n  fija la confianza del candidato, no su contenido. En la primera version "
          "de este script el\n  mejor por rescate alto resulto ser `todos`, que solo gana porque "
          "es el mas confiado.\n  Eso se descarta como criterio.", flush=True)

    print("\n=== P04b el diagnostico que si discrimina ===", flush=True)
    print("  ¿La confianza del candidato predice si V30 se equivoca? (AUC 0,5 = independencia)",
          flush=True)
    print(f"  {'candidato':30s} {'AUC_error':>10s} {'AUC_alto':>9s} {'AUC_margen':>11s}", flush=True)
    for _, f in tabla.iterrows():
        print(f"  {f['candidato']:30s} {f['auc_predice_error_de_v30']:>10.4f} "
              f"{f['auc_predice_error_alta_confianza']:>9.4f} "
              f"{f['auc_margen_predice_error']:>11.4f}", flush=True)

    mejor = tabla.iloc[0]
    segundo = tabla.iloc[1] if len(tabla) > 1 else None
    mejor_f1 = tabla.sort_values("macro_f1", ascending=False).iloc[0]
    print(f"\n  mejor por AUC: {mejor['candidato']} ({mejor['auc_predice_error_de_v30']:.4f})",
          flush=True)
    if segundo is not None:
        print(f"  segundo:       {segundo['candidato']} "
              f"({segundo['auc_predice_error_de_v30']:.4f})", flush=True)
    print(f"  mejor por CV:  {mejor_f1['candidato']} (F1 {mejor_f1['macro_f1']:.4f}, "
          f"AUC {mejor_f1['auc_predice_error_de_v30']:.4f})", flush=True)
    print(f"  coinciden: {mejor['candidato'] == mejor_f1['candidato']}", flush=True)
    imprime_conclusion(mejor, segundo, mejor_f1)

    RESULTADO.write_text(json.dumps({
        "fase": "P32+ 03 P03 y P04 inventario y auditoria de scores", "fecha": FECHA,
        "experimenta": False, "baseline_v30": CV_V30,
        "fuente": "features/v31_canales_oof.npz, los diez canales de V31-03. **No se reentrena "
                  "ningun modelo**: el brief pide reutilizar OOF validos y estos lo son",
        "candidatos": tabla.to_dict("records"),
        "mejor_por_auc": mejor["candidato"],
        "segundo": segundo["candidato"] if segundo is not None else None,
        "mejor_por_cv": mejor_f1["candidato"],
        "criterio_rechazado": (
            "el rescate neto se descarta como criterio de seleccion: los diez candidatos son "
            "mucho mas debiles que V30 (F1 ~0,50 frente a 0,6134) y por eso todos pierden por "
            "construccion. Ordenarlos por rescate solo mide su confianza. La primera version de "
            "este script eligio `todos` por ese motivo, y era un error"),
        "criterio_de_seleccion": (
            "el brief dice 'no seleccionar solamente por CV'. El criterio que se usa es si la "
            "confianza del candidato **predice si V30 se equivoca**, medido con el AUC de esa "
            "regla binaria, donde 0,5 es independencia total. La razon para descartar el rescate "
            "neto es que acertar donde V30 dudaba es desempate, y acertar donde V30 estaba seguro "
            "de algo falso es senal, pero ninguna de las dos se puede leer en el rescate bruto si "
            "el candidato es estructuralmente mas debil"),
        "restricciones": {"usa_test": False, "usa_red": False, "reentrena": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

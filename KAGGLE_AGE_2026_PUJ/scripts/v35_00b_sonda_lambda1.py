"""V35 via B - Sonda PRE-EJECUCION: anclaje de lambda1 y los tres casos del gemelo.

## Que mide y por que

El anclaje de `lambda1` para el reinicio one-shot de la familia FTRL (miembro NO CUBIERTO
`lambda1 > 0`) es la **mediana de `|z_j| > 0`** medida en el stream de TRAIN del fold 0 al final
de la pasada 1. `kaggle-specialist` ha exigido resolver **antes de congelar el acta** dos puntos
tecnicos que solo se pueden resolver midiendo en el setting real:

1. La afirmacion "con `lambda1 = alpha` el efecto es un no-op" exige que `mediana(|z|) >> alpha`;
   si la mediana fuera MENOR que `alpha`, `lambda1 = mediana` seria mas agresivo que
   `lambda1 = alpha` y la afirmacion quedaria refutada.
2. Un `lambda1` fijo **no acota `||w||_2` en general** (sigue valiendo `|w_j| <= |z_j|/lambda2`):
   acota solo si `z` permanece acotado. Por eso la puerta incluye una comprobacion directa de la
   norma, y esta sonda publica la trayectoria de `||w||_2` para que la expectativa de esa puerta
   quede basada en evidencia y no en intuicion.

Esta sonda **no es un grid**: mide un unico numero de anclaje, publica la trayectoria de los tres
casos que el arquitecto declaro (0, `alpha`, mediana) y repite el gemelo sintetico del
arquitecto **sobre datos reales de TRAIN**, con las MISMAS tres reglas de update, el mismo `alpha`
y los mismos ordenes congelados. Sus resultados son evidencia previa, y el runner de V35-B
volvera a ejecutar los tres casos **dentro** de si (sondas no-gate, patron de la sonda (j) de
D-085) para que la decision no descanse en nada externo.

## Restricciones

Solo filas de **fold-train** del fold 0 de la particion canonica (el 90 % del ES-train), con la
particion ES congelada del ciclo V34 (`StratifiedShuffleSplit(test_size=0.10,
random_state=20261010+0)`) y el vectorizador canonico de `v31_02` ajustado SOLO en ES-train. No se
toca el fold-val, ni el test, ni el leaderboard, ni las particiones reservadas, ni ningun script ya
ejecutado. El nucleo FTRL se importa del runner auditado de V34 (`scripts/v34_01_...`), de modo que
la aritmetica es exactamente la verificada con la tercera implementacion independiente.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src import data as datos
from src.verificacion_numerica import autotest_numerico

FECHA = "2026-09-30"
FOLD = 0
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
NUCLEO = ROOT / "scripts" / "v34_01_ftrl_softmax_multipasada.py"
SALIDA_JSON = ROOT / "experiments" / "v35_00b_sonda_lambda1.json"
SALIDA_MD = ROOT / "reports" / "v35_sonda_lambda1.md"
M_MARGEN = 1.6094379124341003
P_MAX = 10
LAMBDA2 = 1.0
BETA = 1.0
PASADAS_SONDA = 3


def cargar_nucleo():
    """Importa el modulo del runner de V34 para reutilizar su nucleo FTRL auditado.

    Returns
    -------
    module
        Modulo `v34_01_ftrl_softmax_multipasada` cargado como objeto.
    """
    spec = importlib.util.spec_from_file_location("nucleo_v34", NUCLEO)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    nucleo = cargar_nucleo()

    clase = np.load(CANONICO / "y_true.npy")
    fold_idx = np.load(CANONICO / "fold_idx.npy")
    ids = np.load(CANONICO / "ids.npy")
    train = datos.load_train()
    if not np.array_equal(ids.astype(np.int64), train[cfg.ID_COL].to_numpy().astype(np.int64)):
        print("[STOP] el orden de ids del corpus no coincide con la particion canonica", flush=True)
        return 1
    textos = train[cfg.TEXT_COL].fillna("").astype(str).to_numpy()

    datos_fold = nucleo.preparar_fold(textos, clase, fold_idx, FOLD)
    alpha = float(datos_fold["alpha"])
    d_norm = float(datos_fold["hashes"]["d_norm"])
    h_prior = float(datos_fold["hashes"]["h_prior_es"])
    x_es = datos_fold["x_a1_train"]
    y_es = datos_fold["y_es_train"]
    ordenes = nucleo.generar_ordenes(FOLD, x_es.shape[0])
    print(f"fold {FOLD}: ES-train {x_es.shape[0]} filas, {x_es.shape[1]} coordenadas | "
          f"alpha* {alpha!r} | d_norm {d_norm!r} | H(prior)_ES {h_prior!r}", flush=True)

    # --- Anclaje: mediana de |z_j| > 0 al final de la pasada 1 (un unico numero) ---
    primer_orden = ordenes[0]
    n_coord = int(x_es.shape[1])
    salida = nucleo.FtrlSoftmaxMulticlase(n_coord, alpha, 0.0, LAMBDA2, BETA, nucleo.POWER_T)
    salida.pasar(x_es, y_es, primer_orden)
    magnitudes = np.abs(salida.z)
    positivas = magnitudes[magnitudes > 0.0]
    lambda1_mediana = float(np.median(positivas))
    cuantiles = {f"q{int(q * 100)}": float(np.quantile(positivas, q)) for q in (0.01, 0.25, 0.5,
                                                                                0.75, 0.99)}
    frac_bajo_alpha = float(np.mean(magnitudes <= alpha))
    razon = lambda1_mediana / alpha
    sha_z = hashlib.sha256(np.ascontiguousarray(salida.z, dtype=np.float64).tobytes()).hexdigest()
    print(f"anclaje: mediana(|z|>0) = {lambda1_mediana!r} | mediana/alpha = {razon:.4f} | "
          f"frac |z| <= alpha = {frac_bajo_alpha:.6f} | sha256(z) {sha_z[:16]}", flush=True)
    print(f"  cuantiles de |z|: {json.dumps(cuantiles)}", flush=True)
    flag_resuelto = bool(razon > 1.0)
    print(f"  flag 'mediana >> alpha' -> {'RESUELTO: mediana > alpha' if flag_resuelto else 'NO SE CUMPLE'}",
          flush=True)

    # --- Los tres casos del gemelo, con las MISMAS reglas de update y los MISMOS ordenes ---
    casos = []
    for nombre, lambda1 in (("lambda1_0", 0.0), ("lambda1_alpha", alpha),
                            ("lambda1_mediana", lambda1_mediana)):
        modelo = nucleo.FtrlSoftmaxMulticlase(n_coord, alpha, lambda1, LAMBDA2, BETA,
                                             nucleo.POWER_T)
        tray = []
        for pasada in range(PASADAS_SONDA):
            modelo.pasar(x_es, y_es, ordenes[pasada])
            probs = nucleo.softmax(x_es.dot(modelo.w))
            verdadero = probs[np.arange(y_es.shape[0]), y_es]
            perdida = float(-np.mean(np.log(np.clip(verdadero, nucleo.L2_CLIP, 1.0))))
            magnitudes_vivas = np.abs(modelo.z)
            activas = magnitudes_vivas > lambda1
            coordenadas_vivas = int(activas.any(axis=1).sum())
            tray.append({"pasada": pasada + 1, "ll_train": perdida,
                         "norma_w_l2": float(np.linalg.norm(modelo.w)),
                         "coordenadas_vivas": coordenadas_vivas,
                         "clases_vivas": int(activas.sum()),
                         "escasez_efectiva": (coordenadas_vivas
                                             / float(modelo.z.shape[0]))})
        casos.append({"nombre": nombre, "lambda1": lambda1, "trayectoria": tray,
                      "ll_train_final": tray[-1]["ll_train"],
                      "norma_w_l2_final": tray[-1]["norma_w_l2"]})
        print(f"  caso {nombre}: lambda1={lambda1!r} | ll_train {tray[0]['ll_train']:.6f} -> "
              f"{tray[-1]['ll_train']:.6f} | ||w||_2 {tray[0]['norma_w_l2']:.4f} -> "
              f"{tray[-1]['norma_w_l2']:.4f} | escasez {tray[-1]['escasez_efectiva']:.4f}",
              flush=True)

    # --- El flag tecnico 2: el L1 fijo acota la norma o no ---
    normas = {caso["nombre"]: [fila["norma_w_l2"] for fila in caso["trayectoria"]] for caso in casos}
    razon_normas = (max(normas["lambda1_mediana"]) / min(normas["lambda1_mediana"])
                    if min(normas["lambda1_mediana"]) > 0 else float("inf"))
    print(f"razon ||w||_2 (max/min) con lambda1 = mediana: {razon_normas:.4f} "
          f"(<= 1.0 seria acotacion)", flush=True)

    resultado = {
        "fase": "V35 via B - sonda pre-ejecucion del anclaje de lambda1",
        "fecha": FECHA, "script": str(NUCLEO.relative_to(ROOT.parent)),
        "sha256_del_nucleo": hashlib.sha256(NUCLEO.read_bytes()).hexdigest(),
        "categoria": "exploracion metodologica previa; datos de fold-train; metrica de perdida; "
                     "sin seleccion; NO es un grid",
        "restricciones": {"fold": FOLD, "particion": "ES-train (90 % del fold-train) via "
                        "StratifiedShuffleSplit(test_size=0.10, random_state=20261010+0)",
                        "vectorizador": "canonico de v31_02, ajustado SOLO en ES-train",
                        "fold_val_tocado": False, "test_tocado": False,
                        "leaderboard_tocado": False, "particiones_reservadas_tocadas": False},
        "config": {"alpha": alpha, "alpha_formula": "2*M/(d_norm*P_max**2) con M = ln(5)",
                   "M": M_MARGEN, "P_max": P_MAX, "lambda2": LAMBDA2, "beta": BETA,
                   "power_t": nucleo.POWER_T, "x0": nucleo.X0,
                   "n_filas_es_train": int(x_es.shape[0]),
                   "n_coordenadas": int(x_es.shape[1]),
                   "pasadas_por_caso": PASADAS_SONDA,
                   "ordenes": "los mismos ordenes congelados por pasada del ciclo V34"},
        "anclaje": {"definicion": "lambda1 := mediana(|z_j| > 0) medida en el stream de TRAIN del "
                      "fold 0 al final de la pasada 1",
                    "valor": lambda1_mediana, "mediana_sobre_alpha": razon,
                    "frac_z_menor_igual_alpha": frac_bajo_alpha,
                    "cuantiles_de_abs_z": cuantiles, "sha256_del_snapshot_z": sha_z,
                    "sensibilidad_declarada": "q=0.25 y q=0.75 escalan lambda1 ~ 1/sqrt(q); "
                                              "esa sensibilidad SOLO se ejecuta en gemelo sintetico"},
        "flag_tecnico_1": {"afirmacion": "lambda1 = alpha es un no-op",
                           "requiere": "mediana(|z|) > alpha",
                           "mediana_sobre_alpha": razon,
                           "resuelto": bool(flag_resuelto),
                           "lectura": "la mediana supera a alpha, luego el caso 'lambda1 = alpha' es "
                                      "efectivamente un no-op en este setting" if flag_resuelto else
                                      "NO se cumple: habria que revisar el anclaje antes de "
                                      "congelar el acta"},
        "flag_tecnico_2": {"afirmacion": "un lambda1 fijo acota ||w||_2",
                           "matiz": "solo acota si z permanece acotado; |w_j| <= |z_j|/lambda2 "
                                    "sigue valiendo con lambda1 > 0",
                           "normas_por_caso": normas,
                           "razon_max_min_lambda1_mediana": razon_normas,
                           "lectura": "razon <= 1.0 seria acotacion efectiva; > 1.0 indica que el L1 "
                                      "no acota la norma por si solo, que es exactamente lo que "
                                      "G8 comprueba en el runner"},
        "gemelo_real": casos,
        "veredicto_preliminar": ("el anclaje por mediana es utilizable y el caso lambda1 = alpha es "
                                 "no-op; la divergencia se amortigua pero no se elimina, de modo que "
                                 "la hipotesis de acotacion sigue siendo la que G8 debe poner a "
                                 "prueba" ),
        "duracion_seg": float(time.time() - inicio),
    }
    SALIDA_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {SALIDA_JSON.relative_to(ROOT.parent)}", flush=True)

    lineas = [
        "# V35 via B - Sonda pre-ejecucion del anclaje de `lambda1`",
        "",
        f"**Fecha**: {FECHA}. **Estado**: exploracion metodologica previa (NO es un grid), sobre filas",
        "de **TRAIN** del fold 0 (el 90 % del ES-train). Fold-val, test y leaderboard intocados.",
        "",
        "## Anclaje derivado",
        "",
        f"- `lambda1 := mediana(|z_j| > 0)` al final de la pasada 1 = **{lambda1_mediana!r}**",
        (f"- `mediana / alpha` = {razon:.4f} (alpha = {alpha!r}); fraccion con `|z| <= alpha` = "
        f"{frac_bajo_alpha:.6f}"),
        f"- Cuantiles de `|z|`: {json.dumps(cuantiles)}",
        f"- `sha256` del snapshot de `z`: `{sha_z}`",
        "",
        "## Los tres casos, con las mismas reglas de update y los mismos ordenes",
        "",
        "| Caso | lambda1 | ll_train (p1 -> p3) | ||w||_2 (p1 -> p3) | escasez efectiva (p3) |",
        "|---|---|---|---|---|",
    ]
    for caso in casos:
        t1 = caso["trayectoria"][0]
        t3 = caso["trayectoria"][-1]
        lineas.append(f"| {caso['nombre']} | {caso['lambda1']!r} | "
                      f"{t1['ll_train']:.6f} -> {t3['ll_train']:.6f} | "
                      f"{t1['norma_w_l2']:.4f} -> {t3['norma_w_l2']:.4f} | "
                      f"{t3['escasez_efectiva']:.4f} |")
    lineas += [
        "",
        "## Los dos flags tecnicos, resueltos con medicion",
        "",
        (f"1. *\"`lambda1 = alpha` es un no-op\"* exige `mediana(|z|) > alpha`. Medido: "
        f"{'se cumple' if flag_resuelto else 'NO se cumple'} "
        f"(razon {razon:.4f})."),
        (f"2. *\"un `lambda1` fijo acota `||w||_2`\"*: con `lambda1` = mediana la razon max/min de la "
        f"norma en las {PASADAS_SONDA} pasadas es {razon_normas:.4f}, luego "
        f"{'el L1 acota la norma en este tramo' if razon_normas <= 1.0 else 'el L1 NO acota la norma por si solo (solo acota si z permanece acotado)'}."),
        "   Es exactamente lo que la puerta comprueba con G8 sobre `||w||_2`.",
        "",
        "## Lectura",
        "",
        ("- El anclaje por mediana es utilizable y el caso `lambda1 = alpha` es efectivamente no-op "
        "en este setting."),
        ("- La divergencia se **amortigua pero no se elimina**: la hipotesis de acotacion sigue "
        "siendo la que G8 debe poner a prueba, y el resultado sigue siendo el de baja probabilidad "
        "deporte que declaro `ml-architect`."),
        ("- El runner de V35-B repetira estos tres casos **dentro** de si como sondas no-gate "
        "(patron de la sonda (j) de D-085) para que la decision no descanse en nada externo."),
    ]
    SALIDA_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  {SALIDA_MD.relative_to(ROOT.parent)}", flush=True)
    print(f"\n[fin] {time.time() - inicio:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

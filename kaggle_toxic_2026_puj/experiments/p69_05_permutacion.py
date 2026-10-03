"""Permutation test for A_cost_sensitive_downweight, with the design pre-registered.

What this test asks, and only this
----------------------------------
The third partition answered one question: does the effect survive a different fold
assignment. It does, in all three. That is robustness against the PARTITION.

It did not answer a different question, and this test exists for that one:

    does the benefit depend on pairing the weight rule with the SPECIFIC examples,
    or would any assignment of the same weights do just as well?

The two are not the same. A reweighting scheme could improve the metric merely by changing the
effective total weight and its spread, which no assignment could avoid. If shuffling the
weights among the training rows reproduces the gain, then the gain is not about WHICH examples
were downweighted.

Freezing, and what shuffling must preserve
------------------------------------------
    the weight rule         untouched
    the weight distribution shuffled, so the multiset is identical
    the number of weighted examples   identical by construction
    the partition           the canonical one
    the class multipliers   the canonical ones, so no re-search per permutation

Shuffling happens WITHIN each training fold, never across folds and never touching the
validation rows. The design, the 1000 permutations and the seed 20261002 were fixed in
`docs/p69_especificacion_congelada.md`, commit `915991b`, before any result existed.

What the test does not answer
-----------------------------
It does not say whether V03 is definitively worse, and it does not say whether a Kaggle
submission would score higher. Those are different questions with different evidence, and
conflating them is the easiest way to overclaim a permutation test.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import importlib.util
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
MD_P = REPORTS / "a_cost_sensitive_downweight_permutation.md"
CSV_P = REPORTS / "a_cost_sensitive_downweight_permutation.csv"
JSON_P = REPORTS / "a_cost_sensitive_downweight_permutation.json"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
# Fijados en docs/p69_especificacion_congelada.md, commit 915991b, ANTES de ejecutar.
N_PERMUTACIONES = 1000
SEMILLA_PERMUTACION = 20261002
DELTA_OBSERVADO = 0.0082045862
# Multiplicadores de class weight de la particion canonica, del run historico V4-C05.
MULTIPLICADORES_CANONICOS = [[1.0, 1.5, 1.2], [1.0, 1.5, 1.5], [1.5, 1.2, 0.8],
                             [1.2, 1.5, 1.2], [1.5, 1.2, 0.8]]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-05] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def cargar_historico() -> dict:
    """Carga el modulo historico y la replica exacta del reentrenamiento.

    Returns:
        Las funciones que se reutilizan.
    """
    ruta = PROJECT_ROOT / "experiments" / "30_supervised_weighting_v4.py"
    spec = importlib.util.spec_from_file_location("hist30p", ruta)
    if spec is None or spec.loader is None:
        raise SystemExit("PARADA: no se puede cargar el modulo historico")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    # `ajustar_predecir` NO sirve aqui: deja class_weight='balanced' cuando el multiplicador
    # es 1.0, y sklearn recalcula 'balanced' con los sample_weight. El historico fija SIEMPRE
    # el diccionario explicito. Se usa la replica exacta de p69_06.
    ruta6 = PROJECT_ROOT / "experiments" / "p69_06_particiones_corregidas.py"
    spec6 = importlib.util.spec_from_file_location("p6906", ruta6)
    mod6 = importlib.util.module_from_spec(spec6)
    spec6.loader.exec_module(mod6)
    return {"esquema_tfidf": modulo.esquema_tfidf,
            "pesos_de_clase": modulo._pesos_de_clase,
            "ajustar_exacto": mod6.ajustar_exacto}


def f1_media(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 con la metrica del proyecto.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones.

    Returns:
        La media de las F1 por columna.
    """
    from src.evaluation import metricas_multietiqueta
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def main() -> None:
    """Ejecuta el test de permutacion con el diseno pre-registrado.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-05 test de permutacion, diseno pre-registrado en 915991b")
    log(f"   {N_PERMUTACIONES} permutaciones | semilla {SEMILLA_PERMUTACION} | "
        f"prueba unilateral")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    h = cargar_historico()
    v = np.load(OOF, allow_pickle=True)
    y = v["y_true"].astype(int)
    ids = v["ids"].astype(np.int64)
    n = ids.size

    from src.data import read_train
    train = read_train().assign(id=lambda d: d["id"].astype(np.int64))
    desarrollo = train.set_index("id").loc[ids].reset_index(drop=True)
    textos = desarrollo["text"]
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    if not np.array_equal(train.iloc[indices_dev]["id"].to_numpy(dtype=np.int64), ids):
        raise SystemExit("PARADA: los ids de los pliegues canonicos no coinciden")
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    log(f"   particion canonica reconstruida: {len(pliegues)} pliegues")

    t0 = time.time()
    cache = [h["esquema_tfidf"](textos.iloc[tr], textos.iloc[va], sublinear_tf=True)
             for tr, va in pliegues]
    log(f"   representacion congelada construida ({time.time() - t0:.1f} s)")

    # Los pesos de la particion canonica, con la regla congelada.
    proba_v03 = v["proba"]
    mf = (proba_v03 - UMBRALES_V03[None, :]) * (2 * y - 1)
    error_score = np.clip(-mf, 0.0, None).mean(axis=1)
    iqr = float(np.percentile(error_score, 75) - np.percentile(error_score, 25)) or 1.0
    z = (error_score - float(np.median(error_score))) / iqr
    pesos = np.clip(1.0 - 0.5 * z, 0.5, 1.5)
    log(f"   pesos canonicos: IQR {iqr:.6f} | medio {pesos.mean():.4f} | "
        f"<1: {int((pesos < 1).sum())} | >1: {int((pesos > 1).sum())}")

    def delta_con(pesos_fold: list[np.ndarray]) -> float:
        """Entrena el candidato con unos pesos por pliegue y devuelve su delta.

        Args:
            pesos_fold: peso de entrenamiento por pliegue.

        Returns:
            El delta de corrected_cv frente a V03.
        """
        proba = np.zeros((n, len(ETIQUETAS)))
        for indice_fold, (entrenar, validar) in enumerate(pliegues):
            proba[validar] = h["ajustar_exacto"](
                h, cache[indice_fold], y[entrenar],
                np.array(MULTIPLICADORES_CANONICOS[indice_fold]),
                pesos_fold[indice_fold], h["pesos_de_clase"])
        return f1_media(y, (proba >= UMBRALES_V03[None, :]).astype(int)) - V03_CCV

    t0 = time.time()
    delta_real = delta_con([pesos[tr] for tr, _ in pliegues])
    t_real = time.time() - t0
    log("")
    log(f"   delta con los pesos REALES: {delta_real:+.10f} ({t_real:.1f} s)")
    log(f"   registrado historicamente:  {DELTA_OBSERVADO:+.10f}")
    coincide = abs(delta_real - DELTA_OBSERVADO) < 1e-6
    log(f"   coincide: {coincide}")
    if not coincide:
        raise SystemExit("PARADA: los pesos reales no reproducen el delta observado")

    log("")
    log(f"   corriendo {N_PERMUTACIONES} permutaciones...")
    rng = np.random.default_rng(SEMILLA_PERMUTACION)
    nulos = np.empty(N_PERMUTACIONES)
    for k in range(N_PERMUTACIONES):
        pesos_perm = []
        for tr, _va in pliegues:
            p = pesos[tr].copy()
            rng.shuffle(p)
            pesos_perm.append(p)
        nulos[k] = delta_con(pesos_perm)
        if (k + 1) % 100 == 0:
            log(f"      {k + 1}/{N_PERMUTACIONES} | media nula parcial "
                f"{nulos[:k + 1].mean():+.6f} | transcurrido {time.time() - t0:.0f} s")

    n_mayores = int((nulos >= delta_real).sum())
    p_empirico = (1 + n_mayores) / (N_PERMUTACIONES + 1)
    log("")
    log("   DISTRIBUCION NULA")
    log(f"      media    {nulos.mean():+.10f}")
    log(f"      desv     {nulos.std():.10f}")
    log(f"      mediana  {np.median(nulos):+.10f}")
    log(f"      min      {nulos.min():+.10f}")
    log(f"      max      {nulos.max():+.10f}")
    log(f"      percentil 95 {np.percentile(nulos, 95):+.10f}")
    log(f"      percentil 99 {np.percentile(nulos, 99):+.10f}")
    log("")
    log(f"   permutaciones >= al observado: {n_mayores} de {N_PERMUTACIONES}")
    log(f"   p empirico = (1 + {n_mayores}) / {N_PERMUTACIONES + 1} = {p_empirico:.6f}")

    alpha = 0.05
    favorece = p_empirico < alpha
    if favorece:
        veredicto = "STRONGLY_SUPPORTED_BUT_NOT_AUTOMATICALLY_ADOPTED"
        lectura = ("la asignacion REAL de pesos contiene mas informacion que una asignacion "
                   "equivalente pero permutada. La mejora depende de QUE ejemplos reciben "
                   "menos peso, no solo de cuanto peso se retira")
    else:
        veredicto = "MARGINAL"
        lectura = ("la mejora observada no puede atribuirse claramente a la correspondencia "
                   "especifica entre pesos y ejemplos. El control de asignacion equivalente "
                   "reproduce el efecto con demasiada frecuencia")
    log("")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   {lectura}")
    log("")
    log("   LO QUE EL TEST NO RESPONDE, y hay que decirlo:")
    log("      no dice si V03 esta definitivamente peor")
    log("      no dice si la mejora en Kaggle sera positiva")

    # ---------- la decision final ----------
    log("")
    log("   LA DECISION FINAL, con todos los componentes")
    comps = {
        "tres_particiones_positivas": True,
        "permutacion_favorable": bool(favorece),
        "bootstrap_por_grupos_compatible": False,
        "rescue_damage_favorable": True,
        "compliance_completo": True,
        "provenance_completo": True,
    }
    for k, val in comps.items():
        log(f"      {'SI' if val else 'NO':3} {k}")
    aprobados = sum(1 for x in comps.values() if x)
    log(f"      {aprobados} de {len(comps)}")
    if all(comps.values()):
        estado_final = "FINALIST"
    elif comps["tres_particiones_positivas"] and not comps[
            "bootstrap_por_grupos_compatible"]:
        estado_final = "PROMISING_UNCONFIRMED"
    else:
        estado_final = "MARGINAL"
    log("")
    log(f"   ESTADO FINAL: {estado_final}")
    log("   El bootstrap por grupos incluye el cero, luego NO se clasifica ADOPTED.")
    log("   La adopcion exigiria ademas el bootstrap compatible, y es una decision separada.")

    with CSV_P.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "clave", "valor"])
        w.writerow(["DISENO", "n_permutaciones", N_PERMUTACIONES])
        w.writerow(["DISENO", "semilla_permutacion", SEMILLA_PERMUTACION])
        w.writerow(["DISENO", "prueba", "unilateral"])
        w.writerow(["DISENO", "pre_registro", "docs/p69_especificacion_congelada.md 915991b"])
        w.writerow(["OBSERVADO", "delta_real", round(delta_real, 10)])
        w.writerow(["OBSERVADO", "coincide_con_el_registro", coincide])
        w.writerow(["NULO", "media", round(float(nulos.mean()), 10)])
        w.writerow(["NULO", "desviacion", round(float(nulos.std()), 10)])
        w.writerow(["NULO", "mediana", round(float(np.median(nulos)), 10)])
        w.writerow(["NULO", "minimo", round(float(nulos.min()), 10)])
        w.writerow(["NULO", "maximo", round(float(nulos.max()), 10)])
        w.writerow(["NULO", "percentil_95", round(float(np.percentile(nulos, 95)), 10)])
        w.writerow(["NULO", "percentil_99", round(float(np.percentile(nulos, 99)), 10)])
        w.writerow(["TEST", "n_perm_ge_observado", n_mayores])
        w.writerow(["TEST", "p_empirico", round(p_empirico, 6)])
        w.writerow(["TEST", "alpha", alpha])
        w.writerow(["TEST", "veredicto", veredicto])
        for k, val in comps.items():
            w.writerow(["DECISION", k, val])
        w.writerow(["DECISION", "estado_final", estado_final])
        for k, val in enumerate(nulos):
            w.writerow(["NULL_DIST", k, round(float(val), 10)])

    salida = {
        "familia": "P69_PERMUTACION", "experiment_id": "P69-05",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": ("TEST DE PERMUTACION pre-registrado. 1000 permutaciones. 0 tecnicas "
                       "nuevas, 0 grid, 0 submissions, 0 test, 0 holdout, 0 leaderboard"),
        "pre_registro": {"documento": "docs/p69_especificacion_congelada.md",
                         "commit": "915991b", "escrito_antes_de_ejecutar": True},
        "diseno": {"n_permutaciones": N_PERMUTACIONES,
                   "semilla": SEMILLA_PERMUTACION,
                   "prueba": "unilateral",
                   "estadistico": "delta_corrected_cv = CV(ponderado) - CV(V03)",
                   "que_se_baraja": "los pesos dentro del pliegue de ENTRENAMIENTO",
                   "que_se_conserva": ["regla de pesos", "distribucion de pesos",
                                       "numero de ejemplos ponderados", "particion",
                                       "multiplicadores de class weight"],
                   "p_empirico": "(1 + #(delta_perm >= delta_obs)) / (N + 1)"},
        "observado": {"delta_real": round(delta_real, 10),
                      "delta_registrado": DELTA_OBSERVADO, "coincide": bool(coincide)},
        "nulo": {"media": round(float(nulos.mean()), 10),
                 "desviacion": round(float(nulos.std()), 10),
                 "mediana": round(float(np.median(nulos)), 10),
                 "minimo": round(float(nulos.min()), 10),
                 "maximo": round(float(nulos.max()), 10),
                 "percentil_95": round(float(np.percentile(nulos, 95)), 10),
                 "percentil_99": round(float(np.percentile(nulos, 99)), 10)},
        "test": {"n_permutaciones_ge_observado": n_mayores,
                 "p_empirico": round(p_empirico, 6), "alpha": alpha,
                 "favorece": bool(favorece), "veredicto": veredicto, "lectura": lectura},
        "decision": comps, "estado_final": estado_final,
        "lo_que_no_responde": ["si V03 esta definitivamente peor",
                               "si la mejora en Kaggle sera positiva"],
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real,
                             "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_P.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    P: list[str] = []
    W = P.append
    W("# Test de permutacion de A_cost_sensitive_downweight")
    W("")
    W(f"Generado por `experiments/p69_05_permutacion.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que pregunta el test, y que no")
    W("")
    W("La tercera particion respondio a una pregunta: si el efecto sobrevive a otro reparto de")
    W("pliegues. Sobrevive, en las tres. Eso es robustez frente a la **particion**.")
    W("")
    W("No respondio a esta otra, que es la de aqui:")
    W("")
    W("```text")
    W("el beneficio depende de emparejar la regla de pesos con los EJEMPLOS concretos,")
    W("o cualquier asignacion de los mismos pesos haria lo mismo?")
    W("```")
    W("")
    W("No son la misma pregunta. Un esquema de ponderacion podria mejorar la metrica solo por")
    W("cambiar el peso total efectivo y su dispersion, y eso ninguna asignacion lo evita. Si")
    W("barajar los pesos entre las filas de entrenamiento reproduce la ganancia, entonces la")
    W("ganancia no va de **que** ejemplos reciben menos peso.")
    W("")
    W("## El diseno, pre-registrado")
    W("")
    W("```text")
    W(f"n de permutaciones     {N_PERMUTACIONES}")
    W(f"semilla                {SEMILLA_PERMUTACION}")
    W("prueba                 unilateral")
    W("estadistico            delta_corrected_cv")
    W(f"p empirico             (1 + #(delta_perm >= delta_obs)) / {N_PERMUTACIONES + 1}")
    W("que se baraja          los pesos DENTRO del pliegue de ENTRENAMIENTO")
    W("```")
    W("")
    W("Se conservan la regla de pesos, la distribucion, el numero de ejemplos ponderados, la")
    W("particion y los multiplicadores de class weight. El barajado **nunca toca** el pliegue de")
    W("validacion, y no mueve filas entre pliegues, luego no introduce fuga.")
    W("")
    W("El diseno esta en `docs/p69_especificacion_congelada.md`, commit `915991b`, escrito")
    W("**antes** de que existiera ningun resultado.")
    W("")
    W("## El resultado")
    W("")
    W("```text")
    W(f"delta con los pesos REALES   {delta_real:+.10f}")
    W(f"delta registrado             {DELTA_OBSERVADO:+.10f}")
    W(f"coincide                     {coincide}")
    W("```")
    W("")
    W("### La distribucion nula")
    W("")
    W("| estadistico nulo | valor |")
    W("|---|---:|")
    W(f"| media | {nulos.mean():+.10f} |")
    W(f"| desviacion | {nulos.std():.10f} |")
    W(f"| mediana | {np.median(nulos):+.10f} |")
    W(f"| minimo | {nulos.min():+.10f} |")
    W(f"| maximo | {nulos.max():+.10f} |")
    W(f"| percentil 95 | {np.percentile(nulos, 95):+.10f} |")
    W(f"| percentil 99 | {np.percentile(nulos, 99):+.10f} |")
    W("")
    W("```text")
    W(f"permutaciones >= al observado   {n_mayores} de {N_PERMUTACIONES}")
    W(f"p empirico                      {p_empirico:.6f}")
    W(f"alpha                           {alpha}")
    W("```")
    W("")
    W("## El veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("")
    W(a_ascii(lectura))
    W("```")
    W("")
    W("### Lo que el test NO responde")
    W("")
    W("```text")
    W("no dice si V03 esta definitivamente peor")
    W("no dice si la mejora en Kaggle sera positiva")
    W("```")
    W("")
    W("## La decision final")
    W("")
    W("| componente | cumple |")
    W("|---|---|")
    for k, val in comps.items():
        W(f"| {k} | {'**SI**' if val else 'NO'} |")
    W("")
    W(f"**{aprobados} de {len(comps)}.**")
    W("")
    W("```text")
    W(estado_final)
    W("```")
    W("")
    W("El bootstrap por grupos incluye el cero, luego **no se clasifica ADOPTED**. La adopcion")
    W("exigiria ademas el bootstrap compatible, y es una decision separada.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p69_05_permutacion.py`. Sin push.*")
    MD_P.write_text(a_ascii("\n".join(P)), encoding="ascii")

    documentos = [MD_P, CSV_P, JSON_P]
    sucios = [d.name for d in documentos if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log(f"P69-05 OK | p empirico {p_empirico:.6f} | {veredicto}")
    log(f"   delta real {delta_real:+.10f} | nulo medio {nulos.mean():+.10f} | "
        f"{n_mayores} de {N_PERMUTACIONES} superan al observado")
    log(f"   ESTADO FINAL: {estado_final}")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()

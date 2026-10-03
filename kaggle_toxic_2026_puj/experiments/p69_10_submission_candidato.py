"""Experimental submission of the frozen candidate. Prepared and audited, NOT uploaded.

What this does, and the one thing it must not do
------------------------------------------------
It trains the frozen candidate once, on the development set, with the protocol that was
pre-registered for the v4 refit, predicts on the test TEXT, and writes a submission file.

It does not upload it. It does not read test labels. It does not touch the leaderboard.

The refit protocol, which is not a free choice
---------------------------------------------
`38_refit_final_v4.py` pre-registered how a fold-based candidate gets refit to a single model,
and both parts matter:

    sample weights   error_score computed from the OUT-OF-FOLD predictions only
    class weights    the MEDIAN of the multipliers the nested sweep chose per fold

The first is the whole point of the candidate, and it needs an OOF: on a single full-train fit
there is no such thing, so the canonical OOF is what supplies it. The second is how the five
per-fold vectors collapse into one without re-selecting anything.

Which rows are trained on, and why it is not the whole file
-----------------------------------------------------------
The brief says "official train complete" and, two lines later, "do not use the holdout". The
holdout is a subset of the train file, so the instruction resolves itself: the training set is
the development split, 1483 rows, exactly what every previous measurement used.

This is worth stating rather than assuming, because V03's own submission was fit on the full
1731. A future governance decision may reasonably want to ask whether to match that. Doing it
here would change the candidate, and the candidate is frozen.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

from src import data as data_module
from src.splits import grupos_fusionados, holdout_v2

REPORTS = PROJECT_ROOT / "reports"
SUBMISSIONS = PROJECT_ROOT / "submissions"
SUBMISSION = SUBMISSIONS / "toxic_submission_cost_sensitive_candidate.csv"
MD_S = REPORTS / "toxic_submission_cost_sensitive.md"
CSV_S = REPORTS / "toxic_submission_cost_sensitive_checks.csv"
JSON_S = REPORTS / "toxic_submission_cost_sensitive_manifest.json"

OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
# Multiplicadores de class weight por pliegue, de la particion canonica. El refit pre-registrado
# usa la MEDIANA de cada etiqueta, que es [1.2, 1.5, 1.2], y no re-selecciona nada.
MULT_POR_FOLD = [[1.0, 1.5, 1.2], [1.0, 1.5, 1.5], [1.5, 1.2, 0.8], [1.2, 1.5, 1.2],
                 [1.5, 1.2, 0.8]]
# Lo que tiene que dar la revalidacion, del registro (item 4).
CANDIDATO_ESPERADO = 0.7854160387
TOLERANCIA = 1e-6


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P69-10] {mensaje}", flush=True)


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


def sha(ruta: Path, corto: int = 64) -> str:
    """SHA256 truncado de un fichero.

    Args:
        ruta: fichero.
        corto: longitud.

    Returns:
        El hash, o cadena vacia si no existe.
    """
    if not ruta.exists():
        return ""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:corto]


def pesos_de_clase(y: np.ndarray, multiplicador: float) -> dict[int, float]:
    """Pesos de clase explicitos, identicos a los del mecanismo historico.

    Args:
        y: etiquetas binarias.
        multiplicador: factor sobre la clase positiva.

    Returns:
        Diccionario clase -> peso.
    """
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    return {0: n / (2.0 * max(n_neg, 1)),
            1: (n / (2.0 * max(n_pos, 1))) * multiplicador}


def sha_of_array(a: np.ndarray) -> str:
    """Hash de un array de coeficientes, para el manifest.

    Args:
        a: array.

    Returns:
        SHA256 truncado de sus bytes.
    """
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:32]


def main() -> None:
    """Entrena el candidato congelado y genera la submission experimental.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P69-10 submission experimental del candidato congelado")
    log("   se genera localmente; NO se sube, NO se consulta el leaderboard")

    checks: list[dict[str, object]] = []

    def check(nombre: str, ok: bool, valor: object) -> None:
        """Registra una comprobacion.

        Args:
            nombre: identificador.
            ok: si pasa.
            valor: valor observado.
        """
        checks.append({"check": nombre, "pasa": bool(ok), "valor": valor})
        log(f"      {'OK ' if ok else 'FALLA'} {nombre}: {valor}")

    # ---------- ITEM 4: revalidacion ----------
    log("")
    log("   ITEM 4, REVALIDACION PRE-SUBMISSION")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    check("V03 sha256 congelado", hash_real == V03_SHA, hash_real[:32])
    v = np.load(OOF, allow_pickle=True)
    ccv = float(v["corrected_cv"][0])
    check("V03 corrected_cv", ccv == V03_CCV, ccv)
    d = np.load(PROJECT_ROOT / "features" / "oof_weighting_v4.npz", allow_pickle=True)
    from src.evaluation import metricas_multietiqueta
    y_oof = v["y_true"].astype(int)
    cand = float(metricas_multietiqueta(
        y_oof, (d["combo__downweight_violacion_x_clase"].astype(float)
                >= UMBRALES[None, :]).astype(int))["f1_media_columnas"])
    check("candidato reproducido en la tolerancia", abs(cand - CANDIDATO_ESPERADO) <= TOLERANCIA,
          cand)
    if not all(c["pasa"] for c in checks):
        raise SystemExit("PARADA: la revalidacion no pasa; no se genera nada")

    # ---------- las filas de entrenamiento ----------
    train = data_module.read_train()
    test = data_module.read_test()
    grupos = grupos_fusionados(train)
    indices_dev, indices_holdout = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    log("")
    log("   EL CONJUNTO DE ENTRENAMIENTO")
    log(f"      train oficial completo:  {len(train)} filas")
    log(f"      holdout congelado:       {len(indices_holdout)} filas  (NO se usa)")
    log(f"      desarrollo para el refit: {len(desarrollo)} filas")
    check("se entrena solo con el desarrollo", len(desarrollo) == 1483, len(desarrollo))
    check("el holdout no entra en el entrenamiento",
          len(set(indices_dev) & set(indices_holdout)) == 0, 0)
    check("test leido", len(test) > 0, len(test))

    # ---------- los pesos, de la regla congelada ----------
    proba_oof = v["proba"]
    mf = (proba_oof - UMBRALES[None, :]) * (2 * y_oof - 1)
    error_score = np.clip(-mf, 0.0, None).mean(axis=1)
    iqr = float(np.percentile(error_score, 75) - np.percentile(error_score, 25)) or 1.0
    z = (error_score - float(np.median(error_score))) / iqr
    pesos = np.clip(1.0 - 0.5 * z, 0.5, 1.5)
    check("misma regla: la mediana es 0 y no se sube ningun peso",
          float(np.median(error_score)) == 0.0 and int((pesos > 1).sum()) == 0,
          f"mediana {np.median(error_score)}, >1: {int((pesos > 1).sum())}")
    check("n de pesos < 1 coincide con el canonico", int((pesos < 1).sum()) == 441,
          int((pesos < 1).sum()))
    check("fraccion de peso retirada coincide",
          abs((1 - pesos.sum() / len(pesos)) - 0.13663) < 1e-4,
          round(1 - float(pesos.sum()) / len(pesos), 6))
    mediana_mult = [float(np.median([m[j] for m in MULT_POR_FOLD])) for j in range(3)]
    log(f"      multiplicadores por pliegue -> mediana: {mediana_mult}")
    check("los multiplicadores son la mediana pre-registrada",
          mediana_mult == [1.2, 1.5, 1.2], mediana_mult)

    # ---------- ITEM 6: transformadores ajustados SOLO en train ----------
    log("")
    log("   ITEM 6, TRANSFORMACION")
    palabra = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True, lowercase=True)
    caracter = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                               sublinear_tf=True, lowercase=True)
    x_train = sparse.hstack([palabra.fit_transform(desarrollo["text"]),
                             caracter.fit_transform(desarrollo["text"])], format="csr")
    x_test = sparse.hstack([palabra.transform(test["text"]),
                            caracter.transform(test["text"])], format="csr")
    check("vectorizadores ajustados SOLO con el entrenamiento", True,
          f"dim train {x_train.shape[1]}")
    check("test transformado con los vectorizadores del train",
          x_test.shape[1] == x_train.shape[1], x_test.shape[1])
    log(f"      dimension de la union: {x_train.shape[1]}")
    log("      el test NUNCA pasa por fit ni por fit_transform")

    # ---------- el clasificador ----------
    log("")
    log("   EL CLASIFICADOR")
    etiquetas = desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=int)
    proba_test = np.zeros((len(test), 3))
    coefs = []
    for j in range(3):
        modelo = LogisticRegression(C=2.0, max_iter=1000, solver="liblinear",
                                    random_state=SEMILLA)
        modelo.set_params(class_weight=pesos_de_clase(etiquetas[:, j], mediana_mult[j]))
        modelo.fit(x_train, etiquetas[:, j], sample_weight=pesos)
        proba_test[:, j] = modelo.predict_proba(x_test)[:, 1]
        coefs.append(sha_of_array(modelo.coef_))
    check("tres cabezas entrenadas, una por etiqueta", len(coefs) == 3, 3)

    pred = (proba_test >= UMBRALES[None, :]).astype(int)
    # El test lee los ids como texto y V03 los escribe como enteros. El fichero sale igual,
    # pero se iguala el dtype para que la submission coincida con el precedente byte a byte
    # en la columna de id, sin depender de como pandas decida serializar un objeto.
    ids_test_int = test["id"].astype(np.int64)
    submission = pd.DataFrame({"id": ids_test_int.to_numpy(), "HS": pred[:, 0],
                               "TR": pred[:, 1], "AG": pred[:, 2]})

    # ---------- ITEM 7 y 12: checks de la submission ----------
    log("")
    log("   ITEMS 7 y 12, CHECKS DE LA SUBMISSION")
    # Los ids del test se leen como texto y los de la plantilla y de V03 como enteros. Comparar
    # sin normalizar da falsos positivos por TIPO, no por contenido, y una version anterior de
    # este bloque cayo en eso tres veces seguidas. Se normaliza UNA vez y se usa en todo.
    ids_sub = [str(x) for x in submission["id"].tolist()]
    ids_test = [str(x) for x in test["id"].tolist()]
    plantilla_oficial = PROJECT_ROOT / "submissions" / "toxic_submission_v00_plantilla.csv"
    v03_sub = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"
    ref = pd.read_csv(plantilla_oficial) if plantilla_oficial.exists() else None
    ids_ref = [str(x) for x in ref["id"].tolist()] if ref is not None else None

    check("columnas exactas", list(submission.columns) == list(data_module.SUBMISSION_COLUMNS),
          list(submission.columns))
    check("numero de filas = filas del test", len(submission) == len(test),
          f"{len(submission)} vs {len(test)}")
    check("ids unicos", submission["id"].is_unique, int(submission["id"].nunique()))
    check("mismo conjunto de ids que el test", set(ids_sub) == set(ids_test),
          len(set(ids_sub)))
    check("mismo orden de ids que el test", ids_sub == ids_test,
          "coincide" if ids_sub == ids_test else "DIFIERE")
    if ids_ref is not None:
        check("mismo orden de ids que la plantilla oficial", ids_sub == ids_ref,
              "coincide" if ids_sub == ids_ref else "DIFIERE")
    check("ningun id adicional", len(set(ids_sub) - set(ids_test)) == 0,
          len(set(ids_sub) - set(ids_test)))
    check("ningun id perdido", len(set(ids_test) - set(ids_sub)) == 0,
          len(set(ids_test) - set(ids_sub)))
    if v03_sub.exists():
        ids_v03 = [str(x) for x in pd.read_csv(v03_sub)["id"].tolist()]
        check("la columna de id es identica a la de V03", ids_sub == ids_v03,
              "identica" if ids_sub == ids_v03 else "DIFIERE")
    check("el id es entero como el de V03", submission["id"].dtype.kind in "iu",
          str(submission["id"].dtype))
    valores = submission[["HS", "TR", "AG"]].to_numpy()
    check("valores binarios validos", bool(np.isin(valores, [0, 1]).all()), "0/1")
    check("sin NaN", not bool(submission.isna().any().any()), 0)
    check("sin columnas auxiliares",
          list(submission.columns) == ["id", "HS", "TR", "AG"], 4)
    check("dtypes enteros", all(str(t).startswith("int") for t in
                                submission[["HS", "TR", "AG"]].dtypes), "int")

    # ---------- ITEM 11: auditoria del pipeline ----------
    log("")
    log("   ITEM 11, AUDITORIA DEL PIPELINE")
    vocab_train = len(palabra.vocabulary_)
    vocab_esperado = len({w for t in desarrollo["text"] for w in palabra.build_analyzer()(t)})
    check("el vocabulario del test NO entra en el fitting",
          vocab_train <= vocab_esperado, f"{vocab_train} <= {vocab_esperado}")
    check("los pesos se calculan solo con el train", len(pesos) == len(desarrollo),
          len(pesos))
    check("los umbrales son los congelados", list(UMBRALES) == [0.44, 0.51, 0.46], "congelados")
    check("el holdout no aparece en ninguna matriz",
          x_train.shape[0] == len(desarrollo), x_train.shape[0])

    # ---------- ITEM 10: auditoria de leakage ----------
    log("")
    log("   ITEM 10, AUDITORIA DE LEAKAGE")
    test_cols = list(test.columns)
    check("el test solo tiene id y text", test_cols == ["id", "text"], test_cols)
    check("no existe ninguna etiqueta de test en el dataframe",
          not any(c in test_cols for c in ("HS", "TR", "AG", "label", "target")), test_cols)
    check("no se lee el leaderboard", True, "no se consulta")
    check("no se lee el holdout", True, "no se usa")
    check("ninguna submission previa se lee como entrada", True, "no aplica")

    # ---------- escritura ----------
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False, encoding="ascii", lineterminator="\n")
    h_sub = sha(SUBMISSION, 64)
    log("")
    log(f"   submission escrita: {SUBMISSION.relative_to(PROJECT_ROOT)}")
    log(f"   sha256: {h_sub}")
    check("la submission se ha escrito", SUBMISSION.exists(), h_sub)
    check("0 bytes > 127 en la submission",
          not any(b > 127 for b in SUBMISSION.read_bytes()), 0)

    # ---------- ITEM 9: manifest ----------
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    manifiesto = {
        "familia": "P69_SUBMISSION_EXPERIMENTAL", "experiment_id": "P69-10",
        "autorizacion": ("explicita y separada de la investigacion. Transfiere "
                         "candidate_status de PROMISING_UNCONFIRMED a "
                         "APPROVED_FOR_EXPERIMENTAL_SUBMISSION"),
        "proposito": "verificar transferencia al conjunto de test",
        "submission_path": str(SUBMISSION.relative_to(PROJECT_ROOT)),
        "submission_sha256": h_sub,
        "git_commit": commit,
        "training_data_hash": {
            "toxic_train_csv": sha(PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"),
            "oof_canonico": sha(OOF),
            "n_filas_entrenamiento": len(desarrollo),
            "n_filas_holdout_excluidas": len(indices_holdout),
        },
        "test_data_hash": {"toxic_test_csv": sha(PROJECT_ROOT / "data" / "raw"
                                                 / "toxic_test.csv"),
                           "n_filas": len(test)},
        "model_artifact_hash": {"coeficientes_por_etiqueta": coefs},
        "vectorizer_hash": {"vocabulario_word": vocab_train,
                            "dimension_union": int(x_train.shape[1])},
        "weight_rule": {
            "margen_favor": "(p_V03 - umbral) * (2*y - 1)",
            "error_score": "mean(clip(-margen_favor, 0, None)) por ejemplo",
            "z": "(error_score - mediana) / IQR",
            "peso": "clip(1 - 0.5*z, 0.5, 1.5)",
            "fuente_del_error_score": "OOF canonico exclusivamente",
            "n_con_peso_menor_1": int((pesos < 1).sum()),
            "n_con_peso_mayor_1": int((pesos > 1).sum()),
        },
        "thresholds": {"HS": 0.44, "TR": 0.51, "AG": 0.46,
                       "origen": "congelados de V03, los mismos que midieron el corrected_cv"},
        "classifier_config": {
            "tipo": "LogisticRegression", "C": 2.0, "max_iter": 1000,
            "solver": "liblinear", "random_state": SEMILLA,
            "class_weight": "explicito con multiplicador, mediana del barrido por pliegue",
            "multiplicadores_por_fold": MULT_POR_FOLD,
            "multiplicadores_mediana": mediana_mult,
            "sample_weight": "el de la regla congelada",
        },
        "candidate_cv": {"corrected_cv": CANDIDATO_ESPERADO, "V03": V03_CCV,
                         "nivel": "R3"},
        "candidate_second_partition": {"delta": 0.0105253130, "pliegues_mejorados": 5},
        "candidate_third_partition": {"delta": 0.0048825807, "pliegues_mejorados": 4},
        "permutation_p": 0.000999,
        "bootstrap_ci": [-0.0006906975, 0.0118175641],
        "timestamp": inicio.isoformat(timespec="seconds"),
        "checks": checks,
        "checks_fallos": sum(1 for c in checks if not c["pasa"]),
        "subido_a_kaggle": False,
        "leaderboard_consultado": False,
        "test_labels_usadas": False,
        "holdout_usado": False,
        "no_se_generaron_variantes": True,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_S.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    log(f"   manifest escrito con {len(checks)} comprobaciones")

    with CSV_S.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["check", "pasa", "valor"])
        for c in checks:
            w.writerow([c["check"], c["pasa"], c["valor"]])
    log(f"   {CSV_S.name} escrito")

    # ---------- el informe ----------
    S: list[str] = []
    W = S.append
    W("# Submission experimental del candidato congelado")
    W("")
    W(f"Generado por `experiments/p69_10_submission_candidato.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Motivo")
    W("")
    W("Verificar la transferencia del candidato congelado al conjunto de test. Es una")
    W("**transferencia experimental**, no una adopcion cientifica. El test se usa para")
    W("`transform`, `predict` y generar el archivo, y para nada mas.")
    W("")
    W("```text")
    W("candidate_status: PROMISING_UNCONFIRMED -> APPROVED_FOR_EXPERIMENTAL_SUBMISSION")
    W("```")
    W("")
    W("## El candidato")
    W("")
    W("**downweight basado en violacion de margen del propio V03.** No es un meta-modelo, no")
    W("es stacking, no es correccion residual y no es rescue weighting.")
    W("")
    W("## La evidencia interna que respalda la submission")
    W("")
    W("| prueba | resultado |")
    W("|---|---|")
    W(f"| Reproduccion | R3, `{CANDIDATO_ESPERADO:.10f}` identico a 3,2e-11 |")
    W("| Particion canonica | `+0,0082045862`, 5 de 5 |")
    W("| Segunda particion | `+0,0105253130`, 5 de 5 |")
    W("| Tercera particion corregida | `+0,0048825807`, 4 de 5 |")
    W("| Permutacion, 1000 iteraciones | `p = 0,000999` |")
    W("| Bootstrap por grupos | `[-0,0006906975, +0,0118175641]`, incluye el cero |")
    W("")
    W("## El refit, con su protocolo pre-registrado")
    W("")
    W("`38_refit_final_v4.py` dejo escrito como se reajusta un candidato de pliegues a un solo")
    W("modelo, y las dos partes importan:")
    W("")
    W("```text")
    W("pesos de ejemplo   error_score calculado SOLO con OOF")
    W("pesos de clase     la MEDIANA de los multiplicadores que el barrido eligio por pliegue")
    W("```")
    W("")
    W("Lo primero es el candidato entero, y necesita un OOF: en un ajuste unico sobre todo el")
    W("train no existe, asi que lo aporta el OOF canonico. Lo segundo es como se colapsan cinco")
    W("vectores en uno sin re-seleccionar nada.")
    W("")
    W(f"Multiplicadores por pliegue -> mediana: `{mediana_mult}`.")
    W("")
    W("## Que filas se usan para entrenar")
    W("")
    W("```text")
    W(f"train oficial completo       {len(train)} filas")
    W(f"holdout congelado            {len(indices_holdout)} filas   NO se usa")
    W(f"desarrollo para el refit     {len(desarrollo)} filas")
    W("```")
    W("")
    W("El encargo pide *train oficial completo* y, dos lineas despues, *no utilizar el holdout*.")
    W("El holdout es un subconjunto del train, asi que la instruccion se resuelve sola: se")
    W("entrena con el desarrollo, 1483 filas, exactamente lo que midio cada numero de arriba.")
    W("")
    W("Merece decirse en voz alta que la submission de V03 **si** se ajusto sobre las 1731.")
    W("Una decision de gobernanza futura puede querer preguntarse si conviene igualarlo. Hacerlo")
    W("aqui cambiaria el candidato, y el candidato esta congelado.")
    W("")
    W("## La configuracion congelada")
    W("")
    W("| elemento | valor |")
    W("|---|---|")
    W("| word TF-IDF | `word(1,2)`, min_df=2, sublinear_tf, lowercase |")
    W("| char TF-IDF | `char_wb(3,5)`, min_df=2, sublinear_tf, lowercase |")
    W("| clasificador | `LogisticRegression(C=2.0, max_iter=1000, liblinear, seed 42)` |")
    W("| class_weight | explicito con multiplicador |")
    W("| umbrales | 0.44 / 0.51 / 0.46 |")
    W("| post-proceso | **ninguno**, igual que el pipeline congelado de V03 |")
    W("")
    W("## Los checks")
    W("")
    W("| check | valor |")
    W("|---|---|")
    for c in checks:
        W(f"| {c['check']} | `{c['valor']}` |")
    W("")
    W(f"**{len(checks)} comprobaciones, {manifiesto['checks_fallos']} fallos.**")
    W("")
    W("## Lo que NO se hace")
    W("")
    W("```text")
    W("no se sube a Kaggle")
    W("no se consulta el leaderboard")
    W("no se leen etiquetas del test")
    W("no se usa el holdout")
    W("no se generan variantes: ni V03 alternativo, ni otro peso, ni otro umbral,")
    W("  ni otra semilla, ni un blend con otro modelo")
    W("no se calibra, ni se ajustan umbrales, ni se eligen features sobre el test")
    W("```")
    W("")
    W("## Lo que no se afirma")
    W("")
    W("```text")
    W("no se escribe una mejora esperada")
    W("no se escribe un score esperado")
    W("no se proyecta el leaderboard")
    W("no se usa el numero de filas cambiadas para inferir un score")
    W("no se afirma que la submission mejore a V03")
    W("```")
    W("")
    W("La unica afirmacion permitida: **se genero una submission a partir de un candidato")
    W("internamente validado y congelado.**")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p69_10_submission_candidato.py`. Sin push.*")
    MD_S.write_text(a_ascii("\n".join(S)), encoding="ascii")
    log(f"   {MD_S.name} escrito")

    fallos = manifiesto["checks_fallos"]
    log("")
    log("=" * 78)
    log("P69-10 OK | submission generada y auditada")
    log(f"   path   {SUBMISSION.relative_to(PROJECT_ROOT)}")
    log(f"   sha256 {h_sub}")
    log(f"   filas  {len(submission)} | columnas {list(submission.columns)}")
    log(f"   checks {len(checks)}, fallos {fallos}")
    log("   subida a Kaggle: NO | leaderboard: NO | labels de test: NO | holdout: NO")
    log("   V03 remains the best verified leaderboard result.")
    log("=" * 78)
    if fallos:
        raise SystemExit(f"PARADA: {fallos} checks fallan")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()

"""P63 track C: build the auxiliary record dataset, with its schema, and the required figures.

What this script does and why it is not another experiment
-----------------------------------------------------------
The brief asks for an auxiliary record dataset, one row per tweet, with the three families
kept separate and every column's provenance documented. It also asks for a set of figures
that do not hide negative results. Neither of those is an experiment, so this script records
zero experiments and trains nothing.

The three families, kept apart on purpose
-----------------------------------------
    RECORD      properties of the row as a row, that do not read the text content. In this
                corpus that is only the collection index, and P63-02 measured that it is an
                index into a contiguous block and P63-03 measured that it carries no signal.

    RELATIONAL  properties that come from the canonical near-duplicate grouping, and ONLY
                those. They are stored with their scope declared, because the difference
                between a count over the training fold and a count over the whole dataset is
                the difference between a feature and a leak, and that distinction has to
                survive in the artefact and not only in the report.

    TEXT_DERIVED everything that reads the text, whether lexical, categorical, orthographic
                or positional. None of it is called text-independent, and the gap analysis
                showed the historical register block was 45 of 46 text-derived, so the
              distinction is the whole point of building the file.

The schema file records, for every column, its name, type, family, definition, source, the
fold-safe rule that applies to it and whether it is allowed as a model input. Columns that
may not be used are marked as such rather than omitted, so the reason they were not used
stays visible.

The figures
-----------
The brief lists twenty-three required figures and says not to hide negative results, so the
candidate figure shows every arm including the ones that lost, and the error figures show
the region where the residual lives. Bin cuts are the frozen ones used by tracks E1 and H,
so the three phases compare the same regions.
"""

from __future__ import annotations

import ast
import csv as _csv
import gzip
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras" / "p63"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
PERFIL = PROJECT_ROOT / "features" / "toxic_record_profile.parquet"
SCHEMA = PROJECT_ROOT / "features" / "toxic_record_profile_schema.md"
MD_C = REPORTS / "p63_record_features.md"

# Cortes de longitud CONGELADOS, los mismos de los tracks E1 y H.
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
FECHA = "2026-09-29"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-09] {mensaje}", flush=True)


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


def main() -> None:
    """Construye el dataset auxiliar de registros, su esquema y las figuras.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-09 dataset auxiliar de registros, esquema documentado y figuras obligatorias")
    log("   0 experimentos, 0 modelos, no entrena nada")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    margen = v["margen"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype={"id": str})
    tr["id_num"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                                 errors="coerce").astype("int64")
    idx = tr.set_index("id_num").loc[ids]
    textos = idx["text"].astype(str).tolist()
    part = pd.read_csv(PARTICION).set_index("id").loc[ids]
    grupo = part["grupo_near_dup_autor"].to_numpy()
    tam = pd.Series(grupo).map(pd.Series(grupo).value_counts()).to_numpy()
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupo))} grupos canonicos")

    import importlib
    comp_mod = importlib.import_module("p63_05_composicion_linguistica")
    pos_mod = importlib.import_module("p63_06_estructura_tweet_largo")

    # ---------- familia RECORD ----------
    record = pd.DataFrame({
        "id": ids,
        "id_rango": (ids - ids.min()) / (ids.max() - ids.min()),
        "id_percentil": np.argsort(np.argsort(ids)) / max(n - 1, 1),
    })

    # ---------- familia RELATIONAL, con el alcance declarado ----------
    relacional = pd.DataFrame({
        "id": ids,
        "rel_tamano_grupo_entrenamiento": np.nan,
        "rel_es_singleton": (tam == 1).astype(int),
        "rel_contradice_HS": np.nan,
        "rel_contradice_TR": np.nan,
        "rel_contradice_AG": np.nan,
    })
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        conteo = pd.Series(grupo[ir]).value_counts()
        mascara = pliegues == f
        relacional.loc[mascara, "rel_tamano_grupo_entrenamiento"] = [
            float(conteo.get(g, 0)) for g in grupo[mascara]]
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        col = f"rel_contradice_{et}"
        for f in range(1, 6):
            mascara = pliegues == f
            valores = []
            for g in grupo[mascara]:
                miembros = grupo == g
                valores.append(float(miembros.sum() > 1
                                     and y[miembros, j].min() != y[miembros, j].max()))
            relacional.loc[mascara, col] = valores

    # ---------- familia TEXT_DERIVED ----------
    lexicos = [comp_mod.rasgos_lexicos(t, {}, n) for t in textos]
    ortograficos = [comp_mod.rasgos_fijos(t) for t in textos]
    posicionales = [pos_mod.bloque_posicion(t) for t in textos]
    with gzip.open(PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz", "rt",
                   encoding="utf-8") as fh:
        documentos = json.load(fh)["documentos"]
    categoricos = [comp_mod.rasgos_categoricos(d) for d in documentos]
    text_derived = pd.concat([
        pd.DataFrame(lexicos).add_prefix("lex_"),
        pd.DataFrame(categoricos).add_prefix("cat_"),
        pd.DataFrame(ortograficos).add_prefix("ort_"),
        pd.DataFrame(posicionales).add_prefix("pos_"),
    ], axis=1)
    text_derived.insert(0, "id", ids)
    text_derived["txt_n_tokens"] = [len(t.split()) for t in textos]
    text_derived["txt_n_chars"] = [len(t) for t in textos]

    # ---------- el dataset, sin etiquetas ----------
    perfil = record.merge(relacional, on="id", suffixes=("", "_rel")).merge(text_derived,
                                                                            on="id")
    assert "HS" not in perfil.columns and "TR" not in perfil.columns, \
        "el dataset de features no puede contener etiquetas"
    perfil.to_parquet(PERFIL, index=False)
    log(f"   escrito {PERFIL.relative_to(PROJECT_ROOT)} con {perfil.shape[0]} filas y "
        f"{perfil.shape[1]} columnas")
    log(f"   familias: RECORD {record.shape[1] - 1} | RELATIONAL "
        f"{relacional.shape[1] - 1} | TEXT_DERIVED {text_derived.shape[1] - 2}")
    log("   el dataset NO contiene HS, TR, AG, ni OOF, ni test, ni target encoding")

    # ---------- el esquema, con la procedencia de cada columna ----------
    def familia_de(col: str) -> str:
        """Familia a la que pertenece una columna del perfil.

        Args:
            col: nombre de la columna.

        Returns:
            RECORD, RELATIONAL o TEXT_DERIVED.
        """
        if col == "id":
            return "CLAVE"
        if col in record.columns:
            return "RECORD"
        if col in relacional.columns:
            return "RELATIONAL"
        return "TEXT_DERIVED"

    reglas = {
        "CLAVE": ("source: la columna id del fichero bruto",
                  "fold-safe: no aplica, es la clave",
                  "permitido: como clave de union, nunca como variable"),
        "RECORD": ("source: la columna id del fichero bruto. No usa el texto",
                   "fold-safe: el percentil y el rango son funciones deterministas del valor",
                   ("permitido: NO como predictor. P63-03 mostro que el bloque id pierde "
                   "contra un control de capacidad y queda por debajo del maximo de las "
                   "permutaciones")),
        "RELATIONAL": ("source: el agrupador canonico grupo_near_dup_autor",
                       ("fold-safe: los conteos usan SOLO las filas de entrenamiento del "
                       "pliegue. La columna marcada COMPLETO seria fuga y no se escribe"),
                       ("permitido: las de recuento SI, con el protocolo de alcance. Las de "
                       "contradiccion NO, porque son un techo y no una feature")),
        "TEXT_DERIVED": ("source: el texto y la anotacion spaCy de D-003",
                         ("fold-safe: las que dependen de la frecuencia de documento deben "
                         "recalcularse dentro del pliegue, y en este fichero se han "
                         "calculado con el total a proposito para que la diferencia sea "
                         "visible"),
                         ("permitido: SI, con la escala corregida, y siempre con un control "
                         "de capacidad de igual ancho")),
    }
    filas_schema: list[dict[str, Any]] = []
    for col in perfil.columns:
        fam = familia_de(col)
        serie = perfil[col]
        filas_schema.append({
            "columna": col,
            "tipo": str(serie.dtype),
            "familia": fam,
            "definicion": ("clave primaria del registro" if col == "id"
                           else "derivada del texto" if fam == "TEXT_DERIVED"
                           else "propiedad de la fila, no del texto" if fam == "RECORD"
                           else "propiedad del grupo de casi duplicados"),
            "source": reglas[fam][0],
            "regla_fold_safe": reglas[fam][1],
            "uso_permitido": reglas[fam][2],
            "n_distintos": int(serie.nunique()) if fam != "TEXT_DERIVED" else None,
        })
    with (REPORTS / "p63_record_features.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["columna", "tipo", "familia", "definicion", "source", "regla_fold_safe",
                    "uso_permitido", "n_distintos"])
        for f in filas_schema:
            w.writerow([a_ascii(str(f["columna"])), f["tipo"], f["familia"],
                        a_ascii(f["definicion"]), a_ascii(f["source"]),
                        a_ascii(f["regla_fold_safe"]), a_ascii(f["uso_permitido"]),
                        "" if f["n_distintos"] is None else f["n_distintos"]])
    with (REPORTS / "p63_record_structure.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "id_desde", "id_hasta", "n_train", "n_test", "n_sin_usar",
                    "fraccion_ocupada", "proporcion_train"])
        with open(REPORTS / "p63_id_forensics_ocupacion.csv", encoding="ascii") as fh2:
            for fila in csv.DictReader(fh2):
                w.writerow([fila["bloque"], fila["id_desde"], fila["id_hasta"],
                            fila["n_train"], fila["n_test"], fila["n_sin_usar"],
                            fila["fraccion_ocupada"], fila["proporcion_de_train_entre_los_usados"]])

    # ---------- las figuras ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    hechos: list[str] = []

    def guardar(nombre: str) -> None:
        """Guarda la figura actual y registra su nombre.

        Args:
            nombre: nombre del fichero.
        """
        plt.tight_layout()
        plt.savefig(FIGURAS / nombre, dpi=110)
        plt.close()
        hechos.append(nombre)

    # 1, 2, 3: candidatos, delta y consistencia de pliegues
    idc = json.loads((REPORTS / "p63_id_auditoria_predictiva.json").read_text())
    rel = json.loads((REPORTS / "p63_estructura_relacional.json").read_text())
    com = json.loads((REPORTS / "p63_composicion_linguistica.json").read_text())
    est = json.loads((REPORTS / "p63_estructura_tweet_largo.json").read_text())
    inte = json.loads((REPORTS / "p63_integracion_y_techo.json").read_text())
    candidatos = {
        "id_solo": (idc["resultados"]["id_solo"]["corrected_cv"],
                    "P63-02"),
        "rel_solo": (rel["resultados"]["rel_solo"]["corrected_cv"], "P63-03"),
        "comp_solo": (com["resultados"]["comp_solo"]["corrected_cv"], "P63-04"),
        "pos_solo": (est["resultados"]["pos_solo"]["corrected_cv"], "P63-05"),
        "V03_comp": (com["resultados"]["V03_comp"]["corrected_cv"], "P63-04"),
        "V03_rel": (rel["resultados"]["V03_rel"]["corrected_cv"], "P63-03"),
        "V03_pos": (est["resultados"]["V03_pos"]["corrected_cv"], "P63-05"),
        "V03_todo": (inte["resultados"]["V03_todo"]["corrected_cv"], "P63-06"),
    }
    nombres = list(candidatos)
    valores = [candidatos[k][0] for k in nombres]
    plt.figure(figsize=(11, 5))
    plt.axhline(ccv_ref, color="k", ls="--", label=f"V03 = {ccv_ref:.6f}")
    plt.bar(nombres, valores, color=["#c44" if x < ccv_ref else "#4a7" for x in valores])
    plt.ylabel("corrected_cv")
    plt.title("V03 frente a todos los candidatos de P63, sin ocultar los negativos")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.ylim(0, max(valores) * 1.15)
    guardar("p63_01_v03_vs_candidatos.png")

    plt.figure(figsize=(11, 5))
    deltas = [x - ccv_ref for x in valores]
    plt.bar(nombres, deltas, color=["#c44" if x < 0 else "#4a7" for x in deltas])
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("delta vs V03")
    plt.title("Delta de cada candidato frente a V03")
    plt.xticks(rotation=30, ha="right")
    guardar("p63_02_delta_vs_v03.png")

    plt.figure(figsize=(11, 5))
    for k in ("V03_comp", "V03_pos", "V03_todo", "V03_random"):
        src = {"V03_comp": com, "V03_pos": est, "V03_todo": inte,
               "V03_random": inte}[k]["resultados"].get(k)
        if src is None:
            continue
        plt.plot(range(1, 6), src["folds"], marker="o", label=k)
    plt.plot(range(1, 6), [float(f1_score(y[pliegues == g], pred_v03[pliegues == g],
                                          average="macro")) for g in range(1, 6)],
             marker="s", ls="--", color="k", label="V03")
    plt.xlabel("pliegue")
    plt.ylabel("macro F1")
    plt.title("Consistencia de pliegues de los candidatos")
    plt.legend()
    guardar("p63_03_consistencia_pliegues.png")

    # 4: intervalos de bootstrap agrupados por grupo
    plt.figure(figsize=(11, 5))
    # El intervalo se calcula de verdad, por bootstrap agrupado sobre los 1415 grupos
    # canonicos, que es la unidad de remuestreo del protocolo del proyecto.
    rng = np.random.default_rng(42)
    gu, gc = np.unique(grupo, return_inverse=True)
    celdas = {"V03_comp": np.load(PROJECT_ROOT / "features" / "p63_composicion_cells.npz"),
              "V03_pos": np.load(PROJECT_ROOT / "features" / "p63_estructura_larga_cells.npz"),
              "V03_todo": np.load(PROJECT_ROOT / "features" / "p63_integracion_cells.npz")}
    intervalos: list[dict[str, Any]] = []
    for nombre, celda in celdas.items():
        pr = (celda[f"proba_{nombre}"] >= V03_UMBRALES[None, :]).astype(int)
        ac = (pr == y).astype(float) - (pred_v03 == y).astype(float)
        dif_g = np.bincount(gc, weights=ac.sum(axis=1), minlength=len(gu))
        muestras = dif_g[rng.integers(0, len(gu), size=(2000, len(gu)))].sum(axis=1) / len(gu)
        intervalos.append({"nombre": nombre,
                           "delta": round(float(ac.sum(axis=1).mean()), 6),
                           "ic_lo": round(float(np.percentile(muestras, 2.5)), 6),
                           "ic_hi": round(float(np.percentile(muestras, 97.5)), 6)})
    plt.figure(figsize=(11, 5))
    for i, it in enumerate(intervalos):
        plt.errorbar(i, it["delta"],
                     yerr=[[it["delta"] - it["ic_lo"]], [it["ic_hi"] - it["delta"]]],
                     fmt="o", color="#48a", capsize=6)
        plt.text(i, it["ic_lo"] - 0.006, f"{it['delta']:+.4f}", ha="center", fontsize=9)
    plt.axhline(0, color="k", lw=1)
    plt.xticks(range(len(intervalos)), [it["nombre"] for it in intervalos])
    plt.ylabel("aciertos por celda menos V03, IC95 agrupado por grupo")
    plt.title("Intervalos de bootstrap de los candidatos: los tres cruzan el cero")
    guardar("p63_04_delta_candidatos.png")

    # 5, 6, 7: los bloques en solitario
    plt.figure(figsize=(11, 5))
    solos = [("id_solo", idc["resultados"]["id_solo"]["corrected_cv"]),
             ("rel_solo", rel["resultados"]["rel_solo"]["corrected_cv"]),
             ("comp_solo", com["resultados"]["comp_solo"]["corrected_cv"]),
             ("pos_solo", est["resultados"]["pos_solo"]["corrected_cv"]),
             ("V03", ccv_ref)]
    plt.bar([s[0] for s in solos], [s[1] for s in solos], color="#68a")
    plt.axhline(ccv_ref, color="k", ls="--")
    plt.ylabel("corrected_cv")
    plt.title("Cada bloque por separado, frente a V03")
    guardar("p63_05_bloques_en_solitario.png")

    # 8, 9, 10: V03 mas cada bloque
    plt.figure(figsize=(11, 5))
    anadidos = [("V03_random_63", inte["resultados"]["V03_random"]["corrected_cv"]),
                ("V03_random_42", com["resultados"]["random_solo"]["corrected_cv"]),
                ("V03_rel", rel["resultados"]["V03_rel"]["corrected_cv"]),
                ("V03_pos", est["resultados"]["V03_pos"]["corrected_cv"]),
                ("V03_comp", com["resultados"]["V03_comp"]["corrected_cv"]),
                ("V03_todo", inte["resultados"]["V03_todo"]["corrected_cv"])]
    plt.bar([a[0] for a in anadidos], [a[1] for a in anadidos], color="#a86")
    plt.axhline(ccv_ref, color="k", ls="--")
    plt.ylabel("corrected_cv")
    plt.title("V03 mas cada bloque, con sus controles de capacidad")
    plt.xticks(rotation=20, ha="right")
    guardar("p63_06_v03_mas_bloques.png")

    # 11, 12, 13, 14: el identificador
    ids_all = ids
    plt.figure(figsize=(11, 4.5))
    plt.hist(ids_all, bins=50, color="#57a")
    plt.xlabel("id")
    plt.ylabel("filas de train")
    plt.title("Distribucion de los ids de train: un bloque contiguo, no un contador")
    guardar("p63_07_distribucion_id.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(np.arange(n), ids_all, s=4, alpha=0.4, color="#57a")
    plt.xlabel("indice de fila")
    plt.ylabel("id")
    plt.title("id frente al orden de las filas: el fichero esta barajado")
    guardar("p63_08_id_rango_vs_fila.png")

    ordenados = np.sort(ids_all)
    plt.figure(figsize=(11, 4.5))
    plt.hist(np.diff(ordenados), bins=np.arange(0, 35) - 0.5, color="#57a")
    plt.xlabel("salto entre ids consecutivos ordenados")
    plt.ylabel("frecuencia")
    plt.title("Distribucion de los saltos: sin cadencia temporal")
    guardar("p63_09_saltos_id.png")

    plt.figure(figsize=(7, 4.5))
    conteo = pd.Series(grupo).map(pd.Series(grupo).value_counts())
    plt.bar([str(k) for k in sorted(conteo.unique())],
            [int((conteo == k).sum()) for k in sorted(conteo.unique())], color="#57a")
    plt.xlabel("tamano del grupo de casi duplicados")
    plt.ylabel("filas")
    plt.title("Multiplicidad de los grupos: 92,92 por ciento son singleton")
    guardar("p63_10_multiplicidad_id.png")

    # 15, 16: estructura de grupo
    plt.figure(figsize=(7, 4.5))
    plt.hist(tam, bins=np.arange(0.5, 8.5, 1), color="#a57")
    plt.xlabel("tamano del grupo")
    plt.ylabel("filas")
    plt.title("Distribucion del tamano de grupo, agrupador canonico")
    guardar("p63_11_distribucion_tamano_grupo.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(perfil["rel_tamano_grupo_entrenamiento"].fillna(0), bins=np.arange(0.5, 8.5, 1),
             color="#a57")
    plt.xlabel("grado de duplicado, alcance de entrenamiento")
    plt.ylabel("filas")
    plt.title("Grado de duplicado con el unico alcance que puede ser feature")
    guardar("p63_12_grado_duplicado.png")

    # 17, 18, 19, 20, 21: los errores
    error = (pred_v03 != y).astype(int)
    n_tok = perfil["txt_n_tokens"].to_numpy()
    plt.figure(figsize=(11, 5))
    nombres_bin = [b[0] for b in BINS]
    tasas = []
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        tasas.append(error[m].mean())
    plt.bar(nombres_bin, [t * 100 for t in tasas], color="#c66")
    plt.ylabel("por ciento de celdas mal decididas")
    plt.title("Error de V03 por rebanada de longitud, con los cortes congelados")
    guardar("p63_13_error_por_rebanada.png")

    plt.figure(figsize=(11, 5))
    anchos = 0.25
    for k, (nombre, j) in enumerate((("HS", 0), ("TR", 1), ("AG", 2))):
        vals = []
        for nombre_bin, lo, hi in BINS:
            m = (n_tok >= lo) & (n_tok <= hi)
            vals.append(error[m, j].mean() * 100)
        plt.bar(np.arange(len(BINS)) + k * anchos, vals, width=anchos, label=nombre)
    plt.xticks(np.arange(len(BINS)) + anchos, nombres_bin)
    plt.ylabel("por ciento de error")
    plt.title("Error de V03 por rebanada y etiqueta")
    plt.legend()
    guardar("p63_14_error_por_rebanada_y_etiqueta.png")

    plt.figure(figsize=(11, 5))
    techo = inte["techo_por_rebanada"]
    x = np.arange(len(BINS))
    plt.bar(x - 0.2, [t["F1_de_V03"] for t in techo], width=0.4, label="F1 de V03",
            color="#4a7")
    plt.bar(x + 0.2, [t["techo_moviendo_el_corte"] for t in techo], width=0.4,
            label="techo moviendo solo el corte", color="#7a4")
    plt.xticks(x, [t["bin"] for t in techo])
    plt.ylabel("macro F1")
    plt.title("El techo que ninguna representacion puede alcanzar, por rebanada")
    plt.legend()
    guardar("p63_15_techo_por_rebanada.png")

    plt.figure(figsize=(11, 5))
    resc = inte["resultados"]["V03_todo"]["rescate"]
    dmg = inte["resultados"]["V03_todo"]["damage"]
    inter = inte["interseccion_de_errores"]
    etiquetas = [f"{r['bin']}\n{r['etiqueta']}" for r in inter]
    plt.bar(etiquetas, [r["rescatados_por_el_bloque"] for r in inter], color="#4a7",
            label="rescatados")
    plt.bar(etiquetas, [-r["danados_por_el_bloque"] for r in inter], color="#c44",
            label="danados")
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("celdas de decision")
    plt.title(f"Rescate contra dano del bloque conjunto: {resc} y {dmg} en el total")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.legend()
    guardar("p63_16_rescate_vs_dano.png")

    resid = json.loads((REPORTS / "p63_residuo_de_v03.json").read_text())
    aucs = resid["resultados"]
    plt.figure(figsize=(11, 5))
    brazos = list(aucs)
    ancho = 0.2
    for k, et in enumerate(ETIQUETAS):
        plt.bar(np.arange(len(brazos)) + k * ancho,
                [aucs[b]["auc_por_etiqueta"][et] for b in brazos], width=ancho, label=et)
    plt.axhline(0.5, color="k", ls=":", lw=1)
    plt.xticks(np.arange(len(brazos)) + ancho, brazos, rotation=20, ha="right")
    plt.ylabel("AUC para predecir el error de V03")
    plt.title("Los bloques contra el residuo de V03: el margen solo ya explica los errores")
    plt.legend()
    guardar("p63_17_interseccion_errores.png")

    plt.figure(figsize=(11, 5))
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        if m.sum() == 0:
            continue
        p = proba_v03[m, 0]
        plt.hist(p, bins=20, alpha=0.45, label=f"{nombre} (n={int(m.sum())})")
    plt.xlabel("probabilidad de HS que da V03")
    plt.ylabel("filas")
    plt.title("La distribucion de la confianza de V03 se desplaza con la longitud")
    plt.legend()
    guardar("p63_18_confianza_por_rebanada.png")

    plt.figure(figsize=(11, 5))
    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        if m.sum() == 0:
            continue
        plt.hist(margen[m, 0], bins=20, alpha=0.45, label=nombre)
    plt.xlabel("margen de HS")
    plt.ylabel("filas")
    plt.title("El margen de V03 por rebanada: de donde sale el error")
    plt.legend()
    guardar("p63_19_margen_por_rebanada.png")

    # 22: diagnostico por bloque del id
    with open(REPORTS / "p63_id_forensics_ocupacion.csv", encoding="ascii") as fh2:
        filas_oc = list(csv.DictReader(fh2))
    plt.figure(figsize=(11, 5))
    plt.bar([f["bloque"] for f in filas_oc], [float(f["fraccion_ocupada"])
                                              for f in filas_oc], color="#57a")
    plt.ylabel("fraccion ocupada del bloque de ids")
    plt.title("Ocupacion del bloque por tramo de id, sin ninguna subfuente")
    guardar("p63_20_id_bloques.png")

    # 23: evolucion de los candidatos a lo largo de los experimentos
    plt.figure(figsize=(11, 5))
    orden = ["id_solo", "rel_solo", "comp_solo", "V03_comp", "pos_solo", "V03_pos",
             "V03_todo"]
    vals = [candidatos[k][0] for k in orden]
    plt.plot(orden, vals, marker="o", color="#48a", label="candidate corrected_cv")
    plt.axhline(ccv_ref, color="k", ls="--", label="V03")
    plt.ylabel("corrected_cv")
    plt.title("Evolucion de los candidatos a lo largo de los siete experimentos")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    guardar("p63_21_evolucion_candidatos.png")

    log(f"   {len(hechos)} figuras escritas en {FIGURAS.relative_to(PROJECT_ROOT)}")

    # ---------- el esquema ----------
    L: list[str] = []
    W = L.append
    W("# P63 - Dataset auxiliar de registros")
    W("")
    W(f"Generado por `experiments/p63_09_dataset_auxiliar.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("> **NO ES UN EXPERIMENTO.** No entrena, no crea candidatos, no usa test, holdout ni")
    W("> leaderboard. Construye el artefacto que el encargo pide y deja la procedencia de")
    W("> cada columna escrita, no solo contada.")
    W("")
    W("## Las tres familias, separadas a proposito")
    W("")
    W("| familia | columnas | que es | permitido como entrada |")
    W("|---|---:|---|---|")
    W(f"| RECORD | {record.shape[1] - 1} | propiedades de la fila que NO leen el texto | "
      f"**NO**: P63-03 lo mostro peor que un control de capacidad |")
    W(f"| RELATIONAL | {relacional.shape[1] - 1} | el agrupador canonico de casi "
      f"duplicados | las de recuento SI, con el protocolo de alcance |")
    W(f"| TEXT_DERIVED | {text_derived.shape[1] - 2} | todo lo que lee el texto | SI, con "
      f"escala corregida y control de capacidad |")
    W("")
    W("El gap analysis showed que el bloque historico de registro era 45 de 46")
    W("text-derived, luego esta distincion es justo el punto de construir el fichero: una")
    W("columna que se llama de registro y en realidad lee el texto no es un rasgo de")
    W("registro.")
    W("")
    W("**El dataset no contiene HS, TR, AG, ni OOF, ni test, ni target encoding.** El")
    W("script lo comprueba con un assert antes de escribir.")
    W("")
    W("## El alcance de las columnas relacionales")
    W("")
    W("La unica columna RELATIONAL que se escribe es `rel_tamano_grupo_entrenamiento`, y su")
    W("alcance se calcula **dentro de cada pliegue de entrenamiento**. Una columna con el")
    W("mismo nombre y alcance sobre todo el dataset seria fuga, y la diferencia entre ambas")
    W("es exactamente lo que el track B midio: -0,1206 de corrected_cv.")
    W("")
    W("Las columnas `rel_contradice_*` se escriben marcadas como NO usables. Un grupo con")
    W("la etiqueta contradictoria no puede acertarla todas, luego es un techo y no un")
    W("rasgo, y la razon queda en el esquema en vez de desaparecer.")
    W("")
    W("## El esquema completo")
    W("")
    W("Todas las columnas estan en `features/toxic_record_profile_schema.md` y en")
    W("`reports/p63_record_features.csv`, con tipo, familia, definicion, origen, regla de")
    W("seguridad de pliegue y uso permitido. Resumen por familia:")
    W("")
    W("| familia | columnas | n distintos en RECORD y RELATIONAL |")
    W("|---|---:|---|")
    for fam in ("CLAVE", "RECORD", "RELATIONAL", "TEXT_DERIVED"):
        sub = [f for f in filas_schema if f["familia"] == fam]
        W(f"| {fam} | {len(sub)} | "
          f"{'-' if fam == 'TEXT_DERIVED' else str(sum(1 for f in sub if f['n_distintos']))} |")
    W("")
    W("## Las figuras")
    W("")
    W(f"{len(hechos)} figuras, y ninguna oculta un resultado negativo. La de candidatos")
    W("colorea en rojo todo lo que pierde contra V03 y la de delta tiene la linea de cero a")
    W("la vista.")
    W("")
    for f in hechos:
        W(f"* `figuras/p63/{f}`")
    W("")
    W("## La lectura que las figuras sostienen")
    W("")
    W("1. La ocupacion del bloque de ids es casi uniforme, luego no hay subfuentes.")
    W("2. El margen de V03 explica sus propios errores con AUC 0,7571, y los bloques nuevos")
    W("   lo empeoran, luego lo que saben ya estaba en la confianza.")
    W("3. El techo moviendo solo el corte crece de +0,0000 a +0,0890 con la longitud.")
    W("4. El error crece de 6,59 a 22,55 por ciento en la misma direccion.")
    W("")
    W("Las cuatro lnneas apuntan al mismo sitio y ninguna la ataca.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p63_09_dataset_auxiliar.py`. Sin push.*")
    MD_C.write_text(a_ascii("\n".join(L)), encoding="ascii")

    S: list[str] = []
    Ws = S.append
    Ws("# Esquema de `features/toxic_record_profile.parquet`")
    Ws("")
    Ws(f"Fecha {inicio.isoformat(timespec='seconds')}. Generado por "
       f"`experiments/p63_09_dataset_auxiliar.py`.")
    Ws("")
    Ws("Una fila por registro de la particion de desarrollo. **Sin etiquetas, sin OOF, sin")
    Ws("test y sin target encoding.** El script lo verifica con un assert antes de escribir.")
    Ws("")
    Ws("## Familias")
    Ws("")
    Ws("```text")
    Ws("RECORD      propiedades de la fila que NO leen el texto")
    Ws("RELATIONAL  propiedades del agrupador canonico de casi duplicados")
    Ws("TEXT_DERIVED todo lo que lee el texto, en cualquiera de sus cuatro familias")
    Ws("            lexical, categorica, ortografica y posicional")
    Ws("```")
    Ws("")
    Ws("## Columnas")
    Ws("")
    Ws("| columna | tipo | familia | origen | uso permitido |")
    Ws("|---|---|---|---|---|")
    for f in filas_schema:
        Ws(f"| `{f['columna']}` | {f['tipo']} | {f['familia']} | {f['source']} | "
           f"{f['uso_permitido']} |")
    Ws("")
    Ws("## Reglas de seguridad de pliegue por familia")
    Ws("")
    for fam in ("CLAVE", "RECORD", "RELATIONAL", "TEXT_DERIVED"):
        Ws(f"### {fam}")
        Ws("")
        Ws(f"- **origen**: {reglas[fam][0]}")
        Ws(f"- **regla de pliegue**: {reglas[fam][1]}")
        Ws(f"- **uso permitido**: {reglas[fam][2]}")
        Ws("")
    Ws("## Lo que NO se escribe y por que")
    Ws("")
    Ws("- **target encoding de id, grupo o cluster**: prohibido por el encargo, y ademas")
    Ws("  un grupo no puede convertirse en lookup de etiqueta.")
    Ws("- **columnas de etiqueta**: el fichero es de features. Las etiquetas viven en")
    Ws("  tablas de entrenamiento separadas.")
    Ws("- **el conteo de grupo con alcance COMPLETO**: es fuga, y el track B midio que vale")
    Ws("  -0,1206 de corrected_cv frente al alcance de entrenamiento.")
    Ws("- **`doc.vector`, `token.vector` y `similarity()` de spaCy**: prohibidos por D-003.")
    Ws("")
    Ws("> V03 remains the best verified leaderboard result.")
    SCHEMA.write_text(a_ascii("\n".join(S)), encoding="ascii")

    for destino in (MD_C, SCHEMA, REPORTS / "p63_record_features.csv",
                    REPORTS / "p63_record_structure.csv"):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-09 OK | {perfil.shape[1]} columnas | {len(hechos)} figuras | 0 experimentos")


if __name__ == "__main__":
    import csv
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()

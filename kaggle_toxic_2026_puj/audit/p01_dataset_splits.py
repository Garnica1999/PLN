"""P1: integridad del dataset, estructura de etiquetas, grupos y splits.

Read-only sobre los datos; escribe hallazgos en reports/ y audit/out/.
Reproduce el clustering de near-duplicados del proyecto (train+test) y un
contrafactual solo-train para medir el puente test->grupos de validacion.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
REPORTS = PROJ / "reports"
OUT = PROJ / "audit" / "out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJ))

from src import data as data_module  # noqa: E402
from src import splits as splits_module  # noqa: E402


def load_audit_module():
    """Carga experiments/00_auditoria_datos.py como modulo sin ejecutar main."""
    path = PROJ / "experiments" / "00_auditoria_datos.py"
    text = path.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in text, "el modulo 00 no tiene guard de main"
    spec = importlib.util.spec_from_file_location("audit00", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_signature(labels: np.ndarray, ids: np.ndarray) -> dict[int, str]:
    """Mapea cada id a la huella de su grupo (ids ordenados separados por coma)."""
    frames = pd.DataFrame({"id": ids, "grupo": labels})
    members = frames.groupby("grupo")["id"].apply(lambda s: ",".join(map(str, sorted(s))))
    return dict(zip(frames["id"], frames["grupo"].map(members), strict=True))


def overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Numero de etiquetas de grupo compartidas entre dos subconjuntos."""
    return len(set(a.tolist()) & set(b.tolist()))


def main() -> None:  # noqa: PLR0915
    a00 = load_audit_module()
    train = data_module.read_train()
    test = data_module.read_test()
    submission = data_module.read_submission()
    train["id"] = train["id"].astype(int)
    test["id"] = test["id"].astype(int)
    submission["id"] = submission["id"].astype(int)
    findings: dict = {}

    # ---------- 1. Integridad ----------
    integrity = a00.integridad(train, test, submission)
    quality = a00.calidad_de_archivos()
    findings["integridad"] = integrity
    findings["calidad_archivos"] = quality.to_dict(orient="records")
    findings["bytes_reales"] = {
        name: (data_module.RAW_DIR / name).stat().st_size
        for name in data_module.EXPECTED_BYTES
    }
    findings["bytes_esperados"] = data_module.EXPECTED_BYTES
    findings["encodings"] = {
        "toxic_train.csv": data_module.decode_bytes(
            (data_module.RAW_DIR / "toxic_train.csv").read_bytes())[1],
        "toxic_test.csv": data_module.decode_bytes(
            (data_module.RAW_DIR / "toxic_test.csv").read_bytes())[1],
    }

    # ---------- 2. Etiquetas ----------
    marginal, combinaciones = a00.distribucion_etiquetas(train)
    findings["marginal"] = marginal.to_dict(orient="records")
    findings["combinaciones"] = combinaciones.to_dict(orient="records")
    labels = train[["HS", "TR", "AG"]].to_numpy()
    corr = train[["HS", "TR", "AG"]].corr().to_numpy()
    findings["correlaciones_phi"] = {
        "HS-TR": round(float(corr[0, 1]), 6),
        "HS-AG": round(float(corr[0, 2]), 6),
        "TR-AG": round(float(corr[1, 2]), 6),
    }
    findings["texto_no_ascii_train"] = int(sum(
        1 for t in train["text"] if any(ord(c) > 127 for c in t)))
    findings["texto_no_ascii_test"] = int(sum(
        1 for t in test["text"] if any(ord(c) > 127 for c in t)))
    findings["longitud_chars_mediana_train"] = float(train["text"].str.len().median())
    findings["longitud_chars_mediana_test"] = float(test["text"].str.len().median())

    # ---------- 3. Near-duplicados: provenance del CSV ----------
    grupos_csv = pd.read_csv(PROJ / "data" / "processed" / "grupos_near_duplicados.csv")
    findings["grupos_csv_filas"] = len(grupos_csv)
    findings["grupos_csv_por_split"] = grupos_csv["split"].value_counts().to_dict()
    comp = grupos_csv.groupby("grupo")["split"].agg(
        n_train=lambda s: int((s == "train").sum()),
        n_test=lambda s: int((s == "test").sum()),
    )
    solo_train = comp[(comp["n_train"] > 0) & (comp["n_test"] == 0)]
    mixtos = comp[(comp["n_train"] > 0) & (comp["n_test"] > 0)]
    solo_test = comp[(comp["n_train"] == 0) & (comp["n_test"] > 0)]
    findings["grupos_near_dup"] = {
        "total": int(len(comp)),
        "tamano_gt1": int((comp["n_train"] + comp["n_test"] > 1).sum()),
        "solo_train": int(len(solo_train)),
        "solo_test": int(len(solo_test)),
        "mixtos": int(len(mixtos)),
        "filas_train_en_mixtos": int(mixtos["n_train"].sum()),
        "filas_test_en_mixtos": int(mixtos["n_test"].sum()),
        "max_tamano": int((comp["n_train"] + comp["n_test"]).max()),
    }
    mixtos.sort_values(["n_train", "n_test"], ascending=False).to_csv(
        OUT / "p01_grupos_mixtos_train_test.csv")
    findings["grupos_mixtos_detalle"] = mixtos.reset_index().to_dict(orient="records")

    # Similitud recomputada y comparacion con el artefacto guardado
    sim_resumen, _sim_pares = a00.analisis_similitud(train, test)
    findings["similitud_recomputada"] = sim_resumen.to_dict(orient="records")
    stored_sim = REPORTS / "eda" / "similitud_resumen.csv"
    if stored_sim.exists():
        findings["similitud_guardada"] = pd.read_csv(stored_sim).to_dict(orient="records")

    # ---------- 4. Contrafactual: clusters solo-train ----------
    empty_test = test.iloc[0:0].copy()
    asig_trainonly, _ = a00.clusters_near_duplicados(train, empty_test, threshold=0.85)
    asig_plus = grupos_csv[grupos_csv["split"] == "train"][["id", "grupo"]]
    merged = asig_trainonly.merge(asig_plus, on="id", suffixes=("_solo", "_plus"))
    ids_train = merged["id"].to_numpy()
    labels_solo = merged["grupo_solo"].to_numpy()
    labels_plus = merged["grupo_plus"].to_numpy()
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415
    findings["contrafactual_grupos"] = {
        "n_grupos_solo_train": int(len(set(labels_solo.tolist()))),
        "n_grupos_plus": int(len(set(labels_plus.tolist()))),
        "ari_solo_vs_plus": round(float(adjusted_rand_score(labels_solo, labels_plus)), 6),
        "filas_train_que_cambian_de_grupo": int(
            sum(1 for s, p in zip(
                canonical_signature(labels_solo, ids_train).values(),
                canonical_signature(labels_plus, ids_train).values(), strict=True) if s != p)),
    }
    cross = sum(
        1 for g in mixtos.index
        if ((grupos_csv["grupo"] == g) & (grupos_csv["split"] == "train")).any()
    )
    findings["contrafactual_grupos"]["grupos_mixtos_con_train"] = int(cross)

    # ---------- 5. Splits con grupos train+test (como el proyecto) ----------
    grupos = splits_module.grupos_fusionados(train)
    idx_dev, idx_hold = splits_module.holdout_v2(train, grupos)
    hold_ids = set(train.iloc[idx_hold]["id"].tolist())
    stored_hold = pd.read_csv(PROJ / "data" / "processed" / "holdout_v2_ids.csv")
    stored_hold_ids = set(stored_hold[stored_hold["particion"] == "holdout"]["id"].tolist())
    findings["holdout_v2"] = {
        "n_dev": int(len(idx_dev)),
        "n_holdout": int(len(idx_hold)),
        "coincide_csv_ids": bool(hold_ids == stored_hold_ids),
        "interseccion_csv": int(len(hold_ids & stored_hold_ids)),
        "solo_codigo": int(len(hold_ids - stored_hold_ids)),
        "solo_csv": int(len(stored_hold_ids - hold_ids)),
    }
    # Huella de grupos fusionados vs columna del CSV guardado
    sig_codigo = canonical_signature(grupos, train["id"].to_numpy())
    sig_csv = canonical_signature(
        stored_hold["grupo_near_dup_autor"].to_numpy(), stored_hold["id"].to_numpy())
    mismatches = sum(1 for i in train["id"] if sig_codigo[int(i)] != sig_csv[int(i)])
    findings["huella_grupos_fusionados_vs_csv"] = {"filas_discrepantes": int(mismatches)}

    # Pliegues corregidos sobre dev
    dev = train.iloc[idx_dev].reset_index(drop=True)
    dev_groups = grupos[idx_dev]
    folds = splits_module.pliegues_corregidos(dev, dev_groups)
    fold_rows = []
    combos_dev = splits_module.combinaciones(dev)
    for k, (tr_idx, va_idx) in enumerate(folds):
        sub = dev.iloc[va_idx]
        fold_rows.append({
            "fold": k,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "HS_rate": round(float(sub["HS"].mean()), 4),
            "TR_rate": round(float(sub["TR"].mean()), 4),
            "AG_rate": round(float(sub["AG"].mean()), 4),
            "grupos_compartidos": overlap(dev_groups[tr_idx], dev_groups[va_idx]),
            "combinaciones_ausentes": int(sum(
                1 for combo in combos_dev.value_counts()[lambda s: s >= 5].index
                if combo not in set(combos_dev.iloc[va_idx].tolist()))),
        })
    findings["pliegues_corregidos"] = fold_rows
    findings["n_grupos_fusionados"] = int(len(set(grupos.tolist())))
    findings["tamano_grupo_fusionado_max"] = int(
        pd.Series(grupos).value_counts().max())

    # ---------- 6. Contrafactual de splits con grupos solo-train ----------
    near_solo = asig_trainonly.set_index("id")["grupo"]
    ids = train["id"].astype(int).to_numpy()
    autores = splits_module.pseudo_autores(train)
    padre = list(range(len(ids)))

    def encontrar(nodo: int) -> int:
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[nodo]]
            nodo = padre[nodo]
        return nodo

    def unir(a: int, b: int) -> None:
        ra, rb = encontrar(a), encontrar(b)
        if ra != rb:
            padre[max(ra, rb)] = min(ra, rb)

    rep_near: dict[int, int] = {}
    for pos, ident in enumerate(ids):
        g = int(near_solo.get(ident, ident))
        if g in rep_near:
            unir(pos, rep_near[g])
        else:
            rep_near[g] = pos
    rep_autor: dict[str, int] = {}
    for pos, autor in enumerate(autores):
        if not autor:
            continue
        if autor in rep_autor:
            unir(pos, rep_autor[autor])
        else:
            rep_autor[autor] = pos
    grupos_solo = np.array([encontrar(n) for n in range(len(ids))])

    idx_dev2, idx_hold2 = splits_module.holdout_v2(train, grupos_solo)
    hold2 = set(train.iloc[idx_hold2]["id"].tolist())
    folds2 = splits_module.pliegues_corregidos(
        train.iloc[idx_dev2].reset_index(drop=True), grupos_solo[idx_dev2])
    fold_of: dict[int, int] = {}
    for k, (_tr, va) in enumerate(folds):
        va_set = set(va.tolist())
        for i, ident in enumerate(dev["id"].tolist()):
            if i in va_set:
                fold_of[int(ident)] = k
    fold_of2: dict[int, int] = {}
    dev2 = train.iloc[idx_dev2].reset_index(drop=True)
    for k, (_tr, va) in enumerate(folds2):
        va_set = set(va.tolist())
        for i, ident in enumerate(dev2["id"].tolist()):
            if i in va_set:
                fold_of2[int(ident)] = k
    comunes = set(fold_of) & set(fold_of2)
    findings["contrafactual_splits"] = {
        "n_grupos_solo_train": int(len(set(grupos_solo.tolist()))),
        "holdout_distinto": int(len(hold_ids ^ hold2)),
        "filas_cambian_de_pliegue": int(sum(
            1 for ident in comunes if fold_of[ident] != fold_of2[ident])),
        "filas_comparables": int(len(comunes)),
    }

    # ---------- 7. Escritura ----------
    (PROJ / "audit" / "p01_findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=True, default=str), encoding="ascii")
    rows = [
        {
            "Experiment_ID": "GLOBAL/near-dup",
            "Leakage_Type": "L5_potencial",
            "Potential": "si",
            "Detected": "si",
            "Evidence": (f"clusters_near_duplicados ajusta TfidfVectorizer char_wb(4,4) "
                         f"sobre train+test (2885 textos); grupos mixtos="
                         f"{findings['grupos_near_dup']['mixtos']}; filas train en mixtos="
                         f"{findings['grupos_near_dup']['filas_train_en_mixtos']}"),
            "Affected_Result": "protocolo de validacion v1/v2/v4-v8 (grupos de CV)",
            "Severity": "HIGH",
            "Decision": "cuantificar impacto numerico en P6; ver contrafactual",
        },
        {
            "Experiment_ID": "GLOBAL/splits",
            "Leakage_Type": "L0",
            "Potential": "no",
            "Detected": "no",
            "Evidence": ("src/splits.py: grupos_fusionados usa solo train; "
                         "verificacion de no interseccion por fold ejecutada"),
            "Affected_Result": "n/a",
            "Severity": "INFO",
            "Decision": "mantener",
        },
    ]
    with (OUT / "staging_leakage_rows.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("== P1 resumen ==")
    print("integridad:", {k: v for k, v in integrity.items() if k != "columnas"})
    print("marginal:", findings["marginal"])
    print("grupos_near_dup:", findings["grupos_near_dup"])
    print("correlaciones:", findings["correlaciones_phi"])
    print("holdout:", findings["holdout_v2"])
    print("folds:", findings["pliegues_corregidos"])
    print("contrafactual_grupos:", findings["contrafactual_grupos"])
    print("contrafactual_splits:", findings["contrafactual_splits"])
    print("no_ascii train/test:", findings["texto_no_ascii_train"], findings["texto_no_ascii_test"])


if __name__ == "__main__":
    main()

"""P4b: cuantificacion del mecanismo de sospecha de v6_07 (ruido por par).

Pregunta: los pesos de sospecha se calculan con las etiquetas de TODO el
desarrollo (incluido el pliegue de validacion). Cuantificar:
1. relaciones vecino->fila que cruzan pliegues con etiqueta opuesta (mecanismo);
2. contrafactual fold-local: recalcular los pesos usando solo etiquetas de los
   pliegues de entrenamiento y medir el delta del variante F1-completo.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy import sparse

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
sys.path.insert(0, str(PROJ))

from src.evaluation import metrica_macro  # noqa: E402
from src.splits import (  # noqa: E402
    ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos,
)
from src.data import read_train  # noqa: E402


def load_module():
    path = PROJ / "experiments" / "v6_07_ruido_par.py"
    text = path.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in text
    spec = importlib.util.spec_from_file_location("v607", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:  # noqa: PLR0915
    mod = load_module()
    UMBRALES_V03 = np.asarray(mod.UMBRALES_V03, dtype=float)
    train = read_train()
    train["id"] = train["id"].astype(int)
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    ids = desarrollo["id"].astype(int).to_numpy()
    oof = np.load(PROJ / "features" / "oof_v03_final.npz", allow_pickle=True)
    proba_v03 = oof["proba"]
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    ref = metrica_macro(y, pred_v03)
    fold_de_fila = np.full(len(y), -1, dtype=int)
    for k, (_tr, va) in enumerate(pliegues):
        fold_de_fila[va] = k

    # --- 1. Grafo de vecinos (identico al modulo) ---
    vec_w = TfidfVectorizer(**mod.PARAMS_WORD)
    vec_c = TfidfVectorizer(**mod.PARAMS_CHAR)
    matriz = normalize(sparse.hstack(
        [vec_w.fit_transform(textos), vec_c.fit_transform(textos)]).tocsr())
    vecinos = NearestNeighbors(n_neighbors=mod.VECINOS + 1, metric="cosine").fit(matriz)
    distancias, indices = vecinos.kneighbors(matriz)

    cruces = 0
    cruces_opuestos = 0
    filas_con_cruce = set()
    filas_con_opuesto = set()
    for fila in range(len(y)):
        for pos in range(1, len(indices[fila])):
            j = int(indices[fila, pos])
            if ids[j] == ids[fila] or fold_de_fila[j] == fold_de_fila[fila]:
                continue
            cruces += 1
            filas_con_cruce.add(fila)
            opuestas = [e for e in range(3) if y[j, e] != y[fila, e]]
            if opuestas:
                cruces_opuestos += 1
                filas_con_opuesto.add(fila)

    componentes = mod.detectar_sospecha(textos, y, proba_v03, ids)
    esquemas = mod.ESQUEMAS
    sospecha_global = mod.combinar(componentes, esquemas["F3-agresivo"]["componentes"])

    # --- 2. Contrafactual fold-local para el variante con mas senal (F1-completo) ---
    def esquema_por_nombre(nombre: str) -> dict:
        return esquemas[nombre]

    def detectar_fold_local(k: int, lamina: float) -> np.ndarray:
        """Sospecha por fila con etiquetas conocidas solo en pliegues != k."""
        entrenamiento = [i for i, f in enumerate(fold_de_fila) if f != k]
        conocidos = np.zeros(len(y), dtype=bool)
        conocidos[entrenamiento] = True
        sospecha = np.zeros((len(y), 3), dtype=float)
        confianza = 1.0 - np.abs(2.0 * proba_v03 - 1.0)
        margen = np.clip(1.0 - 2.0 * np.abs(proba_v03 - 0.5), 0.0, 1.0)
        for fila in entrenamiento:
            for etiqueta in range(3):
                opuestos = 0
                validos = 0
                for pos in range(1, len(indices[fila])):
                    j = int(indices[fila, pos])
                    if ids[j] == ids[fila] or not conocidos[j]:
                        continue
                    validos += 1
                    if y[j, etiqueta] != y[fila, etiqueta]:
                        opuestos += 1
                densidad = opuestos / max(validos, 1)
                estructural = 0.0
                if opuestos > 0:
                    mejor = min(
                        distancias[fila, pos] for pos in range(1, len(indices[fila]))
                        if int(indices[fila, pos]) != fila)
                    if mejor < 0.25:
                        estructural = 1.0
                sospecha[fila, etiqueta] = (confianza[fila, etiqueta] + margen[fila, etiqueta]
                                            + estructural + densidad) / 4.0
        return np.clip(sospecha, 0.0, 1.0)

    def ajustar_con_sospecha(sospecha: np.ndarray, lamina: float) -> np.ndarray:
        proba = np.zeros((len(textos), 3), dtype=float)
        for entrenamiento, validacion in pliegues:
            vw = TfidfVectorizer(**mod.PARAMS_WORD)
            vc = TfidfVectorizer(**mod.PARAMS_CHAR)
            m = sparse.hstack([vw.fit_transform(textos.iloc[entrenamiento]),
                               vc.fit_transform(textos.iloc[entrenamiento])]).tocsr()
            q = sparse.hstack([vw.transform(textos.iloc[validacion]),
                               vc.transform(textos.iloc[validacion])]).tocsr()
            for etiqueta in range(3):
                peso = np.clip(1.0 - lamina * sospecha[entrenamiento, etiqueta],
                               mod.SUELO_PESO, 1.0)
                est = LogisticRegression(**mod.PARAMETROS_LR)
                est.fit(m, y[entrenamiento, etiqueta], sample_weight=peso)
                proba[validacion, etiqueta] = est.predict_proba(q)[:, 1]
        return proba

    lamina_f1 = float(esquemas["F1-completo"]["lamina"])
    sospecha_foldlocal = np.zeros_like(sospecha_global)
    for k in range(len(pliegues)):
        sospecha_foldlocal = np.where(
            (fold_de_fila == k)[:, None], detectar_fold_local(k, lamina_f1), sospecha_foldlocal)

    proba_global = ajustar_con_sospecha(sospecha_global, lamina_f1)
    proba_local = ajustar_con_sospecha(sospecha_foldlocal, lamina_f1)
    pred_global = (proba_global >= UMBRALES_V03[None, :]).astype(int)
    pred_local = (proba_local >= UMBRALES_V03[None, :]).astype(int)

    resultado = {
        "referencia_v03_recomputada": round(ref, 6),
        "grafo_vecinos": {
            "vecinos_k": mod.VECINOS,
            "relaciones_cruzando_pliegues": int(cruces),
            "relaciones_cruzadas_con_etiqueta_opuesta": int(cruces_opuestos),
            "filas_con_vecino_en_otro_pliegue": len(filas_con_cruce),
            "filas_con_vecino_opuesto_en_otro_pliegue": len(filas_con_opuesto),
        },
        "esquema": "F1-completo",
        "lamina": lamina_f1,
        "delta_global_estilo_v6": round(metrica_macro(y, pred_global) - ref, 6),
        "delta_fold_local": round(metrica_macro(y, pred_local) - ref, 6),
        "efecto_del_mecanismo": round(
            metrica_macro(y, pred_global) - metrica_macro(y, pred_local), 6),
    }
    puede = esquemas if isinstance(esquemas, dict) else {}
    csv_path = PROJ / "reports" / "v6_ruido_par.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        fila = frame[frame.iloc[:, 0].astype(str).str.contains("F1-completo", na=False)]
        if len(fila):
            resultado["reportado_en_v6_ruido_par_csv"] = {
                c: str(fila.iloc[0][c]) for c in frame.columns[:10]}
    (PROJ / "audit" / "p04b_suspicion_foldlocal.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=True), encoding="ascii")
    print(json.dumps(resultado, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

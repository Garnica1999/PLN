"""V9-04 (H1): modelo secuencial sobre los embeddings FastText propios.

Pregunta
--------
La fase v5 solo probo el embedding agregado (media de tokens) + modelo lineal.
¿Aporta algo un mecanismo secuencial (BiGRU) con los mismos vectores congelados?

Metodo
------
1. Vectores congelados: FASTTEXT-BASELINE (10M, dim 200, sha256 e54a363d) del
   data compartido; normalizacion y tokenizacion EMB-NORMAL del proyecto.
2. Protocolo corregido completo (5 pliegues, grupos fusionados) y umbrales
   congelados de V03 (0.44/0.51/0.46).
3. Dos modelos por pliegue: BiGRU(64) sobre la secuencia (max 64 tokens) y
   baseline de media de vectores + LR C=2 (el uso que probo v5).
4. Early stopping estricto (paciencia 3 sobre la perdida de validacion),
   semillas fijas por pliegue, GPU ROCm si esta disponible.

Salidas: reports/v9_04_dl_secuencial_fasttext.json/.md y fila V9-04.

Ejecucion:
    /home/carlos/venv/bin/python experiments/v9_04_dl_secuencial_fasttext.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.embedding_utils import normalizar_emb_normal, tokenizar
from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

REFERENCIA_V03 = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
MAX_LEN = 64
EPOCAS = 12
PACIENCIA = 3
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DIM_GRU = 64
VEC_PATH = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj/data/external/embeddings/fasttext_baseline.vec")
REPORTE_JSON = PROJECT_ROOT / "reports" / "v9_04_dl_secuencial_fasttext.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v9_04_dl_secuencial_fasttext.md"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"


def log(mensaje: str) -> None:
    """Imprime un mensaje con prefijo de fase."""
    print(f"[V9-04] {mensaje}", flush=True)


def cargar_vec(ruta: Path) -> dict[str, np.ndarray]:
    """Carga un archivo .vec de fastText como diccionario palabra -> vector."""
    vectores: dict[str, np.ndarray] = {}
    with ruta.open(encoding="utf-8", errors="ignore") as fh:
        cabecera = fh.readline().split()
        dimension = int(cabecera[1])
        for linea in fh:
            partes = linea.rstrip().split(" ")
            if len(partes) == dimension + 1:
                vectores[partes[0]] = np.asarray(partes[1:], dtype=np.float32)
    return vectores


def secuencias(textos: list[str], indice: dict[str, int]) -> np.ndarray:
    """Convierte textos a ids de embedding con padding a MAX_LEN."""
    matriz = np.zeros((len(textos), MAX_LEN), dtype=np.int64)
    for fila, texto in enumerate(textos):
        tokens = tokenizar(normalizar_emb_normal(texto))[:MAX_LEN]
        for columna, token in enumerate(tokens):
            matriz[fila, columna] = indice.get(token, 0)
    return matriz


class BiGRU(torch.nn.Module):
    """BiGRU sobre embeddings congelados con pooling enmascarado."""

    def __init__(self, matriz_embedding: np.ndarray) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(matriz_embedding), freeze=True, padding_idx=0)
        self.gru = torch.nn.GRU(matriz_embedding.shape[1], DIM_GRU,
                                batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.salida = torch.nn.Linear(2 * DIM_GRU, 3)

    def forward(self, entrada: torch.Tensor) -> torch.Tensor:
        """Devuelve logits por etiqueta."""
        mascara = (entrada != 0).float().unsqueeze(-1)
        embebido = self.embedding(entrada)
        salida, _ = self.gru(embebido)
        suma = (salida * mascara).sum(dim=1)
        cuenta = mascara.sum(dim=1).clamp(min=1.0)
        promedio = suma / cuenta
        return self.salida(self.dropout(promedio))


def entrenar_pliegue(ids_tr: np.ndarray, y_tr: np.ndarray, ids_va: np.ndarray,
                     y_va: np.ndarray, matriz_embedding: np.ndarray,
                     semilla: int, dispositivo: torch.device) -> np.ndarray:
    """Entrena el BiGRU de un pliegue y devuelve probabilidades de validacion."""
    torch.manual_seed(semilla)
    if dispositivo.type == "cuda":
        torch.cuda.manual_seed_all(semilla)
    modelo = BiGRU(matriz_embedding).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterio = torch.nn.BCEWithLogitsLoss()
    x_tr = torch.tensor(ids_tr, device=dispositivo)
    t_tr = torch.tensor(y_tr, dtype=torch.float32, device=dispositivo)
    x_va = torch.tensor(ids_va, device=dispositivo)
    t_va = torch.tensor(y_va, dtype=torch.float32, device=dispositivo)
    mejor_perdida = float("inf")
    mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
    sin_mejora = 0
    for epoca in range(EPOCAS):
        modelo.train()
        permutacion = torch.randperm(len(x_tr), device=dispositivo)
        for inicio in range(0, len(x_tr), BATCH):
            idx = permutacion[inicio:inicio + BATCH]
            optimizador.zero_grad()
            logits = modelo(x_tr[idx])
            perdida = criterio(logits, t_tr[idx])
            perdida.backward()
            optimizador.step()
        modelo.eval()
        with torch.no_grad():
            perdida_va = float(criterio(modelo(x_va), t_va).item())
        if perdida_va < mejor_perdida - 1e-5:
            mejor_perdida = perdida_va
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        proba = torch.sigmoid(modelo(x_va)).cpu().numpy()
    return proba


def main() -> None:  # noqa: PLR0915
    inicio = time.perf_counter()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"dispositivo: {dispositivo}")
    vectores = cargar_vec(VEC_PATH)
    log(f"vectores cargados: {len(vectores)}")

    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    dev = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(dev, grupos[indices_dev])
    y = dev[list(ETIQUETAS)].to_numpy()

    vocabulario = sorted({t for texto in dev["text"]
                          for t in tokenizar(normalizar_emb_normal(texto)) if t in vectores})
    indice = {palabra: i + 1 for i, palabra in enumerate(vocabulario)}
    matriz_embedding = np.zeros((len(vocabulario) + 1, 200), dtype=np.float32)
    for palabra, posicion in indice.items():
        matriz_embedding[posicion] = vectores[palabra]
    ids = secuencias(dev["text"].tolist(), indice)
    log(f"vocabulario usado: {len(vocabulario)} tipos")

    proba_gru = np.zeros_like(y, dtype=float)
    proba_media = np.zeros_like(y, dtype=float)
    for k, (tr, va) in enumerate(pliegues):
        proba_gru[va] = entrenar_pliegue(
            ids[tr], y[tr], ids[va], y[va], matriz_embedding, 42 + k, dispositivo)
        medias_tr = np.zeros((len(tr), 200), dtype=np.float32)
        medias_va = np.zeros((len(va), 200), dtype=np.float32)
        for posicion, fila in enumerate(tr):
            tokens = [indice[t] for t in tokenizar(normalizar_emb_normal(dev["text"].iloc[fila]))
                      if t in indice]
            if tokens:
                medias_tr[posicion] = matriz_embedding[tokens].mean(axis=0)
        for posicion, fila in enumerate(va):
            tokens = [indice[t] for t in tokenizar(normalizar_emb_normal(dev["text"].iloc[fila]))
                      if t in indice]
            if tokens:
                medias_va[posicion] = matriz_embedding[tokens].mean(axis=0)
        for i in range(3):
            modelo = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                        random_state=42)
            modelo.fit(medias_tr, y[tr, i])
            proba_media[va, i] = modelo.predict_proba(medias_va)[:, 1]
        log(f"  pliegue {k + 1} listo")

    def evaluar(proba: np.ndarray, umbrales: dict[str, float]) -> dict:
        pred = np.column_stack([(proba[:, i] >= umbrales[e]).astype(int)
                                for i, e in enumerate(ETIQUETAS)])
        return metricas_multietiqueta(y, pred)

    gru_frozen = evaluar(proba_gru, UMBRALES_V03)
    gru_05 = evaluar(proba_gru, {e: 0.5 for e in ETIQUETAS})
    media_frozen = evaluar(proba_media, UMBRALES_V03)
    media_05 = evaluar(proba_media, {e: 0.5 for e in ETIQUETAS})
    veredicto = ("SECUENCIAL NO SUPERA A LA MEDIA"
                 if gru_frozen["f1_media_columnas"] <= media_frozen["f1_media_columnas"]
                 else "SECUENCIAL SUPERA A LA MEDIA; comparar contra V03")

    reporte = {
        "experimento": "V9-04 (H1)",
        "fecha": "2026-09-26",
        "dispositivo": str(dispositivo),
        "embedding": {"archivo": str(VEC_PATH.name), "dim": 200,
                      "vocabulario_usado": len(vocabulario)},
        "hiperparametros": {"max_len": MAX_LEN, "epocas": EPOCAS, "paciencia": PACIENCIA,
                            "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
                            "dim_gru": DIM_GRU, "embeddings": "congelados"},
        "gru_umbrales_v03": gru_frozen["f1_media_columnas"],
        "gru_umbral_0_5": gru_05["f1_media_columnas"],
        "media_lr_umbrales_v03": media_frozen["f1_media_columnas"],
        "media_lr_umbral_0_5": media_05["f1_media_columnas"],
        "referencia_v03": REFERENCIA_V03,
        "delta_gru_vs_media": round(gru_frozen["f1_media_columnas"]
                                    - media_frozen["f1_media_columnas"], 6),
        "delta_gru_vs_v03": round(gru_frozen["f1_media_columnas"] - REFERENCIA_V03, 6),
        "veredicto": veredicto,
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
    }
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    lineas = [
        "# V9-04 (H1): secuencial (BiGRU) sobre embeddings FastText propios",
        "",
        "| modelo | umbrales V03 | umbral 0.5 |",
        "|---|---|---|",
        f"| BiGRU (congelados) | {gru_frozen['f1_media_columnas']:.4f} | "
        f"{gru_05['f1_media_columnas']:.4f} |",
        f"| Media + LR (v5) | {media_frozen['f1_media_columnas']:.4f} | "
        f"{media_05['f1_media_columnas']:.4f} |",
        f"| V03 (referencia) | {REFERENCIA_V03:.6f} | - |",
        "",
        f"Veredicto: **{veredicto}** (delta secuencial - media "
        f"{reporte['delta_gru_vs_media']:+.4f}; vs V03 {reporte['delta_gru_vs_v03']:+.4f}).",
        "",
        "Cierra la brecha de la seccion 24 de la auditoria: el mecanismo secuencial se",
        "probo con los vectores propios congelados y early stopping estricto.",
        f"\nFecha: 2026-09-26. Duracion: {reporte['duracion_segundos']} s.",
    ]
    REPORTE_MD.write_text("\n".join(lineas), encoding="utf-8")

    fila_reg = {
        "experiment_id": "V9-04",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Aporta un mecanismo secuencial (BiGRU) sobre los embeddings propios congelados?",
        "hypothesis": "El uso secuencial podria capturar orden/contexto que el pooling por media pierde",
        "preprocessing": "EMB-NORMAL + tokenizacion del proyecto; max 64 tokens",
        "feature_groups": "secuencia de embeddings FastText 200d congelados",
        "representation": "FASTTEXT-BASELINE 10M (sha256 e54a363d...)",
        "model": "BiGRU(64) + Linear(3) y baseline media + LR C=2",
        "hyperparameters": json.dumps(reporte["hiperparametros"], ensure_ascii=False),
        "validation": "protocolo corregido 5 pliegues; early stopping; umbrales congelados V03",
        "f1": round(gru_frozen["f1_media_columnas"], 4),
        "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": reporte["duracion_segundos"],
        "dimension": "200",
        "result": json.dumps({"gru": reporte["gru_umbrales_v03"],
                              "media_lr": reporte["media_lr_umbrales_v03"],
                              "delta": reporte["delta_gru_vs_media"],
                              "veredicto": veredicto}, ensure_ascii=False),
        "interpretation": "Comparacion directa secuencial vs agregado con la misma representacion",
        "decision": "Sin adopcion (resultado registrado)",
        "date": "2026-09-26",
        "script": "experiments/v9_04_dl_secuencial_fasttext.py",
        "notes": "Artefactos: reports/v9_04_dl_secuencial_fasttext.json/.md",
    }
    existente = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if "V9-04" in existente.get("experiment_id", pd.Series(dtype=str)).tolist():
        existente = existente[existente["experiment_id"] != "V9-04"]
        existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame([fila_reg]).to_csv(EXPERIMENTS_CSV, mode="a",
                                    header=not EXPERIMENTS_CSV.exists(), index=False)
    log(f"BiGRU {gru_frozen['f1_media_columnas']:.4f} vs media {media_frozen['f1_media_columnas']:.4f} "
        f"vs V03 {REFERENCIA_V03:.6f} -> {veredicto}")


if __name__ == "__main__":
    main()

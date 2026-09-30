"""P73: TextCNN sobre Word2Vec Twitter congelado frente al mismo lector con embedding aleatorio.

El pre-registro va ANTES, en docs/p73_preregistro_textcnn.md, commit 7192a22. Este script lo ejecuta
tal como se declaro alli. Cuando algo se desvie, se declara en el JSON de salida.

La pregunta
-----------
P72 pregunto si Word2Vec mejora una representacion lineal basada en mean pooling. Respondio que no,
y que la geometria aprendida no aporta ventaja sobre direcciones aleatorias a ese lector.

P73 pregunta algo distinto: cuando dejamos de promediar y permitimos que un modelo secuencial
explote orden y contexto local, contiene la representacion aprendida informacion que una matriz
aleatoria equivalente no contiene?

Las dos ramas comparten TODO menos la matriz de embeddings: vocabulario, ids, longitudes, mascara,
particion, pesos iniciales, semilla, orden de lotes, dropout, optimizador, scheduler, early stopping
y umbrales. El diff entre los dos runs debe ser exclusivamente la matriz, y eso se verifica por
hash y por asserts.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import math
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import (
    ETIQUETAS,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

REPORTS = PROJECT_ROOT / "reports"
FEATURES = PROJECT_ROOT / "features"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"
RE_TOKEN = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)
UMBRALES = np.array([0.44, 0.51, 0.46])

MAXLEN = 60
FILTROS = 64
KERNELS = (2, 3, 4)
HIDDEN = 64
DROPOUT = 0.5
LR_PICO = 1e-3
LR_MIN = 1e-4
WARMUP = 3
EPOCAS = 40
BATCH = 32
PACIENCIA = 6
VENTANA_ES = 3
TOL_ES = 1e-3
CLIP = 1.0
WD = 1e-4
SEMILLA = 20261101
SEMILLA_BOOT = 20261102
N_BOOTSTRAP = 2000
FRACCION_INTERNA = 0.8
MAX_INTENTOS_INTERNA = 20
TOL_PREVALENCIA = 0.07
V03_CCV = 0.7772114525165721
P72_EMB_ONLY = 0.5947288906
PISO_PRIOR_ESPERADO = 0.653694

LISTA_ESTRUCTURA = {
    "URL": r"http|www\.|url",
    "HASHTAG": r"#\w",
    "MENCION": r"@\w",
    "ELONGACION": r"(.)\1{2,}",
}
# Lista declarada, solo para rebanar el diagnostico. NO es una feature, no entra en ningun modelo,
# y es el corte menos fiable de los cinco. Se declara aqui para que sea auditable.
LISTA_INSULTOS = ("puta", "puto", "mierda", "gilipollas", "idiota", "imbecil", "estupido",
                  "cabron", "pendejo", "verga", "joder", "cojones", "zorra", "marica",
                  "malparido", "gonorrea", "hp", "hijueputa")


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P73-02] {m}", flush=True)


def sha_arr(x) -> str:
    """sha1 de un array o tensor, para el manifiesto de comparabilidad.

    Args:
        x: array, tensor o secuencia de enteros.

    Returns:
        Los primeros 40 caracteres del sha1.
    """
    if isinstance(x, torch.Tensor):
        b = x.detach().cpu().numpy().tobytes()
    elif isinstance(x, np.ndarray):
        b = np.ascontiguousarray(x).tobytes()
    else:
        b = np.asarray(list(x)).tobytes()
    return hashlib.sha1(b).hexdigest()[:40]


def sha_matriz(m: np.ndarray) -> str:
    """sha1 de una matriz de embeddings.

    Args:
        m: matriz.

    Returns:
        Los primeros 40 caracteres del sha1.
    """
    return hashlib.sha1(np.ascontiguousarray(m).tobytes()).hexdigest()[:40]


def f1m(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 del proyecto sobre filas concatenadas.

    Args:
        y: etiquetas.
        pred: predicciones binarias.

    Returns:
        La media de las F1 por columna.
    """
    return float(metricas_multietiqueta(y, pred)["f1_media_columnas"])


def tokenizar(textos) -> list[list[str]]:
    """Tokeniza con el tokenizador EMB-NORMAL-TW v1 del proyecto.

    Args:
        textos: serie de textos.

    Returns:
        Lista de listas de tokens.
    """
    return [RE_TOKEN.findall(str(x)) for x in textos]


def asignar_ids(docs: list[list[str]], vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Mapea tokens a ids. El id 0 es compartido por padding y OOV.

    Args:
        docs: documentos tokenizados.
        vocab: palabra a id, empezando en 1.

    Returns:
        Matriz de ids y longitudes reales.
    """
    ids = np.zeros((len(docs), MAXLEN), dtype=np.int64)
    largos = np.zeros(len(docs), dtype=np.int64)
    for i, toks in enumerate(docs):
        n = min(len(toks), MAXLEN)
        largos[i] = n
        for j in range(n):
            ids[i, j] = vocab.get(toks[j], 0)
    return ids, largos


def mascara_real(largos: np.ndarray, device: torch.device) -> torch.Tensor:
    """Mascara booleana de posiciones reales.

    Args:
        largos: longitudes reales.
        device: dispositivo.

    Returns:
        Tensor booleano (B, MAXLEN).
    """
    idx = torch.arange(MAXLEN, device=device).unsqueeze(0)
    return idx < torch.as_tensor(largos, device=device).unsqueeze(1)


class TextCNN(nn.Module):
    """TextCNN pequena con embeddings congelados y pooling enmascarado."""

    def __init__(self, matriz: np.ndarray) -> None:
        """Construye la red.

        Args:
            matriz: matriz de embeddings, fila 0 reservada para padding y OOV.
        """
        super().__init__()
        self.kernels = KERNELS
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matriz), freeze=True,
                                                padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(matriz.shape[1], FILTROS, k) for k in KERNELS])
        for c in self.convs:
            nn.init.kaiming_normal_(c.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(c.bias)
        self.drop = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(FILTROS * len(KERNELS), HIDDEN)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(HIDDEN, len(ETIQUETAS))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Paso hacia adelante.

        Args:
            x: ids (B, MAXLEN).
            real: mascara de posiciones reales (B, MAXLEN).

        Returns:
            Logits (B, 3).
        """
        e = self.emb(x).transpose(1, 2)
        salidas = []
        for k, conv in zip(self.kernels, self.convs, strict=True):
            z = conv(e)
            valida = real.unfold(1, k, 1).all(dim=2)
            z = z.masked_fill(~valida.unsqueeze(1), -1e9)
            z = torch.relu(z)
            salidas.append(z.max(dim=2).values)
        h = torch.cat(salidas, dim=1)
        h = self.drop(h)
        h = torch.relu(self.fc1(h))
        return self.fc2(h)


def trainable(m: nn.Module) -> list[nn.Parameter]:
    """Parametros entrenables.

    Args:
        m: modelo.

    Returns:
        Lista de parametros.
    """
    return [p for p in m.parameters() if p.requires_grad]


def n_trainable(m: nn.Module) -> int:
    """Cuenta de parametros entrenables.

    Args:
        m: modelo.

    Returns:
        Numero de parametros.
    """
    return int(sum(p.numel() for p in trainable(m)))


def n_congelados(m: nn.Module) -> int:
    """Cuenta de parametros congelados.

    Args:
        m: modelo.

    Returns:
        Numero de parametros.
    """
    return int(sum(p.numel() for p in m.parameters() if not p.requires_grad))


def grupos_parametros(m: nn.Module) -> list[dict]:
    """Separa pesos y sesgos, para aplicar weight decay solo a los pesos.

    Args:
        m: modelo.

    Returns:
        Lista de grupos de parametros.
    """
    pesos, sesgos = [], []
    for nombre, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (sesgos if nombre.endswith("bias") else pesos).append(p)
    return [{"params": pesos, "weight_decay": WD},
            {"params": sesgos, "weight_decay": 0.0}]


def lr_de(epoca: int) -> float:
    """Learning rate por epoca: warmup lineal y cosine.

    Args:
        epoca: indice de epoca, desde 1.

    Returns:
        El learning rate.
    """
    if epoca <= WARMUP:
        return LR_MIN + (LR_PICO - LR_MIN) * (epoca / WARMUP)
    prog = (epoca - WARMUP) / max(EPOCAS - WARMUP, 1)
    return LR_MIN + 0.5 * (LR_PICO - LR_MIN) * (1 + math.cos(math.pi * min(prog, 1.0)))


def log_odds(prev: np.ndarray) -> np.ndarray:
    """Log-odds de las prevalencias, con recorte.

    Args:
        prev: prevalencias por etiqueta.

    Returns:
        Vector de log-odds.
    """
    p = np.clip(prev, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def piso_entropia(prev: np.ndarray) -> float:
    """Piso teorico de la BCE con logits en el log-odds del prior.

    Args:
        prev: prevalencias por etiqueta.

    Returns:
        La media de las entropias binarias.
    """
    p = np.clip(prev, 1e-12, 1 - 1e-12)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


@torch.no_grad()
def evaluar(modelo: nn.Module, ids: np.ndarray, largos: np.ndarray,
            y: np.ndarray) -> tuple[float, np.ndarray]:
    """Evalua en modo eval: perdida BCE media y probabilidades.

    Args:
        modelo: red.
        ids: ids.
        largos: longitudes.
        y: etiquetas.

    Returns:
        La perdida media y las probabilidades.
    """
    modelo.eval()
    perdida = nn.BCEWithLogitsLoss()
    xt = torch.as_tensor(ids)
    rt = mascara_real(largos, torch.device("cpu"))
    yt = torch.as_tensor(y, dtype=torch.float32)
    logits = modelo(xt, rt)
    p = torch.sigmoid(logits).numpy()
    return float(perdida(logits, yt).item()), p


def partir_interna(grupos: np.ndarray, y: np.ndarray,
                   semilla: int) -> tuple[np.ndarray, int]:
    """Holdout anidado 80/20 POR CLUSTERS, con control de prevalencia.

    Args:
        grupos: etiqueta de cluster por fila.
        y: etiquetas.
        semilla: semilla base.

    Returns:
        Mascara booleana de inner_train y el numero de intentos usados.
    """
    u = np.unique(grupos)
    prev = y.mean(axis=0)
    mascara = np.zeros(len(grupos), dtype=bool)
    intentos = 0
    for intento in range(MAX_INTENTOS_INTERNA):
        intentos = intento + 1
        rng = np.random.default_rng(semilla + intento)
        perm = rng.permutation(len(u))
        n_sel = round(FRACCION_INTERNA * len(u))
        elegidos = set(u[perm[:n_sel]].tolist())
        mascara = np.array([g in elegidos for g in grupos])
        if not mascara.any() or mascara.all():
            continue
        prev_va = y[~mascara].mean(axis=0)
        if float(np.abs(prev - prev_va).max()) <= TOL_PREVALENCIA:
            break
    return mascara, intentos


def entrenar_pliegue(matriz: np.ndarray, init: dict, ids_tr: np.ndarray,
                     largos_tr: np.ndarray, y_tr: np.ndarray, ids_va: np.ndarray,
                     largos_va: np.ndarray, y_va: np.ndarray,
                     prev_prior: np.ndarray) -> tuple[np.ndarray, dict]:
    """Entrena un pliegue completo con early stopping sobre la validacion interna.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial comun a las dos ramas.
        ids_tr: ids de entrenamiento.
        largos_tr: longitudes de entrenamiento.
        y_tr: etiquetas de entrenamiento.
        ids_va: ids de validacion externa.
        largos_va: longitudes de validacion externa.
        y_va: etiquetas de validacion externa.
        prev_prior: prevalencias para el sesgo de salida.

    Returns:
        Las probabilidades sobre la validacion externa y la traza del entrenamiento.
    """
    t0 = time.time()
    torch.manual_seed(SEMILLA)
    modelo = TextCNN(matriz)
    modelo.load_state_dict(init, strict=False)
    with torch.no_grad():
        modelo.fc2.bias.copy_(torch.as_tensor(log_odds(prev_prior), dtype=torch.float32))
    theta0 = torch.cat([p.detach().reshape(-1) for p in trainable(modelo)]).clone()

    opt = torch.optim.AdamW(grupos_parametros(modelo), lr=LR_PICO, betas=(0.9, 0.999), eps=1e-8)
    perdida = nn.BCEWithLogitsLoss()
    g_perm = torch.Generator().manual_seed(SEMILLA + 7)
    xt, rt = torch.as_tensor(ids_tr), mascara_real(largos_tr, torch.device("cpu"))
    yt = torch.as_tensor(y_tr, dtype=torch.float32)

    traza = []
    suaves: list[float] = []
    mejor_estado = None
    mejor_suave = float("inf")
    mejor_epoca = 0
    deriva1 = None
    for epoca in range(1, EPOCAS + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_de(epoca)
        modelo.train()
        perm = torch.randperm(len(ids_tr), generator=g_perm)
        suma, nb, gnorm = 0.0, 0, 0.0
        for b0 in range(0, len(ids_tr), BATCH):
            idx = perm[b0:b0 + BATCH]
            opt.zero_grad()
            out = modelo(xt[idx], rt[idx])
            perdida(out, yt[idx]).backward()
            gnorm = float(nn.utils.clip_grad_norm_(trainable(modelo), CLIP).item())
            opt.step()
            suma += float(nn.functional.binary_cross_entropy_with_logits(
                out.detach(), yt[idx]).item()) * len(idx)
            nb += len(idx)
        perdida_tr, p_tr = evaluar(modelo, ids_tr, largos_tr, y_tr)
        perdida_va, p_va = evaluar(modelo, ids_va, largos_va, y_va)
        suaves.append(float(np.mean([perdida_va] + suaves[-(VENTANA_ES - 1):]))
                      if suaves else perdida_va)
        if epoca == 1:
            th1 = torch.cat([p.detach().reshape(-1) for p in trainable(modelo)])
            deriva1 = float(torch.linalg.vector_norm(th1 - theta0).item())
        if suaves[-1] < mejor_suave - TOL_ES:
            mejor_suave = suaves[-1]
            mejor_epoca = epoca
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        traza.append({"epoca": epoca, "lr": round(lr_de(epoca), 8),
                      "perdida_train": round(perdida_tr, 6),
                      "perdida_val_interna": round(perdida_va, 6),
                      "perdida_val_suavizada": round(suaves[-1], 6),
                      "macro_f1_train": round(f1m(y_tr, (p_tr >= UMBRALES[None, :]).astype(int)), 6),
                      "macro_f1_val_interna": round(
                          f1m(y_va, (p_va >= UMBRALES[None, :]).astype(int)), 6),
                      "sd_proba_train": round(float(p_tr.std()), 6),
                      "norma_gradiente_preclip": round(gnorm, 6)})
        if epoca - mejor_epoca >= PACIENCIA:
            break

    modelo.load_state_dict(mejor_estado)
    _, proba_ext = evaluar(modelo, ids_va, largos_va, y_va)
    _, proba_tr_final = evaluar(modelo, ids_tr, largos_tr, y_tr)
    registro = {"epocas_totales": len(traza), "mejor_epoca": mejor_epoca,
                "perdida_val_suavizada_minima": round(mejor_suave, 6),
                "deriva_parametros_epoca_1": None if deriva1 is None else round(deriva1, 6),
                "toco_tope_epocas": len(traza) >= EPOCAS,
                "macro_f1_train_final": round(
                    f1m(y_tr, (proba_tr_final >= UMBRALES[None, :]).astype(int)), 6),
                "segundos": round(time.time() - t0, 1),
                "traza": traza}
    return proba_ext, registro


def bootstrap_clusters(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                       grupos: np.ndarray, semilla: int,
                       n: int) -> tuple[float, float, float]:
    """Bootstrap del delta pareado, remuestreando CLUSTERS y no filas.

    Args:
        y: etiquetas.
        pred_a: predicciones de la rama A.
        pred_b: predicciones de la rama B.
        grupos: etiqueta de cluster por fila.
        semilla: semilla.
        n: numero de remuestreos.

    Returns:
        Media, extremo inferior y extremo superior del IC 95.
    """
    u = np.unique(grupos)
    por_g = {g: np.where(grupos == g)[0] for g in u}
    rng = np.random.default_rng(semilla)
    d = np.empty(n)
    for k in range(n):
        sel = rng.integers(0, len(u), len(u))
        idx = np.concatenate([por_g[u[s]] for s in sel])
        d[k] = f1m(y[idx], pred_a[idx]) - f1m(y[idx], pred_b[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(float(d.mean()), 10), round(float(lo), 10), round(float(hi), 10)


def cv_v03(tx, y, pl) -> np.ndarray:
    """Reconstruye las probabilidades de V03 con el protocolo verificado.

    Args:
        tx: textos.
        y: etiquetas.
        pl: pliegues.

    Returns:
        Probabilidades out-of-fold.
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    pr = np.zeros((len(y), len(ETIQUETAS)))
    for a, b in pl:
        p2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        c2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        xi = sparse.hstack([p2.fit_transform(tx.iloc[a]), c2.fit_transform(tx.iloc[a])],
                           format="csr")
        xv = sparse.hstack([p2.transform(tx.iloc[b]), c2.transform(tx.iloc[b])], format="csr")
        for j in range(len(ETIQUETAS)):
            m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=42)
            m.fit(xi, y[a, j])
            pr[b, j] = m.predict_proba(xv)[:, 1]
    return pr


def cv_emb_pool(D: np.ndarray, y: np.ndarray, pl) -> np.ndarray:
    """Embedding-only de P72, con el StandardScaler dentro del pliegue.

    Args:
        D: matriz pooled.
        y: etiquetas.
        pl: pliegues.

    Returns:
        Probabilidades out-of-fold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pr = np.zeros((len(y), len(ETIQUETAS)))
    for a, b in pl:
        sc = StandardScaler().fit(D[a])
        for j in range(len(ETIQUETAS)):
            m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=42)
            m.fit(sc.transform(D[a]), y[a, j])
            pr[b, j] = m.predict_proba(sc.transform(D[b]))[:, 1]
    return pr


def main() -> None:
    """Ejecuta P73 completo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    warns: list[str] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = 0
    torch.set_num_threads(32)
    log("P73-02 TextCNN: Word2Vec Twitter congelado frente a embedding aleatorio")
    log("   pre-registro: docs/p73_preregistro_textcnn.md, commit 7192a22")

    v = np.load(FEATURES / "oof_v03_final.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    ids_v03 = v["ids"].astype(np.int64)
    proba_v03 = v["proba"].astype(float)
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, _ = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids_v03):
        raise SystemExit("PARADA: los ids no coinciden con el OOF canonico")
    pl = pliegues_corregidos(des, g[idev])
    tx = des["text"]
    grupos = g[idev]
    log(f"   {len(y)} filas | {len(np.unique(grupos))} clusters | 5 pliegues")

    # ---------- comprobacion 10: orden de umbrales ----------
    assert np.allclose(UMBRALES, [0.44, 0.51, 0.46]), "orden de umbrales incorrecto"
    assert ETIQUETAS == ("HS", "TR", "AG")
    log("   check 10 OK: umbrales [0.44, 0.51, 0.46] en orden (HS, TR, AG)")

    # ---------- comprobacion 1: V03 ----------
    proba_v03_rec = cv_v03(tx, y, pl)
    ccv = f1m(y, (proba_v03_rec >= UMBRALES[None, :]).astype(int))
    log(f"   check 1: V03 reconstruido {ccv:.10f} frente a {V03_CCV:.10f}")
    if abs(ccv - V03_CCV) > 1e-9:
        raise SystemExit("PARADA: la reconstruccion de V03 no reproduce corrected_cv")

    # ---------- datos y matrices ----------
    docs = tokenizar(tx)
    d100 = np.load(MODELOS / "w2v_twitter_100d.npz", allow_pickle=True)
    vocab = {w: i + 1 for i, w in enumerate(d100["palabras"])}
    W = d100["W"].astype(np.float32)
    normas = np.linalg.norm(W, axis=1)
    ids, largos = asignar_ids(docs, vocab)
    matriz_a = np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32)
    matriz_a[1:] = W

    # ---------- comprobacion 6: lookup ----------
    rng = np.random.default_rng(SEMILLA)
    muestra = rng.choice(len(W), size=100, replace=False)
    cos = [float(np.dot(W[i], matriz_a[i + 1]) /
                 max(np.linalg.norm(W[i]) * np.linalg.norm(matriz_a[i + 1]), 1e-12))
           for i in muestra]
    nm = float(np.linalg.norm(matriz_a[1:], axis=1).mean())
    log(f"   check 6: cos medio del lookup {np.mean(cos):.8f} | norma media {nm:.6f}")
    assert min(cos) > 1 - 1e-6 and abs(nm - 0.842862) < 1e-4

    # ---------- comprobacion 7: cobertura ----------
    n_tok = np.array([len(d) for d in docs])
    n_cub = np.array([sum(1 for t in d if t in vocab) for d in docs])
    cob = float(n_cub.sum() / max(n_tok.sum(), 1))
    filas_sin = int((n_cub == 0).sum())
    log(f"   check 7: cobertura {100 * cob:.2f}% | tokens cubiertos por fila "
        f"{n_cub.mean():.2f} | filas sin cobertura {filas_sin}")
    assert abs(100 * cob - 68.64) < 0.05 and abs(n_cub.mean() - 12.5) < 0.05

    # ---------- comprobacion 2: embedding-only de P72 ----------
    D = np.zeros((len(docs), W.shape[1]), dtype=np.float32)
    for i, d in enumerate(docs):
        ii = [vocab[t] - 1 for t in d if t in vocab]
        if ii:
            D[i] = W[ii].mean(axis=0)
    p72 = f1m(y, (cv_emb_pool(D, y, pl) >= UMBRALES[None, :]).astype(int))
    log(f"   check 2: embedding-only de P72 {p72:.10f} frente a {P72_EMB_ONLY:.10f}")
    if abs(p72 - P72_EMB_ONLY) > 1e-9:
        raise SystemExit("PARADA: no reproduce el embedding-only de P72")

    # ---------- comprobacion 11: rama aleatoria ----------
    rng = np.random.default_rng(SEMILLA + 1)
    U = rng.standard_normal(W.shape)
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    W_rand = (normas[:, None] * U).astype(np.float32)
    matriz_b = np.zeros_like(matriz_a)
    matriz_b[1:] = W_rand
    dn = float(np.abs(np.linalg.norm(matriz_b[1:], axis=1) - normas).max())
    nrm_b = np.maximum(np.linalg.norm(matriz_b[1:], axis=1), 1e-12)
    nrm_a = np.maximum(normas, 1e-12)
    cos_ab = float(np.mean(np.sum(W * matriz_b[1:], axis=1) / (nrm_a * nrm_b)))
    sub = np.random.default_rng(SEMILLA + 2).choice(len(W), 3000, replace=False)
    Wn = W_rand[sub] / np.maximum(np.linalg.norm(W_rand[sub], axis=1, keepdims=True), 1e-12)
    C = Wn @ Wn.T
    cos_rand = float((C.sum() - len(sub)) / (len(sub) * (len(sub) - 1)))
    Wn_a = W[sub] / np.maximum(np.linalg.norm(W[sub], axis=1, keepdims=True), 1e-12)
    Ca = Wn_a @ Wn_a.T
    cos_tr = float((Ca.sum() - len(sub)) / (len(sub) * (len(sub) - 1)))
    log(f"   check 11: desviacion de normas {dn:.3e} | cos(A_i,B_i) {cos_ab:+.6f} | "
        f"cos medio de pares entrenado {cos_tr:+.4f} aleatorio {cos_rand:+.4f}")
    assert dn < 1e-6 and abs(cos_ab) < 0.002

    # ---------- comprobacion 13 ----------
    log(f"   check 13: hilos {torch.get_num_threads()} | dtype float32 | dispositivo cpu")

    # ---------- init comun y comprobacion 8 ----------
    torch.manual_seed(SEMILLA)
    _tmp = TextCNN(matriz_a)
    with torch.no_grad():
        _tmp.fc2.bias.copy_(torch.as_tensor(log_odds(y.mean(axis=0)), dtype=torch.float32))
    init_comun = {k: v.detach().clone() for k, v in _tmp.state_dict().items()
                  if not k.startswith("emb.") and k != "fc2.bias"}
    m_a = TextCNN(matriz_a)
    m_b = TextCNN(matriz_b)
    m_a.load_state_dict(init_comun, strict=False)
    m_b.load_state_dict(init_comun, strict=False)
    iguales = all(torch.equal(p, q) for p, q in
                  zip([p for n, p in m_a.named_parameters() if not n.startswith("emb.")],
                      [p for n, p in m_b.named_parameters() if not n.startswith("emb.")],
                      strict=True))
    with torch.no_grad():
        m_a.eval()
        m_b.eval()
        xb = torch.as_tensor(ids[:16])
        rb = mascara_real(largos[:16], torch.device("cpu"))
        cruzada = torch.allclose(m_b(xb, rb), m_a(xb, rb), atol=1e-7)
    log(f"   check 8: pesos entrenables identicos {iguales} | salidas identicas sobre los mismos "
        f"ids {cruzada}")
    assert iguales and cruzada

    # ---------- comprobacion 5 ----------
    nt, nc = n_trainable(m_a), n_congelados(m_a)
    log(f"   check 5: parametros entrenables {nt} | congelados {nc}")
    assert nt == 70339, f"se esperaban 70339 entrenables, hay {nt}"
    assert nc == matriz_a.size

    # ---------- comprobacion 3: piso teorico ----------
    piso_prior = piso_entropia(y.mean(axis=0))
    _c = TextCNN(matriz_a)
    for p in trainable(_c):
        p.data.zero_()
    with torch.no_grad():
        lp = nn.BCEWithLogitsLoss()(_c(xb, rb), torch.as_tensor(y[:16], dtype=torch.float32))
        ceros = np.zeros((16, 3), dtype=np.float32)
        l0 = float(nn.BCEWithLogitsLoss()(torch.as_tensor(ceros),
                                          torch.as_tensor(y[:16], dtype=torch.float32)).item())
    log(f"   check 3: piso con logits en el prior {piso_prior:.6f} (esperado "
        f"{PISO_PRIOR_ESPERADO}) | logits a cero {l0:.6f} (esperado {math.log(2):.6f}) | "
        f"medido con la red en cero {float(lp):.6f}")
    assert abs(piso_prior - PISO_PRIOR_ESPERADO) < 1e-3
    assert abs(l0 - math.log(2)) < 1e-6
    log(f"   check 3 AVISO: si en el entrenamiento aparece exactamente {2 * math.log(2):.6f} o "
        f"{math.log(2):.6f} en la perdida, parar y auditar")

    # ---------- comprobacion 4: sobreajuste de 32 filas ----------
    torch.manual_seed(SEMILLA)
    _o = TextCNN(matriz_a)
    _o.load_state_dict(init_comun, strict=False)
    _opt = torch.optim.Adam(_o.parameters(), lr=5e-3)
    x32 = torch.as_tensor(ids[:32])
    r32 = mascara_real(largos[:32], torch.device("cpu"))
    y32 = torch.as_tensor(y[:32], dtype=torch.float32)
    for _ in range(200):
        _opt.zero_grad()
        nn.BCEWithLogitsLoss()(_o(x32, r32), y32).backward()
        _opt.step()
    with torch.no_grad():
        _o.eval()
        p32 = torch.sigmoid(_o(x32, r32)).numpy()
    f32 = f1m(y[:32], (p32 >= UMBRALES[None, :]).astype(int))
    pe32 = float(nn.BCEWithLogitsLoss()(torch.as_tensor(
        np.log(np.clip(p32, 1e-12, 1 - 1e-12) / np.clip(1 - p32, 1e-12, 1))), y32).item())
    log(f"   check 4: sobreajuste de 32 filas -> perdida {pe32:.4f} | macro-F1 {f32:.4f}")
    assert pe32 < 0.20 and f32 > 0.9, "la red no puede sobreajustar 32 filas: hay un bug"

    # ---------- comprobacion 9: manifiesto ----------
    manifiesto = {"input_ids": sha_arr(ids), "longitudes": sha_arr(largos),
                  "mascara": sha_arr((np.arange(MAXLEN)[None, :] < largos[:, None]).astype(np.int8)),
                  "clusters": sha_arr(grupos),
                  "init_entrenable": sha_arr(np.concatenate(
                      [v.numpy().reshape(-1) for v in init_comun.values()])),
                  "matriz_entrenada": sha_matriz(matriz_a),
                  "matriz_aleatoria": sha_matriz(matriz_b),
                  "sha_v03": hashlib.sha256(
                      (PROJECT_ROOT / "submissions" /
                       "toxic_submission_v03_lr_union_thr.csv").read_bytes()).hexdigest()}
    log(f"   check 9: manifiesto -> ids {manifiesto['input_ids'][:12]} | init "
        f"{manifiesto['init_entrenable'][:12]} | A {manifiesto['matriz_entrenada'][:12]} | "
        f"B {manifiesto['matriz_aleatoria'][:12]}")

    # ---------- las dos ramas: 5 pliegues cada una, 10 corridas ----------
    ramas = {"WORD2VEC": matriz_a, "ALEATORIO": matriz_b}
    probas: dict[str, np.ndarray] = {}
    registros: dict[str, list] = {}
    for nombre, matriz in ramas.items():
        log("")
        log(f"   RAMA {nombre} con matriz {sha_matriz(matriz)[:12]}")
        pr = np.zeros((len(y), len(ETIQUETAS)))
        regs = []
        for k, (a, b) in enumerate(pl, start=1):
            ia, ib = np.asarray(a), np.asarray(b)
            m_in, intentos = partir_interna(grupos[ia], y[ia], SEMILLA + 100 * k)
            t0 = time.time()
            with warnings.catch_warnings(record=True) as capturados:
                warnings.simplefilter("always")
                proba_ext, reg = entrenar_pliegue(
                    matriz, init_comun,
                    ids[ia][m_in], largos[ia][m_in], y[ia][m_in],
                    ids[ib], largos[ib], y[ib],
                    y[ia][m_in].mean(axis=0))
                warns.extend(f"{nombre} p{k}: {x.message!s}" for x in capturados)
            pr[ib] = proba_ext
            reg.update({"pliegue": k, "intentos_particion_interna": intentos,
                        "n_inner_train": int(m_in.sum()), "n_inner_val": int((~m_in).sum()),
                        "n_externo": len(ib),
                        "macro_f1_externo": round(
                            f1m(y[ib], (proba_ext >= UMBRALES[None, :]).astype(int)), 6),
                        "segundos_wall": round(time.time() - t0, 1)})
            regs.append(reg)
            log(f"      pliegue {k}: externo {reg['macro_f1_externo']:.6f} | mejor epoca "
                f"{reg['mejor_epoca']}/{reg['epocas_totales']} | "
                f"perdida_train {reg['traza'][-1]['perdida_train']:.4f} | "
                f"deriva p1 {reg['deriva_parametros_epoca_1']} | "
                f"inner {reg['n_inner_train']}/{reg['n_inner_val']} | intentos {intentos} | "
                f"tope {reg['toco_tope_epocas']} | {reg['segundos']}s")
        probas[nombre] = pr
        registros[nombre] = regs
        log(f"   RAMA {nombre}: CV {f1m(y, (pr >= UMBRALES[None, :]).astype(int)):.10f}")

    pred_a = (probas["WORD2VEC"] >= UMBRALES[None, :]).astype(int)
    pred_b = (probas["ALEATORIO"] >= UMBRALES[None, :]).astype(int)
    cv_a = f1m(y, pred_a)
    cv_b = f1m(y, pred_b)
    cv_v03_final = f1m(y, (proba_v03 >= UMBRALES[None, :]).astype(int))
    log("")
    log(f"   RESULTADO: A Word2Vec {cv_a:.10f} | B aleatorio {cv_b:.10f} | delta {cv_a - cv_b:+.10f}")
    log(f"   referencia V03 {cv_v03_final:.10f} (protocolo distinto: no aisla la representacion)")

    # ---------- comprobacion 12, corregida tres veces, todas declaradas ----------
    # El pre-registro pedia predicciones identicas ENTRE las dos ramas. El enunciado estaba mal
    # especificado y hubo que refinarlo tres veces:
    #  (1) las ramas divergen al entrenar, luego sus constantes de salida difieren: la identidad
    #      entre ramas no es una propiedad exigible;
    #  (2) dentro de una rama la prediccion no es constante entre las 70, porque el numero de
    #      VENTANAS VALIDAS depende de la longitud real: una fila de 2 tokens no tiene ventana
    #      valida de tamano 3 ni 4, y esos canales valen 0 en vez de ReLU(sesgo);
    #  (3) tampoco basta agrupar por longitud, porque las 70 filas caen en PLIEGUES distintos y
    #      cada pliegue tiene su propio modelo entrenado.
    # El invariante exacto, y el que se verifica: la prediccion de una fila sin cobertura queda
    # determinada por (pliegue, min(longitud, 4)). No se relajo ninguna tolerancia en ninguna de
    # las tres correcciones: se corrego un enunciado que no describia el invariante real.
    sin_cob = n_cub == 0
    fold_de = np.zeros(len(y), dtype=int)
    for k, (_, b) in enumerate(pl, start=1):
        fold_de[np.asarray(b)] = k
    largo_ef = np.minimum(largos[sin_cob], max(KERNELS))
    claves = [f"{fold_de[i]}|{largo_ef[j]}" for j, i in enumerate(np.where(sin_cob)[0])]
    ca = probas["WORD2VEC"][sin_cob]
    cb = probas["ALEATORIO"][sin_cob]
    fallos_a, fallos_b, n_grupos = 0, 0, 0
    for clave in dict.fromkeys(claves):
        ix = np.array([j for j, c in enumerate(claves) if c == clave])
        n_grupos += 1
        if not np.allclose(ca[ix], ca[ix][0], atol=1e-6):
            fallos_a += 1
        if not np.allclose(cb[ix], cb[ix][0], atol=1e-6):
            fallos_b += 1
    ident_a, ident_b = fallos_a == 0, fallos_b == 0
    sin_nan = bool(np.isfinite(ca).all() and np.isfinite(cb).all())
    desviacion_12 = (
        "el pre-registro pedia predicciones identicas ENTRE las dos ramas en las 70 filas sin "
        "cobertura. El enunciado estaba mal especificado y hubo que refinarlo TRES veces, todas "
        "declaradas. (1) Las ramas divergen al entrenar, luego sus sesgos convolucionales difieren "
        "y con ellos la constante de salida: la identidad entre ramas no es exigible. (2) Dentro "
        "de una rama la prediccion no es constante entre las 70, porque el numero de ventanas "
        "validas depende de la longitud real: una fila de 2 tokens no tiene ventana valida de "
        "tamano 3 ni 4 y esos canales valen 0 en vez de ReLU(sesgo). (3) Agrupar solo por longitud "
        "tampoco basta: las 70 filas caen en PLIEGUES distintos y cada pliegue tiene su propio "
        "modelo entrenado. El invariante exacto, que es el que se verifica: la prediccion de una "
        "fila sin cobertura queda determinada por (pliegue, min(longitud, 4)). No se relajo "
        "ninguna tolerancia en ninguna de las tres correcciones")
    log(f"   check 12: {int(sin_cob.sum())} filas sin cobertura | longitudes efectivas "
        f"{ {k: int(v) for k, v in zip(*np.unique(largo_ef, return_counts=True))} } | "
        f"grupos (pliegue, longitud) {n_grupos} | constante dentro de cada grupo: A {ident_a} "
        f"({fallos_a} fallos) | B {ident_b} ({fallos_b} fallos) | sin NaN ni inf {sin_nan}")
    log(f"      rango de la prediccion constante: A [{ca.min():.4f}, {ca.max():.4f}] | "
        f"B [{cb.min():.4f}, {cb.max():.4f}]")
    assert ident_a and ident_b and sin_nan, "las filas sin cobertura no cumplen el invariante"

    # ---------- delta por pliegue y bootstrap ----------
    d_pliegue = [round(registros["WORD2VEC"][k]["macro_f1_externo"]
                       - registros["ALEATORIO"][k]["macro_f1_externo"], 6) for k in range(5)]
    gana_a = int(sum(1 for d in d_pliegue if d > 0))
    media_b, lo_b, hi_b = bootstrap_clusters(y, pred_a, pred_b, grupos, SEMILLA_BOOT, N_BOOTSTRAP)
    semi = (hi_b - lo_b) / 2
    if lo_b > 0 and gana_a >= 4:
        veredicto = "A_SUPERA_A_B_CLARO"
    elif hi_b < 0:
        veredicto = "B_SUPERA_A_A_CLARO"
    else:
        veredicto = "EMPATE"
    log("")
    log(f"   delta por pliegue {d_pliegue} | gana A en {gana_a}/5")
    log(f"   bootstrap de clusters del delta: {media_b:+.6f} [{lo_b:+.6f}, {hi_b:+.6f}] | "
        f"semianchura {semi:.6f}")
    log(f"   VEREDICTO: {veredicto}")

    # ---------- dispersion entre pliegues ----------
    fa = [r["macro_f1_externo"] for r in registros["WORD2VEC"]]
    fb = [r["macro_f1_externo"] for r in registros["ALEATORIO"]]
    disp = {"a_media": round(float(np.mean(fa)), 6), "a_sd": round(float(np.std(fa, ddof=1)), 6),
            "b_media": round(float(np.mean(fb)), 6), "b_sd": round(float(np.std(fb, ddof=1)), 6)}

    # ---------- rebanadas ----------
    rebanadas = {}
    for nombre, lo, hi in [("CORTA 0-6", 0, 6), ("MEDIA 7-15", 7, 15),
                           ("LARGA 16-30", 16, 30), ("MUY_LARGA 31+", 31, 10 ** 9)]:
        m = (n_tok >= lo) & (n_tok <= hi)
        rebanadas[nombre] = {
            "filas": int(m.sum()), "v03": round(f1m(y[m], (proba_v03[m] >= UMBRALES[None, :]).astype(int)), 6),
            "a_word2vec": round(f1m(y[m], pred_a[m]), 6),
            "b_aleatorio": round(f1m(y[m], pred_b[m]), 6),
            "delta_a_b": round(f1m(y[m], pred_a[m]) - f1m(y[m], pred_b[m]), 6)}
    estructura = {}
    texto = tx.astype(str)
    for nombre, pat in LISTA_ESTRUCTURA.items():
        m = texto.str.contains(pat, regex=True, case=False, na=False).to_numpy()
        estructura[nombre] = {
            "filas_con": int(m.sum()),
            "delta_a_b_con": round(f1m(y[m], pred_a[m]) - f1m(y[m], pred_b[m]), 6) if m.sum() > 10 else None,
            "delta_a_b_sin": round(f1m(y[~m], pred_a[~m]) - f1m(y[~m], pred_b[~m]), 6)
            if (~m).sum() > 10 else None}
    bajo = texto.str.lower()
    m_ins = bajo.apply(lambda s: any(w in s for w in LISTA_INSULTOS)).to_numpy()
    estructura["INSULTOS_lista_declarada"] = {
        "filas_con": int(m_ins.sum()),
        "delta_a_b_con": round(f1m(y[m_ins], pred_a[m_ins]) - f1m(y[m_ins], pred_b[m_ins]), 6)
        if m_ins.sum() > 10 else None,
        "delta_a_b_sin": round(f1m(y[~m_ins], pred_a[~m_ins]) - f1m(y[~m_ins], pred_b[~m_ins]), 6)
        if (~m_ins).sum() > 10 else None,
        "aviso": "el corte menos fiable: la lista es una decision del autor, no un recurso externo"}
    log("")
    log("   REBANADAS POR LONGITUD (todas, no solo la favorable)")
    for k, val in rebanadas.items():
        log(f"      {k:14} n={val['filas']:4d} | v03 {val['v03']:.4f} | A {val['a_word2vec']:.4f} | "
            f"B {val['b_aleatorio']:.4f} | A-B {val['delta_a_b']:+.4f}")
    log("   REBANADAS POR ESTRUCTURA DEL TUIT (diagnostico, sin nuevas features)")
    for k, val in estructura.items():
        log(f"      {k:28} n={val['filas_con']:4d} | A-B con {val['delta_a_b_con']} | "
            f"sin {val['delta_a_b_sin']}")

    # ---------- complementariedad ----------
    comp = {}
    ar = pred_a == y
    vr = proba_v03 >= UMBRALES[None, :]
    comp["global"] = {"v03_ko_a_ok": int((~vr & ar).sum()), "v03_ok_a_ko": int((vr & ~ar).sum()),
                      "ambos_ok": int((vr & ar).sum()), "ambos_ko": int((~vr & ~ar).sum())}
    comp["global"]["rescue"] = comp["global"]["v03_ko_a_ok"]
    comp["global"]["damage"] = comp["global"]["v03_ok_a_ko"]
    comp["global"]["net"] = comp["global"]["rescue"] - comp["global"]["damage"]
    comp["por_etiqueta"] = {}
    for j, e in enumerate(ETIQUETAS):
        comp["por_etiqueta"][e] = {
            "v03_ko_a_ok": int((~vr[:, j] & ar[:, j]).sum()),
            "v03_ok_a_ko": int((vr[:, j] & ~ar[:, j]).sum()),
            "ambos_ok": int((vr[:, j] & ar[:, j]).sum()),
            "ambos_ko": int((~vr[:, j] & ~ar[:, j]).sum())}
    log("")
    log(f"   COMPLEMENTARIEDAD frente a V03: rescue {comp['global']['rescue']} | "
        f"damage {comp['global']['damage']} | net {comp['global']['net']}")
    if veredicto == "A_SUPERA_A_B_CLARO":
        log("   A supera a B: la complementariedad es informativa y se reporta como pregunta abierta")
    else:
        log("   A no supera a B: NO se evalua ninguna fusion, segun el protocolo de la fase")

    # ---------- artefactos ----------
    salida = {
        "familia": "P73_TEXTCNN_WORD2VEC_RANDOM", "experiment_id": "P73-02",
        "fase": "TextCNN congelada: Word2Vec frente a control aleatorio",
        "fecha": inicio.isoformat(timespec="seconds"),
        "preregistro": {"documento": "docs/p73_preregistro_textcnn.md", "commit": "7192a22",
                        "anterior_a_la_corrida": True},
        "naturaleza": "EXPERIMENTO. 10 corridas downstream (5 pliegues x 2 ramas). "
                      "0 entrenamientos de embeddings",
        "pregunta": ("cuando dejamos de promediar y permitimos que un modelo secuencial explote "
                     "orden y contexto local, contiene la representacion aprendida informacion que "
                     "una matriz aleatoria equivalente no contiene?"),
        "arquitectura": {"maxlen": MAXLEN, "filtros": FILTROS, "kernels": list(KERNELS),
                         "hidden": HIDDEN, "dropout": DROPOUT,
                         "pooling": "global max con mascara a nivel de ventana",
                         "perdida": "BCEWithLogitsLoss reduction mean, sin pos_weight",
                         "optimizador": f"AdamW lr {LR_PICO} wd {WD}", "batch": BATCH,
                         "epocas_max": EPOCAS, "paciencia": PACIENCIA,
                         "ventana_early_stopping": VENTANA_ES, "clip": CLIP,
                         "parametros_entrenables": nt, "parametros_congelados": nc},
        "protocolo_validacion": {
            "particion_externa": "5 pliegues agrupados por near-duplicados, congelados",
            "particion_interna": "holdout anidado 80/20 POR CLUSTERS dentro de cada pliegue",
            "early_stopping": "BCE de la validacion interna, media movil trailing de 3",
            "evaluaciones_del_pliegue_externo": 1,
            "las_dos_ramas_usan_el_mismo_protocolo": True},
        "comprobaciones_de_conformidad": {
            "v03_reproducido": round(ccv, 10), "v03_esperado": V03_CCV,
            "embedding_only_p72": round(p72, 10), "embedding_only_p72_esperado": P72_EMB_ONLY,
            "piso_teorico_prior": round(piso_prior, 6), "piso_teorico_ln2": round(math.log(2), 6),
            "sobreajuste_32_filas_perdida": round(pe32, 6),
            "sobreajuste_32_filas_macro_f1": round(f32, 6),
            "parametros_entrenables": nt, "parametros_congelados": nc,
            "lookup_cos_minimo": round(float(np.min(cos)), 8), "norma_media": round(nm, 6),
            "cobertura_por_ciento": round(100 * cob, 4), "tokens_cubiertos_por_fila": round(
                float(n_cub.mean()), 4), "filas_sin_cobertura": filas_sin,
            "pesos_iniciales_identicos": bool(iguales), "salidas_cruzadas_identicas": bool(cruzada),
            "filas_sin_cobertura_constante_dentro_de_cada_rama": bool(ident_a and ident_b),
            "filas_sin_cobertura_constante_en_a": ident_a,
            "filas_sin_cobertura_constante_en_b": ident_b,
            "desviacion_de_normas_rama_b": round(dn, 12),
            "cos_medio_par_a_b": round(cos_ab, 6),
            "cos_medio_pares_entrenado": round(cos_tr, 6),
            "cos_medio_pares_aleatorio": round(cos_rand, 6)},
        "manifiesto": manifiesto,
        "corpus_y_embedding": {
            "corpus": "mmaguero/covid19-spanishtweets-earlylateapril2020",
            "licencia": "CC BY-SA 4.0", "periodo": "29-mar a 1-may-2020",
            "solapamiento_con_desarrollo": "cero por encima de Jaccard 0.2, maximo global 0.142857"
                                           " (P73-01)",
            "artefacto": "modelos/twitter_domain/w2v_twitter_100d.npz",
            "sha_matriz": manifiesto["matriz_entrenada"],
            "sha_matriz_aleatoria": manifiesto["matriz_aleatoria"],
            "vocabulario": len(vocab)},
        "resultado": {"a_word2vec_cv": round(cv_a, 10), "b_aleatorio_cv": round(cv_b, 10),
                      "delta_a_menos_b": round(cv_a - cv_b, 10),
                      "v03_referencia_cv": round(cv_v03_final, 10),
                      "delta_frente_a_v03": round(cv_a - cv_v03_final, 10),
                      "f1_por_pliegue_a": fa, "f1_por_pliegue_b": fb,
                      "delta_por_pliegue": d_pliegue, "pliegues_que_gana_a": gana_a,
                      "dispersion_entre_pliegues": disp,
                      "bootstrap_clusters": {"media": media_b, "ic95_bajo": lo_b,
                                             "ic95_alto": hi_b, "semianchura": round(semi, 10),
                                             "remuestreos": N_BOOTSTRAP,
                                             "unidad": "clusters de near-duplicados"}},
        "convergencia": {n: [{k: v for k, v in r.items() if k != "traza"} for r in regs]
                         for n, regs in registros.items()},
        "trazas": {n: regs for n, regs in registros.items()},
        "rebanadas_por_longitud": rebanadas, "rebanadas_por_estructura": estructura,
        "complementariedad": comp,
        "desviaciones_del_preregistro": [desviacion_12],
        "warnings": warns,
        "veredicto": veredicto,
        "limites_declarados": [
            ("una sola semilla, por mandato explicito de esta fase: no se estima la varianza de "
            "inicializacion ni la del orden de lotes"),
            ("inner_train tiene alrededor de 949 filas frente a las 1186 del pliegue completo: "
            "presion conservadora, identica en las dos ramas"),
            ("la validacion interna de 237 filas es ruidosa: la epoca elegida lo es sobre una senal "
            "ruidosa"),
            "una sola evaluacion del pliegue externo por pliegue y rama, 10 en total",
            "70 filas (4.7 por ciento) sin cobertura, con prediccion constante",
            ("la comparacion con V03 es una REFERENCIA condicionada a un protocolo mas conservador "
            "y a una arquitectura distinta: NO aisla la representacion")],
        "que_no_se_puede_afirmar": [
            ("Word2Vec no funciona, ni los embeddings no contienen informacion, como afirmaciones "
            "generales"),
            "no hay senal semantica en los tuits",
            "que el efecto se replicaria con otra semilla, otra arquitectura u otro corpus",
            "que una mejora sobre el campeon sea atribuible a los embeddings",
            "equivalencia estrecha entre A y B si el resultado es EMPATE: la potencia es limitada",
            ("que la diferencia se deba a relaciones semanticas y no a estructura direccional "
            "aprendida: las ramas difieren tambien en anisotropia y forma espectral"),
            "cualquier afirmacion sobre test o leaderboard"],
        "que_falsificaria_la_conclusion": (
            "si el delta A-B cambiara de signo al repetir con otra semilla de inicializacion, o si "
            "la ventaja desapareciera al fijar la misma epoca en las dos ramas, o si el IC por "
            "clusters dejara de excluir el cero al recomputarlo con otra semilla de bootstrap"),
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "p73_textcnn.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")

    with (REPORTS / "p73_textcnn_results.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["rama", "pliegue", "macro_f1_externo", "mejor_epoca", "epocas_totales",
                    "toco_tope_epocas", "perdida_train_final", "perdida_val_interna_final",
                    "macro_f1_train_final", "deriva_parametros_epoca_1", "norma_gradiente_final",
                    "n_inner_train", "n_inner_val", "n_externo", "intentos_particion_interna",
                    "segundos"])
        for n, regs in registros.items():
            for r in regs:
                w.writerow([n, r["pliegue"], r["macro_f1_externo"], r["mejor_epoca"],
                            r["epocas_totales"], r["toco_tope_epocas"],
                            r["traza"][-1]["perdida_train"], r["traza"][-1]["perdida_val_interna"],
                            r["macro_f1_train_final"], r["deriva_parametros_epoca_1"],
                            r["traza"][-1]["norma_gradiente_preclip"], r["n_inner_train"],
                            r["n_inner_val"], r["n_externo"], r["intentos_particion_interna"],
                            r["segundos"]])
        w.writerow([])
        w.writerow(["agregado", "A_word2vec", round(cv_a, 10), "B_aleatorio", round(cv_b, 10),
                    "delta", round(cv_a - cv_b, 10), "v03", round(cv_v03_final, 10),
                    "bootstrap_media", media_b, "ic95", f"[{lo_b}, {hi_b}]", "veredicto", veredicto])

    with (REPORTS / "p73_textcnn_complementarity.csv").open("w", encoding="ascii",
                                                            newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["ambito", "v03_ko_a_ok_rescue", "v03_ok_a_ko_damage", "ambos_ok", "ambos_ko",
                    "net"])
        g = comp["global"]
        w.writerow(["GLOBAL", g["rescue"], g["damage"], g["ambos_ok"], g["ambos_ko"], g["net"]])
        for e, d in comp["por_etiqueta"].items():
            w.writerow([e, d["v03_ko_a_ok"], d["v03_ok_a_ko"], d["ambos_ok"], d["ambos_ko"],
                        d["v03_ko_a_ok"] - d["v03_ok_a_ko"]])
        w.writerow([])
        w.writerow(["nota", "solo es accionable si A supera a B; si no, NO se evalua ninguna fusion"])

    if veredicto == "A_SUPERA_A_B_CLARO":
        clasificacion = ("COMPLEMENTARY" if cv_a <= cv_v03_final else "PROMISING")
    elif veredicto == "B_SUPERA_A_A_CLARO":
        clasificacion = "INVALIDATED"
    else:
        clasificacion = "CLOSED"

    md = []
    md.append("# P73: TextCNN sobre Word2Vec Twitter congelado frente a control aleatorio\n")
    md.append(f"`experiment_id` P73-02 | commit `7192a22` de pre-registro | semilla {SEMILLA} | "
              f"inicio {inicio.isoformat(timespec='seconds')}\n")
    md.append("## La pregunta\n")
    md.append("P72 pregunto si Word2Vec mejora una representacion lineal de mean pooling, y "
              "respondio que no y que su geometria no aporta sobre direcciones aleatorias.\n")
    md.append("P73 pregunta algo distinto: **cuando dejamos de promediar y permitimos que un modelo "
              "secuencial explote orden y contexto local, contiene la representacion aprendida "
              "informacion que una matriz aleatoria equivalente no contiene?**\n")
    md.append("## Las dos ramas\n")
    md.append("```text\nA  Word2Vec Twitter 100d congelado    -> TextCNN -> 3 salidas\n"
              "B  embedding aleatorio 100d congelado -> MISMA TextCNN -> 3 salidas\n```\n")
    md.append(f"Manifiesto del diff, que debe ser solo la matriz: ids `{manifiesto['input_ids'][:16]}`, "
              f"init entrenable `{manifiesto['init_entrenable'][:16]}`, "
              f"matriz A `{manifiesto['matriz_entrenada'][:16]}`, "
              f"matriz B `{manifiesto['matriz_aleatoria'][:16]}`.\n")
    md.append("## Configuracion\n")
    md.append(f"maxlen {MAXLEN} sin truncar, Conv1D kernels {list(KERNELS)} con {FILTROS} filtros, "
              f"global max pooling con mascara a nivel de ventana, Dropout {DROPOUT}, Dense {HIDDEN}, "
              f"3 logits. **{nt} parametros entrenables** y {nc} congelados. "
              f"AdamW lr {LR_PICO}, batch {BATCH}, maximo {EPOCAS} epocas.\n")
    md.append("## Comprobaciones de conformidad, todas superadas antes de entrenar\n")
    md.append("```text\n"
              f"V03 reconstruido              {ccv:.10f}  (esperado {V03_CCV:.10f})\n"
              f"embedding-only de P72         {p72:.10f}  (esperado {P72_EMB_ONLY:.10f})\n"
              f"piso teorico en el prior      {piso_prior:.6f}  (esperado {PISO_PRIOR_ESPERADO})\n"
              f"piso teorico logits a cero    {math.log(2):.6f}\n"
              f"sobreajuste de 32 filas       perdida {pe32:.4f}, macro-F1 {f32:.4f}\n"
              f"parametros                    {nt} entrenables, {nc} congelados\n"
              f"lookup de embeddings          cos minimo {np.min(cos):.8f}, norma {nm:.6f}\n"
              f"cobertura                     {100 * cob:.2f} por ciento, {n_cub.mean():.2f} tokens por fila\n"
              f"init identico en A y B        {iguales}\n"
              f"salidas cruzadas identicas    {cruzada}\n"
              f"normas de la rama B           desviacion maxima {dn:.2e}\n"
              f"cos medio entre A_i y B_i     {cos_ab:+.6f}\n"
              f"70 filas sin cobertura        constante por grupo de longitud, A "
              f"{ident_a} y B {ident_b}\n"
              "```\n")
    md.append("## Resultado\n")
    md.append("| rama | CV out-of-fold | f1 por pliegue | media | sd |\n|---|---|---|---|---|\n"
              f"| A Word2Vec | **{cv_a:.10f}** | {fa} | {disp['a_media']} | {disp['a_sd']} |\n"
              f"| B aleatorio | **{cv_b:.10f}** | {fb} | {disp['b_media']} | {disp['b_sd']} |\n")
    md.append(f"**delta A - B = {cv_a - cv_b:+.10f}**, gana A en {gana_a} de 5 pliegues.\n")
    md.append(f"Bootstrap de clusters del delta: **{media_b:+.6f} [{lo_b:+.6f}, {hi_b:+.6f}]**, "
              f"semianchura {semi:.6f}, {N_BOOTSTRAP} remuestreos sobre clusters de "
              f"near-duplicados.\n")
    md.append(f"Referencia contra el campeon: V03 {cv_v03_final:.10f}, delta "
              f"{cv_a - cv_v03_final:+.10f}. **Esta comparacion NO aisla la representacion**: el "
              f"protocolo de P73 es mas conservador (inner_train de 949 filas frente a 1186) y la "
              f"arquitectura es distinta.\n")
    md.append("## Convergencia\n")
    md.append("| rama | pliegue | mejor epoca | tope | perdida train | train F1 | val int. F1 | "
              "deriva p1 | grad |\n|---|---|---|---|---|---|---|---|---|\n")
    for n, regs in registros.items():
        for r in regs:
            md.append(f"| {n} | {r['pliegue']} | {r['mejor_epoca']}/{r['epocas_totales']} | "
                      f"{r['toco_tope_epocas']} | {r['traza'][-1]['perdida_train']:.4f} | "
                      f"{r['macro_f1_train_final']:.4f} | "
                      f"{r['traza'][-1]['macro_f1_val_interna']:.4f} | "
                      f"{r['deriva_parametros_epoca_1']} | "
                      f"{r['traza'][-1]['norma_gradiente_preclip']:.4f} |\n")
    md.append("## Rebanadas por longitud\n")
    md.append("| rebanada | filas | V03 | A Word2Vec | B aleatorio | A - B |\n|---|---|---|---|---|---|\n")
    for k, val in rebanadas.items():
        md.append(f"| {k} | {val['filas']} | {val['v03']} | {val['a_word2vec']} | "
                  f"{val['b_aleatorio']} | {val['delta_a_b']:+.4f} |\n")
    md.append("## Rebanadas por estructura del tuit\n")
    md.append("| corte | filas con | A - B con | A - B sin |\n|---|---|---|---|\n")
    for k, val in estructura.items():
        md.append(f"| {k} | {val['filas_con']} | {val['delta_a_b_con']} | {val['delta_a_b_sin']} |\n")
    md.append("## Complementariedad frente a V03\n")
    md.append(f"rescue {comp['global']['rescue']} | damage {comp['global']['damage']} | "
              f"net {comp['global']['net']}\n")
    if veredicto != "A_SUPERA_A_B_CLARO":
        md.append("\nA no supera a B, luego **no se evalua ninguna fusion**, segun el protocolo de "
                  "la fase.\n")
    md.append("## Limites declarados\n")
    for x in salida["limites_declarados"]:
        md.append(f"- {x}\n")
    md.append("## Que no se puede afirmar\n")
    for x in salida["que_no_se_puede_afirmar"]:
        md.append(f"- {x}\n")
    md.append(f"\n## Que falsificaria la conclusion\n\n{salida['que_falsificaria_la_conclusion']}\n")
    md.append("\n## Veredicto\n")
    md.append(f"**P73 = {clasificacion}**\n")
    (REPORTS / "p73_textcnn_random_control.md").write_text("".join(md), encoding="utf-8")

    dec = [f"# Decision de P73\n\n**P73 = {clasificacion}**\n\n",
           (f"delta A - B {cv_a - cv_b:+.10f} | bootstrap de clusters "
           f"[{lo_b:+.6f}, {hi_b:+.6f}] | gana A en {gana_a}/5 pliegues | "
           f"veredicto interno {veredicto}\n\n")]
    if clasificacion == "CLOSED":
        dec.append("El Word2Vec Twitter no demostro ventaja sobre una representacion aleatoria "
                   "equivalente cuando la secuencia fue procesada por una TextCNN pequena y con "
                   "embeddings congelados.\n\nNo se afirma que Word2Vec no funcione en general.\n")
    elif clasificacion == "INVALIDATED":
        dec.append("La representacion aprendida no ofrece ventaja bajo el lector evaluado. No se "
                   "intenta rescatarla cambiando dimension.\n")
    else:
        dec.append("Hay evidencia de que la geometria aprendida aporta informacion que una "
                   "representacion aleatoria no ofrece al lector secuencial. Antes de abrir "
                   "fine-tuning: bootstrap, complementariedad, segunda particion agrupada y "
                   "revisiones de compliance y de DL.\n")
    dec.append(f"\n```text\nV03 = FROZEN\nAGE = sin cambios\nWord2Vec Twitter 100d = congelado\n"
               f"P72 = CLOSED\nP73 = {clasificacion}\ntest = 0\nholdout = 0\nleaderboard = 0\n"
               f"submission = 0\n```\n")
    (REPORTS / "p73_textcnn_decision.md").write_text("".join(dec), encoding="utf-8")

    docs = ["p73_textcnn.json", "p73_textcnn_results.csv", "p73_textcnn_complementarity.csv",
            "p73_textcnn_random_control.md", "p73_textcnn_decision.md"]
    sucios = [n for n in docs if any(b > 127 for b in (REPORTS / n).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")
    log("")
    log("=" * 78)
    log(f"P73-02 OK | {clasificacion} | veredicto interno {veredicto}")
    log(f"   A {cv_a:.10f} | B {cv_b:.10f} | delta {cv_a - cv_b:+.10f} | pliegues A {gana_a}/5")
    log(f"   bootstrap {media_b:+.6f} [{lo_b:+.6f}, {hi_b:+.6f}] | V03 {cv_v03_final:.10f}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()

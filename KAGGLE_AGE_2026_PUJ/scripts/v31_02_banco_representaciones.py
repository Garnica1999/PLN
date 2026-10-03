"""V31+ - 02 - banco de representaciones: se varia la representacion, se fija todo lo demas.

## Por que este script existe

El brief pide separar dos cosas que los experimentos anteriores siempre confundieron:

```text
signal missing      ->  la representacion no expresa la informacion
model capacity missing -> la representacion la expresa pero el modelo no la lee
```

Para distinguirlas hay que **cambiar una sola variable**: la representacion. Aqui el modelo, los
folds, la semilla, el vectorizador y la metrica son los mismos para todas las variantes. La unica
cosa que cambia es que texto ve el vectorizador.

## El protocolo

- Mismos folds que V20: ``StratifiedGroupKFold`` de 5 con ``dup_clusters``.
- Mismo modelo: ``LinearSVC`` con la configuracion fijada abajo. Se elige por determinista y
  rapido; el objetivo es **comparar representaciones**, no maximizar la metrica.
- Mismo vectorizador: TF-IDF de palabra con la misma configuracion para todas las variantes.
- La metrica es macro-F1 de 6 clases, la misma del proyecto.

Ninguna variante puede mirar el test. Ninguna usa el test para decidir. Todas son
deterministas y no dependen de ningun recurso externo: ni embeddings, ni LLM, ni parser, ni red.

## La comparacion que decide el Track A

```text
raw        el texto tal cual, con la URL dentro
url_mask   la URL borrada del texto
url_struct la URL sustituida por un token estructural (URL_TCO, URL_EXT, URL_TRUNC)
```

Si ``url_mask`` mejora sobre ``raw``, la superficie de la URL es ruido que diluye la
representacion. Si ``url_struct`` mejora, el modelo estaba leyendo identidad textual donde deberia
leer estructura. Son dos mecanismos distintos y se distinguen por separado.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v31_02_banco_representaciones.json"
BANCO_CSV = cfg.REPORTS / "v31_representation_benchmark.csv"
FECHA = "2026-09-30"
CV_V30 = 0.6134471
N_CLASES = 6
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DOMINIO_RE = re.compile(r"^(?:https?://)?(?:www\.)?([^/\s?#]+)", re.IGNORECASE)

#: Dominios cortos que en realidad son `t.co` cortado al final del texto. Se detectan aparte
#: porque son un artefacto de truncado y no un dominio real, y como tal pueden ser senal.
DOMINIOS_CONOCIDOS = {
    "t.co", "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "is.gd", "buff.ly", "youtu.be",
    "twitpic.com", "twitcam.com", "yfrog.com", "twitter.com", "x.com", "lnkd.in",
    "cutt.ly", "rebrand.ly", "trib.al", "rb.gy",
}

#: Configuracion del vectorizador. Fija para todas las variantes: es la variable que NO se toca.
VECTORIZADOR = dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.5,
                    sublinear_tf=True, lowercase=True, strip_accents=None,
                    token_pattern=r"(?u)\b\w+\b")
MODELO = dict(C=1.0, loss="squared_hinge", penalty="l2", max_iter=3000, tol=1e-4)


# ============================================================================ transformaciones
def _dominio(url: str) -> str:
    """Dominio de una URL, o cadena vacia si no se puede extraer."""
    m = DOMINIO_RE.match(url.strip())
    return (m.group(1).lower() if m else "").rstrip("/")


def t_raw(texto: str) -> str:
    """Identidad: el texto tal cual."""
    return texto


def t_url_mask(texto: str) -> str:
    """La URL desaparece del texto. Sirve para medir cuanto aporta la superficie de la URL."""
    return URL_RE.sub(" ", texto)


def t_url_struct(texto: str) -> str:
    """La URL se sustituye por un token que describe su **estructura**, no su identidad.

    Es la contraparte directa de `t_url_mask`: si esta gana, el problema no era la presencia de
    la URL sino que el modelo estaba leyendo el identificador opaco de 10 caracteres como si
    fuera informacion.
    """
    def sustituir(m: re.Match) -> str:
        url = m.group(0)
        dom = _dominio(url)
        if dom in DOMINIOS_CONOCIDOS:
            return " URL_TCO "
        if not dom or len(dom) <= 3:
            # `t`, `t.`, `t.c`: el dominio quedo cortado. Es un artefacto del truncado, no un
            # dominio, asi que se marca como tal en vez de inventarse un dominio.
            return " URL_TRUNC "
        return " URL_EXT "
    return URL_RE.sub(sustituir, texto)


def t_url_canal(texto: str) -> str:
    """El texto intacto, mas un canal URL explicito anadido al final.

    A diferencia de `t_url_struct`, aqui la URL sigue presente en el texto y ademas se le da un
    canal propio. Si esto gana sobre `raw`, la informacion de URL es real pero el modelo la
    estaba leyendo enredada con el resto del texto.
    """
    etiquetas = []
    for url in URL_RE.findall(texto):
        dom = _dominio(url)
        if dom in DOMINIOS_CONOCIDOS:
            etiquetas.append("URLPRESENTE URLTCO")
        elif not dom or len(dom) <= 3:
            etiquetas.append("URLPRESENTE URLTRUNC")
        else:
            etiquetas.append("URLPRESENTE URLEXT")
    if not etiquetas:
        etiquetas.append("URLAUSENTE")
    return texto + " " + " ".join(etiquetas)


def t_solo_url(texto: str) -> str:
    """Solo el canal URL, sin el texto. Sirve de control: mide la señal de URL aislada."""
    etiquetas = []
    for url in URL_RE.findall(texto):
        dom = _dominio(url)
        tipo = "URLTCO" if dom in DOMINIOS_CONOCIDOS else (
            "URLTRUNC" if len(dom) <= 3 else "URLEXT")
        etiquetas.append(f"URLPRESENTE {tipo} URLN{len(url)}")
    if not etiquetas:
        etiquetas.append("URLAUSENTE")
    return " ".join(etiquetas)


TRANSFORMACIONES: dict[str, Callable[[str], str]] = {
    "raw": t_raw,
    "url_mask": t_url_mask,
    "url_struct": t_url_struct,
    "url_canal": t_url_canal,
    "solo_url": t_solo_url,
}


# ============================================================================ evaluacion
def evaluar(nombre: str, transformacion: Callable[[str], str], textos: np.ndarray,
            clase: np.ndarray, folds: list[dict]) -> dict:
    """Evalua una transformacion con modelo y folds fijos. Devuelve metricas y el OOF."""
    inicio = time.time()
    corpus = np.array([transformacion(t) for t in textos], dtype=object)
    vacios = int(sum(1 for c in corpus if not c.strip()))
    # El vectorizador se ajusta DENTRO de cada fold, sobre el corpus de train del fold. Es lo
    # honesto: ajustar el IDF con todo el corpus antes de partir filtraria informacion del
    # fold de validacion a traves de las frecuencias de termino, aunque el modelo no sea
    # supervisado.
    oof = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    por_fold = []
    for f in folds:
        ent, val = f["train_idx"], f["val_idx"]
        vec = TfidfVectorizer(**VECTORIZADOR)
        x_ent = vec.fit_transform(corpus[ent])
        x_val = vec.transform(corpus[val])
        modelo = LinearSVC(**MODELO)
        modelo.fit(x_ent, clase[ent])
        dec = modelo.decision_function(x_val)
        if dec.ndim == 1:
            dec = np.column_stack([-dec, dec])
        oof[val] = dec
        por_fold.append(float(f1_score(clase[val], np.argmax(dec, axis=1), average="macro",
                                       labels=list(range(N_CLASES)))))
    cv = float(f1_score(clase, np.argmax(oof, axis=1), average="macro",
                        labels=list(range(N_CLASES))))
    por_clase = f1_score(clase, np.argmax(oof, axis=1), average=None,
                         labels=list(range(N_CLASES)))
    return {
        "representacion": nombre, "macro_f1": round(cv, 7), "folds": [round(v, 7) for v in por_fold],
        "delta_por_fold": None, "f1_por_clase": [round(float(v), 5) for v in por_clase],
        "documentos_vacios": vacios, "vocabulario_medio": None,
        "segundos": round(time.time() - inicio, 1), "_oof": oof,
    }


def main() -> int:
    """Ejecuta el banco de representaciones y guarda los OOF para el analisis posterior."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    textos = train["text"].fillna("").astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    print(f"folds: {[len(f['val_idx']) for f in folds]} (los mismos que V20)", flush=True)

    filas = []
    oofs = {}
    for nombre, fn in TRANSFORMACIONES.items():
        res = evaluar(nombre, fn, textos, clase, folds)
        oofs[nombre] = res.pop("_oof")
        filas.append(res)
        print(f"  {nombre:<12} macro_f1 {res['macro_f1']:.6f} | folds "
              f"{[round(v, 4) for v in res['folds']]} | {res['segundos']} s", flush=True)

    base = next(f for f in filas if f["representacion"] == "raw")
    for f in filas:
        f["delta_vs_raw"] = round(f["macro_f1"] - base["macro_f1"], 7)
        f["delta_por_fold"] = [round(a - b, 7) for a, b in zip(f["folds"], base["folds"])]
        f["folds_positivos"] = int(sum(1 for d in f["delta_por_fold"] if d > 0))
    tabla = pd.DataFrame([{k: v for k, v in f.items() if k != "folds"} for f in filas])
    tabla = tabla.sort_values("macro_f1", ascending=False)
    tabla.to_csv(BANCO_CSV, index=False)

    print("\n=== orden ===", flush=True)
    for _, f in tabla.iterrows():
        marca = "BASE" if f["representacion"] == "raw" else f"{f['delta_vs_raw']:+.6f}"
        print(f"  {f['representacion']:<12} {f['macro_f1']:.6f}  {marca:>10}  "
              f"folds+ {f['folds_positivos']}/5", flush=True)

    np.savez_compressed(cfg.FEATURES / "v31_representation_oof.npz", **{k: v for k, v in oofs.items()})

    RESULTADO.write_text(json.dumps({
        "fase": "V31+ 02 banco de representaciones", "fecha": FECHA, "experimenta": True,
        "baseline_v30": CV_V30,
        "objetivo": "aislar la representacion: modelo, folds, semilla, vectorizador y metrica "
                    "son identicos en todas las variantes, lo unico que cambia es el texto",
        "protocolo": {"folds": "StratifiedGroupKFold 5, dup_clusters, identicos a V20",
                      "modelo": "LinearSVC(C=1.0, squared_hinge)", "vectorizador": VECTORIZADOR,
                      "metrica": "macro-F1 de 6 clases"},
        "variantes": filas,
        "restricciones": {"usa_test": False, "usa_red": False, "usa_embeddings": False,
                          "usa_parser_externo": False, "usa_labels_de_validacion": False},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

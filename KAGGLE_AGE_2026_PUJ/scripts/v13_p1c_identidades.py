"""V13 · P1c — ¿La evidencia por token es señal lingüística o una tabla de identidades?

P1a dio 0,3997 de macro-F1 en el control *block-only*. Al mirar los tokens mas asociados a cada
clase la respuesta es inmediata y no conviene disimularla: son **nombres de cuenta**
(``matthunter`` en 13-17, ``dannapaola`` en 18-24, ``wilvercuellar`` en 35-49, ``tiagemita`` en
65-xx). El bloque esta learniendo, en la practica, "de que comunidad escribe esta persona", que es
una forma legitima de senal pero muy distinta de la hipotesis inicial de "evidencia lexical de la
edad", y con una fragilidad obvia: depende de que la cuenta mencionada aparezca tambien en el train.

Este experimento separa las dos cosas sin reentrenar nada del meta-modelo. Reconstruye el mismo
bloque de P1a tres veces, cambiando unicamente el vocabulario permitido:

1. **Completo**: tal cual P1a (linea base comparable).
2. **Sin cuentas**: se excluyen los tokens que aparecen en el corpus como mencion (``@nombre``).
   Si la senal desaparece, el bloque era una tabla de consulta de identidades.
3. **Sin cuentas ni hashtags**: se excluyen ademas los tokens que aparecen como hashtag. Asi el
   bloque no se solapa con la pista P2 (``P(clase | hashtag)``), que es objeto aparte.

Las menciones y los hashtags se extraen con las expresiones regulares del propio proyecto y el
vocabulario excluido se calcula **dentro del fold** (sobre el train interior), igual que el
vocabulario del bloque, para no abrir una via de leakage por la puerta de atras.

Criterio de lectura: si la variante *sin cuentas* cae cerca del suelo de azar (0,1667), la senal de
P1a no es evidencia lingüistica de la edad sino reconocimiento de cuentas, y P1b debe leerse con
esa etiqueta. Si se mantiene, hay dos componentes y ambos son útiles.

Entregables: ``reports/v13_p1_identidades.md``, ``reports/v13_p1_identidades.csv``,
``experiments/v13_p1c_identidades.json``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v13_p1a_evidencia_tokens import (BANDAS, CARACTERISTICAS_POR_BANDA, SUAVIZADO,  # noqa: E402
                                      frecuencia_de_documento, perfil_de_clase,
                                      resumir_documentos, tokeniza_todo)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v13_p1_identidades.md"
TABLA = cfg.REPORTS / "v13_p1_identidades.csv"
RESULTADO = cfg.EXPERIMENTS / "v13_p1c_identidades.json"
NOMBRES = cfg.FEATURES / "v13_p1_tokens_nombres.json"

#: Extraccion de menciones y hashtags del propio corpus (mismo criterio que el bloque de entidades).
RE_MENCION = re.compile(r"@(\w+)")
RE_HASHTAG = re.compile(r"#(\w+)")
SUELO_AZAR = 1.0 / len(cfg.CLASSES)


def nombres_de_cuenta(textos: pd.Series) -> tuple[set[str], set[str]]:
    """Conjunto de nombres de cuenta mencionados y de hashtags, en minúsculas.

    Se construye sobre **un subconjunto de documentos** para poder hacerlo por fold.
    """
    menciones: set[str] = set()
    hashtags: set[str] = set()
    for texto in textos:
        texto = str(texto)
        menciones.update(RE_MENCION.findall(texto.lower()))
        hashtags.update(RE_HASHTAG.findall(texto.lower()))
    return menciones, hashtags


def bloque_por_fold(indice, punteros, textos, folds, y, clusters, clase, n_tokens,
                    identificador: dict[str, int], tipos_excluidos: set[str]
                    ) -> dict[int, np.ndarray]:
    """Reconstruye el bloque de P1a con el vocabulario indicado, sin leakage.

    Los identificadores excluidos (menciones y hashtags) se.detectan **dentro de cada fold
    exterior, sobre su train**, igual que el vocabulario del bloque.
    """
    matrices: dict[int, np.ndarray] = {}
    nombres = json.loads(NOMBRES.read_text(encoding="utf-8"))
    for fold in folds:
        numero = fold["fold"]
        validacion = fold["val_idx"]
        train_exterior = np.concatenate([otro["val_idx"] for otro in folds
                                         if otro["fold"] != numero])
        interiores = cvmod.make_folds(y[train_exterior], groups=clusters[train_exterior],
                                      n_splits=5, n_repeats=1, scheme="group")
        menciones, hashtags = nombres_de_cuenta(textos.iloc[train_exterior])
        excluidos: set[str] = set()
        if "mencion" in tipos_excluidos:
            excluidos |= menciones
        if "hashtag" in tipos_excluidos:
            excluidos |= hashtags
        identificadores = {identificador[token] for token in excluidos if token in identificador}

        matriz = np.zeros((len(y), len(nombres)), dtype=np.float32)
        for numero_banda, (banda, (bajo, alto)) in enumerate(BANDAS.items()):
            inicio = numero_banda * CARACTERISTICAS_POR_BANDA
            bloque = slice(inicio, inicio + CARACTERISTICAS_POR_BANDA)
            # Primero el ajuste cruzado dentro del train exterior, despues la validacion exterior.
            pasos = [(np.setdiff1d(train_exterior, train_exterior[interior["val_idx"]]),
                      train_exterior[interior["val_idx"]]) for interior in interiores]
            pasos.append((train_exterior, validacion))
            for documentos, destino in pasos:
                frecuencia = frecuencia_de_documento(indice, punteros, documentos, n_tokens)
                seleccion = np.flatnonzero((frecuencia >= bajo) & (frecuencia < alto))
                seleccion = np.array([token for token in seleccion
                                      if int(token) not in identificadores], dtype=np.int64)
                if not len(seleccion):
                    continue
                perfil = perfil_de_clase(indice, punteros, documentos, clase, seleccion,
                                         len(cfg.CLASSES))
                matriz[destino, bloque] = resumir_documentos(indice, punteros, destino, perfil,
                                                            seleccion, len(cfg.CLASSES))
        matrices[numero] = matriz
        print(f"[fold {numero}] identificadores excluidos: {len(identificadores):,}", flush=True)
    return matrices


def main() -> int:
    """Compara el control block-only con y sin identificadores de cuenta."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    textos = train["text"]
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    indice, punteros, vocabulario = tokeniza_todo(textos)
    identificador = {token: posicion for posicion, token in enumerate(vocabulario)}
    print(f"[vocabulario] {len(vocabulario):,} tokens del corpus", flush=True)

    variantes = {
        "completo": set(),
        "sin_cuentas": {"mencion"},
        "sin_cuentas_ni_hashtags": {"mencion", "hashtag"},
    }
    filas: list[dict] = []
    detalle: dict[str, dict] = {}
    # Se cuentan los identificadores una vez, sobre el train exterior del primer fold, para
    # dimensionar el efecto sin recalcularlos en cada variante.
    train_primer_fold = np.concatenate([otro["val_idx"] for otro in folds
                                        if otro["fold"] != folds[0]["fold"]])
    menciones_primer, hashtags_primer = nombres_de_cuenta(textos.iloc[train_primer_fold])
    excluidos_por_fold = {
        "menciones": sum(1 for token in menciones_primer if token in identificador),
        "hashtags": sum(1 for token in hashtags_primer if token in identificador)}
    for nombre, tipos in variantes.items():
        matrices = bloque_por_fold(indice, punteros, textos, folds, y, clusters, clase,
                                   len(vocabulario), identificador, tipos)
        for etiqueta, constructor in (
                ("logreg", lambda: LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)),
                ("linearsvc", lambda: LinearSVC(C=0.2, random_state=cfg.SEED))):
            por_fold = []
            for fold in folds:
                validacion = fold["val_idx"]
                train_exterior = np.concatenate([otro["val_idx"] for otro in folds
                                                 if otro["fold"] != fold["fold"]])
                estimador = constructor()
                estimador.fit(matrices[fold["fold"]][train_exterior], clase[train_exterior])
                por_fold.append(float(f1_score(clase[validacion],
                                               estimador.predict(
                                                   matrices[fold["fold"]][validacion]),
                                               average="macro",
                                               labels=list(range(len(cfg.CLASSES))))))
            filas.append({"variante": nombre, "modelo": etiqueta,
                          "macro_f1": round(float(np.mean(por_fold)), 4),
                          "std": round(float(np.std(por_fold)), 4),
                          "suelo_azar": round(SUELO_AZAR, 4),
                          "per_fold": [round(valor, 4) for valor in por_fold]})
            detalle.setdefault(nombre, {})[etiqueta] = round(float(np.mean(por_fold)), 4)
            print(f"[{nombre} · {etiqueta}] {np.mean(por_fold):.4f} ± {np.std(por_fold):.4f}",
                  flush=True)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLA, index=False)

    completa = detalle["completo"]["logreg"]
    sin_cuentas = detalle["sin_cuentas"]["logreg"]
    lineas = [
        "# V13 · P1c — ¿Señal lingüística o tabla de identidades?", "",
        "Los tokens mas asociados a cada clase en P1a son **nombres de cuenta** "
        "(`matthunter` en 13-17, `dannapaola` en 18-24, `wilvercuellar` en 35-49, `tiagemita` en "
        "65-xx). Este experimento separa ese componente del resto sin reentrenar el meta-modelo.", "",
        "## 1. Método", "",
        "Se reconstruye el bloque de P1a tres veces cambiando solo el vocabulario permitido: el "
        "completo, el que excluye los tokens que aparecen en el corpus como mención (`@nombre`) y "
        "el que excluye además los hashtags. Los identificadores se detectan con las expresiones "
        "regulares del propio proyecto y sobre el train de los folds exteriores, no sobre el "
        "corpus entero.", "",
        "## 2. Control block-only", "",
        "| Variante | LogReg | LinearSVC |",
        "|---|---|---|",
    ]
    for nombre in variantes:
        sub = tabla[tabla["variante"] == nombre]
        lineas.append(f"| {nombre} | {detalle[nombre]['logreg']:.4f} | "
                      f"{detalle[nombre]['linearsvc']:.4f} |")
    lineas += ["", f"Suelo de azar: {SUELO_AZAR:.4f}.", "", "## 3. Lectura", ""]
    caida = completa - sin_cuentas
    proporcion = 100 * caida / max(1e-9, completa - SUELO_AZAR)
    lineas += [
        f"- El bloque completo da {completa:.4f} y sin cuentas {sin_cuentas:.4f}: la senal cae "
        f"**{caida:.4f}**, es decir un **{proporcion:.1f} %** de la mejora sobre el azar.",
        f"- Identificadores detectados en el train exterior: "
        f"{excluidos_por_fold['menciones']:,} nombres de cuenta mencionados y "
        f"{excluidos_por_fold['hashtags']:,} hashtags.",
    ]
    if sin_cuentas < SUELO_AZAR + 0.05:
        lineas.append("- **Veredicto: el bloque es esencialmente una tabla de consulta de "
                      "identidades.** Prácticamente toda su potencia viene de reconocer cuentas, no "
                      "de la evidencia léxica de la edad. P1b debe leerse con esa etiqueta: una "
                      "mejora ahí depende de que las cuentas mencionadas también estén en el train "
                      "y no es una propiedad transferible del texto.")
    elif proporcion > 60:
        lineas.append("- **Veredicto: predomina el componente de identidad** (más del 60 % de la "
                      "mejora), pero queda un componente léxico que no es una tabla de consulta. "
                      "Conviene separar las dos lecturas al informar de P1b.")
    else:
        lineas.append("- **Veredicto: el bloque es sobre todo evidencia léxica**: excluir las "
                      "cuentas deja intacta la mayor parte de la señal, que es lo que la hipótesis "
                      "original esperaba.")
    lineas += ["", "## 4. Por qué importa aunque la señal sea real", "",
               "Reconocer una cuenta **es** información legítima sobre la edad del autor (dice en "
               "qué comunidad escribe), y el protocolo lo respeta: todo se calcula dentro del "
               "fold y el test no se toca. Pero es una señal frágil y de naturaleza distinta: no "
               "describe cómo escribe la persona sino a quién menciona, y no se transfiere a "
               "cuentas no vistas. Por eso se mide por separado en vez de esconderlo dentro de una "
               "única cifra de macro-F1.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V13 · P1c identidad frente a evidencia lexica", "fecha": "2026-09-25",
        "variantes": {nombre: detalle[nombre] for nombre in variantes},
        "identificadores_detectados": excluidos_por_fold,
        "suelo_azar": round(SUELO_AZAR, 4),
        "caida_al_excluir_cuentas": round(caida, 4),
        "proporcion_de_la_mejora_que_es_identidad": round(proporcion, 1),
        "suavizado": SUAVIZADO, "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ncaida al excluir cuentas: {caida:.4f} ({proporcion:.1f} % de la mejora sobre el azar)")
    print(f"artefactos: {INFORME.name}, {TABLA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

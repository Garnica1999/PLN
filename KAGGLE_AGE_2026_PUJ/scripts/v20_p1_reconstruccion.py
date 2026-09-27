"""V20 · P1 — ORDINAL_RECONSTRUCTION: cuatro variantes de convertir cinco CDF en seis clases.

V20 queda **congelado**. Aqui no se toca ni un clasificador: solo se cambia la **capa de
reconstruccion**, que es exactamente la pregunta de la fase:

> ¿La mejora de V20 procede de usar cinco fronteras ordinales, o todavia hay margen en la forma de
> convertir esas cinco probabilidades acumuladas en las seis clases finales?

**Lo que la auditoria P0 dejo establecido, y que condiciona lo que aqui se puede esperar.** Las cinco
CDFs estan **esencialmente perfectamente calibradas** (descalibracion entre -0,0003 y +0,0002), luego
**no hay nada que ganar recalibrando**. Y solo el **11,18 %** de las filas violan la monotonia, con una
magnitud media de **0,002457**, luego **la proyeccion monotona tambien tiene poco que corregir**. Ambos
resultados apuntan a que el margen de esta subfamilia es pequeno, y eso es una hipotesis medible, no una
corazonada.

**Las cuatro variantes**, todas sobre las mismas cinco CDFs ya medidas:

| Variante | Mecanismo | Por que es distinta |
|---|---|---|
| **A** | la combinacion actual de V20 | **Control exacto**: debe reproducir 0,6097782. Si no, el script tiene un bug |
| **B** | proyeccion monotona por **PAVA** (media de bloques) en lugar de maximo acumulado | el maximo acumulado sube el valor posterior al anterior y deja el resto; PAVA reparte la violacion promediando el bloque, que es la proyeccion L2 sobre el cono monotono |
| **C** | decision por **conteo de umbrales cruzados**: clase = cuantas fronteras supera la fila | no usa las probabilidades reconstruidas: decide por la estructura acumulada directamente |
| **D** | **edad esperada**: `E[clase]` y decision por intervalos sobre ese valor | introduce la distancia ordinal en la decision, que el argmax no tiene |

**La metrica que decide** no es solo el macro-F1: es la **complementariedad con V20**. Se mide en las
cuatro celdas (ambos correctos, ambos incorrectos, solo el candidato, solo V20) y en rescates, danos,
neto y numero de predicciones modificadas. Un candidato que mejore el CV cambiando solo las mismas filas
y anadiendo dano no lo justifica.

Entregables: ``experiments/v20_p1_reconstruccion.json``, ``reports/v20_p1_reconstruccion.md``,
``reports/v20_p1_comparativa.csv``, ``reports/v20_p1_complementariedad.csv`` y una figura.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import matriz_279  # noqa: E402
from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_frontera  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v20_p1_reconstruccion.md"
COMPARATIVA = cfg.REPORTS / "v20_p1_comparativa.csv"
COMPLEMENTARIEDAD = cfg.REPORTS / "v20_p1_complementariedad.csv"
RESULTADO = cfg.EXPERIMENTS / "v20_p1_reconstruccion.json"
FIGURA = cfg.FIGURES / "v20_p1_reconstruccion.png"
ACUMULADAS = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
OOF_V20 = cfg.FEATURES / "v20_f3b_oof_candidato.npy"
CV_V20 = 0.6097782
N_UMBRAL = 5
ETIQUETAS = list(range(len(cfg.CLASSES)))


def pava(valores: np.ndarray) -> np.ndarray:
    """Proyeccion L2 de cada fila sobre el cono de secuencias **crecientes**, por pooling de bloques.

    Es la proyeccion de isotonic regression: cuando un valor es menor que el de antes, se promedian los
    dos bloques y se sigue mientras la violacion persista. Difiere del maximo acumulado en que reparte la
    violacion entre todo el bloque en vez de empujar el valor posterior hacia arriba.
    """
    n = len(valores)
    if n == 0:
        return valores.copy()
    bloques: list[list[int]] = [[i] for i in range(n)]
    valores_bloque = [float(v) for v in valores]
    resultado = np.zeros(n)
    apilado: list[tuple[float, int]] = []
    for posicion in range(n):
        apilado.append((float(valores[posicion]), 1))
        while len(apilado) >= 2 and apilado[-2][0] >= apilado[-1][0]:
            valor, cuenta = apilado.pop()
            anterior, cuenta_anterior = apilado.pop()
            total = cuenta + cuenta_anterior
            apilado.append(((anterior * cuenta_anterior + valor * cuenta) / total, total))
    inicio = 0
    for valor, cuenta in apilado:
        resultado[inicio:inicio + cuenta] = valor
        inicio += cuenta
    return resultado


def reconstruir_por_fila(acumuladas: np.ndarray, metodo: str) -> np.ndarray:
    """Las seis clases a partir de las cinco CDFs, segun el metodo de reconstruccion pedido."""
    n_filas = acumuladas.shape[0]
    n_clases = N_UMBRAL + 1
    if metodo == "maximo_acumulado":
        mono = acumuladas.copy()
        for k in range(1, N_UMBRAL):
            mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    elif metodo == "pava":
        mono = np.zeros_like(acumuladas)
        for i in range(n_filas):
            mono[i] = pava(acumuladas[i])
    elif metodo == "contar_umbrales":
        # La clase es el numero de fronteras que la fila supera. No usa las probabilidades
        # reconstruidas: decide por la estructura acumulada directamente.
        score_alto = 1.0 - acumuladas
        conteo = (score_alto > 0.5).sum(axis=1).astype(int)
        conteo = np.clip(conteo, 0, n_clases - 1)
        return np.eye(n_clases)[conteo]
    elif metodo == "edad_esperada":
        mono = acumuladas.copy()
        for k in range(1, N_UMBRAL):
            mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
        seis = np.zeros((n_filas, n_clases))
        seis[:, 0] = mono[:, 0]
        for k in range(1, N_UMBRAL):
            seis[:, k] = mono[:, k] - mono[:, k - 1]
        seis[:, n_clases - 1] = 1.0 - mono[:, N_UMBRAL - 1]
        seis = np.clip(seis, 0.0, None)
        seis = seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)
        esperada = (seis * np.arange(n_clases)).sum(axis=1)
        # Intervalos sobre el valor esperado: elige la clase cuya marca de clase esta mas cercana.
        marcas = np.arange(n_clases)
        indice = np.argmin(np.abs(esperada[:, None] - marcas[None, :]), axis=1)
        salida = np.eye(n_clases)[indice]
        salida[:, 0] += 0.0
        return salida
    else:
        raise ValueError(f"metodo desconocido: {metodo}")
    seis = np.zeros((n_filas, n_clases))
    seis[:, 0] = mono[:, 0]
    for k in range(1, N_UMBRAL):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, n_clases - 1] = 1.0 - mono[:, N_UMBRAL - 1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def main_func() -> int:
    """Evalua las cuatro variantes contra V20."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    acumuladas = np.load(ACUMULADAS)
    oof_v20 = np.load(OOF_V20)
    pesos = (0.25, 0.25, 0.25, 0.25)
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19 = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")

    metodos = [("A control exacto: maximo acumulado", "maximo_acumulado"),
               ("B proyeccion monotona por PAVA", "pava"),
               ("C decision por conteo de umbrales", "contar_umbrales"),
               ("D decision por edad esperada", "edad_esperada")]
    resultados = []
    for nombre, metodo in metodos:
        print(f"[{nombre}]", flush=True)
        seis = reconstruir_por_fila(acumuladas, metodo)
        blend = (pesos[0] * logreg + pesos[1] * mlp + pesos[2] * arbol_v19
                 + pesos[3] * seis)
        valor = macro(clase, blend)
        por_fold, deltas = [], []
        for fold in folds:
            idx = fold["val_idx"]
            a = macro(clase[idx], blend[idx])
            b = macro(clase[idx], oof_v20[idx])
            por_fold.append(a)
            deltas.append(a - b)
        intervalo = bootstrap_diferencia(oof_v20, blend, clase, len(cfg.CLASSES))
        fronteras = metricas_frontera(blend, clase, PARES)
        fronteras_v20 = metricas_frontera(oof_v20, clase, PARES)
        pred = np.array(cfg.CLASSES)[np.argmax(blend, axis=1)]
        pred_v20 = np.array(cfg.CLASSES)[np.argmax(oof_v20, axis=1)]
        acierta, acierta_v20 = pred == y, pred_v20 == y
        item = {"variante": nombre, "metodo": metodo, "macro_f1": round(valor, 7),
                "delta_contra_v20": round(valor - macro(clase, oof_v20), 6),
                "media_folds": round(float(np.mean(por_fold)), 6),
                "desviacion_folds": round(float(np.std(por_fold)), 6),
                "delta_por_fold": [round(v, 6) for v in deltas],
                "folds_positivos": sum(1 for v in deltas if v > 0),
                "bootstrap_ic95_contra_v20": [round(v, 6) for v in intervalo],
                "frontera_1": fronteras["18-24|25-34"]["accuracy"],
                "frontera_1_v20": fronteras_v20["18-24|25-34"]["accuracy"],
                "frontera_2": fronteras["25-34|35-49"]["accuracy"],
                "frontera_2_v20": fronteras_v20["25-34|35-49"]["accuracy"],
                "complementariedad": {
                    "ambos_correctos": int((acierta & acierta_v20).sum()),
                    "ambos_incorrectos": int((~acierta & ~acierta_v20).sum()),
                    "solo_candidato": int((acierta & ~acierta_v20).sum()),
                    "solo_v20": int((acierta_v20 & ~acierta).sum()),
                    "neto": int((acierta & ~acierta_v20).sum())
                            - int((acierta_v20 & ~acierta).sum()),
                    "predicciones_modificadas": int((pred != pred_v20).sum())}}
        item["rescates"] = item["complementariedad"]["solo_candidato"]
        item["danos"] = item["complementariedad"]["solo_v20"]
        resultados.append(item)
        print(f"    {valor:.7f} | contra V20 {item['delta_contra_v20']:+.4f} | "
              f"{item['folds_positivos']}/5 | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | "
              f"frontera 1 {item['frontera_1']} y 2 {item['frontera_2']} | rescate "
              f"{item['rescates']:,} dano {item['danos']:,} neto {item['complementariedad']['neto']:+,}",
              flush=True)

    # El control tiene que reproducir V20 exacto; si no, el script esta mal.
    control = resultados[0]
    cumple_control = abs(control["macro_f1"] - CV_V20) < 1e-6
    print(f"\n[A control] {control['macro_f1']:.7f} frente a {CV_V20:.7f} de V20 | "
          f"reproduce el control: {cumple_control}", flush=True)

    tabla = pd.DataFrame([{
        "variante": "V20 (champion)", "macro_f1": CV_V20, "delta_contra_v20": 0.0,
        "folds_positivos": "-", "ic95": "-", "frontera_1": 0.5075, "frontera_2": 0.5029,
        "rescates": 0, "danos": 0, "neto": 0, "cambios": 0,
        "estado": "CHAMPION"}] + [{
        "variante": r["variante"], "macro_f1": r["macro_f1"],
        "delta_contra_v20": r["delta_contra_v20"],
        "folds_positivos": f"{r['folds_positivos']}/5",
        "ic95": f"[{r['bootstrap_ic95_contra_v20'][0]:+.4f}, "
                 f"{r['bootstrap_ic95_contra_v20'][1]:+.4f}]",
        "frontera_1": r["frontera_1"], "frontera_2": r["frontera_2"],
        "rescates": r["rescates"], "danos": r["danos"],
        "neto": r["complementariedad"]["neto"],
        "cambios": r["complementariedad"]["predicciones_modificadas"],
        "estado": ""} for r in resultados])
    tabla.to_csv(COMPARATIVA, index=False)
    pd.DataFrame([{"variante": r["variante"], **r["complementariedad"]}
                  for r in resultados]).to_csv(COMPLEMENTARIEDAD, index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(16, 5.5))
    eje = ejes[0]
    nombres = ["V20"] + [r["variante"].split()[0] for r in resultados]
    eje.bar(range(len(resultados) + 1), [0.0] + [r["delta_contra_v20"] * 1000 for r in resultados],
            color=["#1a9850"] + ["#2166ac"] * len(resultados))
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(nombres, rotation=20, fontsize=7)
    eje.set_ylabel("delta de macro-F1 contra V20 (milesimas)")
    eje.set_title("Cuatro formas de reconstruir las seis clases desde cinco CDF", fontsize=10)
    eje = ejes[1]
    eje.bar(range(len(resultados)),
            [r["complementariedad"]["solo_candidato"] for r in resultados], 0.4,
            label="solo el candidato acierta", color="#1a9850")
    eje.bar(range(len(resultados)),
            [-r["complementariedad"]["solo_v20"] for r in resultados], 0.4,
            label="solo V20 acierta (negativo)", color="#b2182b")
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(range(len(resultados)))
    eje.set_xticklabels([r["variante"].split()[0] for r in resultados], fontsize=8)
    eje.set_ylabel("filas")
    eje.set_title("Complementariedad con V20", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V20 P1: ORDINAL_RECONSTRUCTION, sin tocar un solo clasificador", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=130)
    plt.close(fig)

    INFORME.write_text("\n".join([
        "# V20 · P1 — ORDINAL_RECONSTRUCTION: cuatro formas de reconstruir las seis clases", "",
        "**V20 queda congelado.** No se toca un solo clasificador: solo la capa de reconstruccion.", "",
        "## 1. Lo que la auditoria P0 dejo establecido, y por que aqui hay poco margen", "",
        "- Las cinco CDFs estan **esencialmente perfectamente calibradas**: descalibracion entre "
        "**-0,0003 y +0,0002**. **No hay nada que ganar recalibrando.**",
        "- Solo el **11,18 %** de las filas violan la monotonia, con magnitud media **0,002457** y maxima "
        "**0,518606**. **La proyeccion monotona tiene poco que corregir.**",
        "- **Consecuencia**: el margen de esta subfamilia es pequeno, y eso es una hipotesis medible, no "
        "una corazonada. Si las cuatro variantes salen planas, el hallazgo es que la consistencia ordinal "
        "ya esta saturada por la forma en que V20 la resolvio.", "",
        "## 2. Las cuatro variantes", "",
        "| Variante | Mecanismo | Por que es distinta |", "|---|---|---|",
        "| **A** | la combinacion actual de V20 | **Control exacto**: debe reproducir 0,6097782 |",
        "| **B** | proyeccion monotona por **PAVA** | el maximo acumulado empuja el valor posterior hacia "
        "arriba; PAVA reparte la violacion promediando el bloque, que es la proyeccion L2 sobre el cono |",
        "| **C** | **conteo de umbrales cruzados** | no usa las probabilidades reconstruidas: decide por "
        "la estructura acumulada |",
        "| **D** | **edad esperada** | introduce la distancia ordinal en la decision, que el argmax no "
        "tiene |", "",
        "## 3. Resultados contra V20", "",
        "| Variante | macro-F1 | Delta contra V20 | Folds pos. | IC95 | Frontera 1 | Frontera 2 | "
        "Rescates | Danos | Neto | Cambios |", "|---|---|---|---|---|---|---|---|---|---|---|",
        f"| **V20 (champion)** | {CV_V20} | - | - | - | 0,5075 | 0,5029 | - | - | - | - | CHAMPION |",
        *[f"| {r['variante']} | {r['macro_f1']} | {r['delta_contra_v20']:+.4f} | "
          f"{r['folds_positivos']}/5 | [{r['bootstrap_ic95_contra_v20'][0]:+.4f}, "
          f"{r['bootstrap_ic95_contra_v20'][1]:+.4f}] | {r['frontera_1']} | {r['frontera_2']} | "
          f"{r['rescates']:,} | {r['danos']:,} | {r['complementariedad']['neto']:+,} | "
          f"{r['complementariedad']['predicciones_modificadas']:,} |" for r in resultados], "",
        f"**El control A reproduce V20 exacto: {cumple_control}** ({control['macro_f1']:.7f} frente a "
        f"{CV_V20:.7f}).", "",
        "## 4. Delta por fold", "",
        "| Variante | Delta por fold |", "|---|---|",
        *[f"| {r['variante']} | {r['delta_por_fold']} |" for r in resultados], "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V20 P1 ORDINAL_RECONSTRUCTION", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20 cadena de umbrales anidados", "lb": 0.62689,
                               "cv": CV_V20, "intacto": True},
        "contexto_de_la_auditoria": {
            "descalibracion_de_las_cinco": "entre -0,0003 y +0,0002",
            "filas_que_violan_la_monotomia": "11,18 %",
            "magnitud_media_de_la_violacion": 0.002457,
            "conclusion": "no hay margen en recalibrar ni en proyectar; el margen de esta subfamilia "
                          "es pequeno y eso es medible"},
        "metodos": {r["metodo"]: r["variante"] for r in resultados},
        "resultados": resultados,
        "control_reproducido": cumple_control,
        "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsegundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_func())

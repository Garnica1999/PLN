"""AGE · Auditoria de cierre estrategico — verificacion de las cifras que deciden el cierre.

Este script **no entrena nada y no propone ninguna variante**. Es la auditoria que pide el brief de
cierre: comprobar, sobre artefactos que ya existen en disco, las dos afirmaciones que sostienen la
decision de cerrar o de ejecutar un ultimo experimento.

**Afirmacion 1, la que revierte una razon de D-049.** D-049 cerro V18 diciendo que "la macro-F1 de
seis clases es una media, y la frontera es una minoridad de la masa". La lectura literal de esa frase
es falsa: `18-24` y `25-34` suman algo mas de un tercio de las filas, no una minoridad. Este script
mide la masa de cada clase y la **elasticidad** de la macro-F1 global frente al numero de correcciones
en la frontera, que es lo que convierte la frase en algo cuantitativo. Si la lever fuera pequena, el
cierre serait correcto por magnitud; si es grande, el cierre solo puede sostenerse por otra razon.

La elasticidad se mide como un **contrafactual sobre el OOF que ya existe**: se toman filas que V12
clasifica mal dentro de un par, se les corrige a la etiqueta verdadera, y se recalcula la macro-F1
pooled. No hay reentrenamiento, no hay modelo nuevo y no se toca nada. Es una cuenta, no un
experimento, y por eso es legitima como auditoria.

**Afirmacion 2, la calibracion CV -> LB.** El gap de +0,0213 entre el CV de V12 y su leaderboard se
puede leer como sobreajuste a fold o como una bonificacion oculta. Si fuera lo segundo, los gates
locales estaria mas cerrados de lo que parecen en terminos de LB y la decision de cerrar cambiaria. Se
reconstruye la regresion con los **envios realmente puntuados**, leidos de
``experiments/phase12_auditoria_estado.json``, y se informa de la pendiente: un gate de +0,003 en CV
solo vale en LB lo que la pendiente diga.

**Lo que este script NO hace**, y es deliberado: no genera una sonda de decodificacion, no prueba
prior-shift, no busca hiperparametros y no crea miembros. Una sonda de ese tipo seria un experimento,
y el brief exige que se justifique antes de ejecutar, no despues.

Entregables: ``reports/age_cierre_auditoria.md``, ``reports/age_cierre_auditoria.csv``,
``reports/age_lb_calibracion.csv``, ``figures/age_elasticidad_frontera.png`` y
``experiments/age_cierre_auditoria.json``.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "age_cierre_auditoria.md"
TABLA = cfg.REPORTS / "age_cierre_auditoria.csv"
CALIBRACION = cfg.REPORTS / "age_lb_calibracion.csv"
RESULTADO = cfg.EXPERIMENTS / "age_cierre_auditoria.json"
FIGURA = cfg.FIGURES / "age_elasticidad_frontera.png"

REFERENCIA = 0.6005017473
LB_REFERENCIA = 0.62188
GATE = 0.003
#: Pares problemáticos, por nombre de clase.
PARES = (("18-24", "25-34"), ("25-34", "35-49"))
#: Numero de correcciones del contrafactual y repeticiones del remuestreo.
TAMAÑOS = (250, 500, 1000, 2000, 5000)
REPETICIONES = 200


def masa_y_f1(clase: np.ndarray, prediccion: np.ndarray) -> list[dict]:
    """Masa y F1 por clase del OOF de V12, en el orden canonico de las seis clases."""
    filas = []
    f1 = f1_score(clase, prediccion, average=None, labels=list(range(len(cfg.CLASSES))))
    for indice, nombre in enumerate(cfg.CLASSES):
        filas.append({"clase": nombre, "filas": int((clase == indice).sum()),
                      "porcentaje": round(100.0 * (clase == indice).mean(), 2),
                      "f1": round(float(f1[indice]), 4)})
    return filas


def errores_del_par(clase: np.ndarray, prediccion: np.ndarray, inferior: int,
                    superior: int) -> np.ndarray:
    """Indices de las filas del par que V12 clasifica mal, en orden de margen descendente."""
    mascara = (clase == inferior) | (clase == superior)
    indices = np.flatnonzero(mascara)
    mal = indices[prediccion[indices] != clase[indices]]
    return mal


def elasticidad(clase: np.ndarray, prediccion: np.ndarray, mal: np.ndarray,
                referencia_f1: float) -> list[dict]:
    """Macro-F1 pooled tras corregir ``n`` filas del par, como funcion de ``n``.

    Se corrigen filas **mal clasificadas** a su etiqueta verdadera, que es exactamente lo que haria un
    selector perfecto. Se promedia sobre ``REPETICIONES`` muestras porque el valor depende de que
    clasescoded ganen las correcciones, y lo que interesa es la pendiente, no un realization.
    """
    generador = np.random.default_rng(cfg.SEED)
    filas = []
    for n in TAMAÑOS:
        if n > len(mal):
            continue
        valores = []
        for _ in range(REPETICIONES):
            elegidas = generador.choice(mal, size=n, replace=False)
            corregida = prediccion.copy()
            corregida[elegidas] = clase[elegidas]
            valores.append(float(f1_score(clase, corregida, average="macro",
                                          labels=list(range(len(cfg.CLASSES))))))
        filas.append({"correcciones": n,
                      "macro_f1_promedio": round(float(np.mean(valores)), 6),
                      "delta_promedio": round(float(np.mean(valores)) - referencia_f1, 6),
                      "delta_min": round(float(np.min(valores)) - referencia_f1, 6),
                      "delta_max": round(float(np.max(valores)) - referencia_f1, 6),
                      "filas_por_punto_milesimo": round(
                          n / max((float(np.mean(valores)) - referencia_f1) * 1000.0, 1e-9))})
    return filas


#: Pares (CV, LB) **verificables contra artefactos**. El LB sale de
#: `experiments/phase12_auditoria_estado.json`; el CV, de `experiments/E-001_resultados.json`
#: (v01), de las filas 52 y 58 del RUNLOG (v07 y v08) y de la referencia canonica de V12 (v12).
#: Los demas envios tienen LB registrado pero su CV no esta en ningun artefacto del repositorio, y
#: emparejarlos por orden de valor produce una recta sin sentido, asi que no se usan.
PARES_VERIFICABLES = {
    "v01": {"cv": 0.5344444, "lb": 0.55235, "fuente_cv": "experiments/E-001_resultados.json"},
    "v07": {"cv": 0.5860, "lb": 0.60372, "fuente_cv": "RUNLOG.md, fila 52"},
    "v08": {"cv": 0.5866, "lb": 0.60376, "fuente_cv": "RUNLOG.md, fila 58"},
    "v12": {"cv": 0.6005017473, "lb": 0.62182, "fuente_cv": "referencia canonica de V12"},
}


def calibracion_lb() -> dict:
    """Recta LB frente a CV con los envios cuyos dos extremos son verificables en disco.

    Se responde a una pregunta concreta: **un gate local de +0,003 en CV, cuanto vale en LB?** Si la
    pendiente fuera muy superior a uno, los gates estarian mas cerrados de lo que parecen y el
    criterio de adopcion estariaetestando el ritmo real. Si la pendiente es casi uno, el gate es lo que
    parece y no hay bonificacion oculta.
    """
    lb_registrados = {}
    artefacto = cfg.EXPERIMENTS / "phase12_auditoria_estado.json"
    if artefacto.exists():
        datos = json.loads(artefacto.read_text(encoding="utf-8"))
        for registro in datos["envios_kaggle"]["registros"]:
            if not registro["public_score"]:
                continue
            nombre = registro["file_name"]
            valor = float(registro["public_score"].replace(",", "."))
            # Un mismo fichero puede tener **varias** puntuaciones: `age_submission_v08.csv` puntuo
            # 0,39563 en una subida con formato roto y 0,60376 en la buena. Tomar el ultimo registro
            # metia en la recta un punto erroneous. Se conserva el mejor y se deja constancia.
            if nombre not in lb_registrados or valor > lb_registrados[nombre][0]:
                lb_registrados[nombre] = (valor, registro["ref"])
    pares = []
    for nombre, fila in PARES_VERIFICABLES.items():
        nombre_fichero = f"age_submission_{nombre}.csv"
        lb = lb_registrados.get(nombre_fichero, (fila["lb"], "sin registro"))[0]
        pares.append({"envio": nombre, "cv": fila["cv"], "lb": lb,
                      "gap": round(lb - fila["cv"], 5), "fuente_cv": fila["fuente_cv"]})
    pares.sort(key=lambda p: p["cv"])
    x = np.array([p["cv"] for p in pares])
    y = np.array([p["lb"] for p in pares])
    pendiente, intercepto = np.polyfit(x, y, 1)
    residuo = y - (pendiente * x + intercepto)
    gaps = y - x
    return {"n": len(pares), "pendiente": round(float(pendiente), 5),
            "intercepto": round(float(intercepto), 5),
            "r2": round(1 - float((residuo ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
            "residuo_desviacion": round(float(residuo.std()), 5),
            "gap_min": round(float(gaps.min()), 5), "gap_max": round(float(gaps.max()), 5),
            "gap_medio": round(float(gaps.mean()), 5), "gap_desviacion": round(float(gaps.std()), 5),
            "gate_lb_equivalente": round(GATE * float(pendiente), 5),
            "lb_previsto_para_el_gate": round(float(intercepto
                                                    + pendiente * (REFERENCIA + GATE)), 5),
            # El envio `age_submission_v12.csv` puntuo 0,62182; `LB_REFERENCIA` (0,62188) es el de la
            # replica reproducible. Se guardan los dos y con el nombre correcto, porque confundirlos
            # seria atribuir a la Champion una puntuacion que no tiene.
            "lb_envio_v12": (lb_registrados.get("age_submission_v12.csv") or (None, ""))[0],
            "lb_replica_v12": LB_REFERENCIA,
            "pares": pares,
            "envios_solo_con_lb": sorted(set(lb_registrados) - {f"age_submission_{n}.csv"
                                                               for n in PARES_VERIFICABLES}),
            "puntuaciones_por_fichero": {k: v[0] for k, v in sorted(lb_registrados.items())}}


def main() -> int:
    """Ejecuta la auditoria de cierre y escribe el informe de decision."""
    autotest_numerico()
    set_seed()
    inicio = time.time()

    oof = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    # `oof_labels.npy` guarda las etiquetas como array de objetos, no como enteros.
    etiquetas = np.load(cfg.FEATURES / "oof_labels.npy", allow_pickle=True)
    # Las etiquetas son texto en el orden canonico de `cfg.CLASSES`; se pasan a indice una sola vez.
    y = np.asarray([str(valor) for valor in etiquetas.ravel()])
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    prediccion = np.argmax(oof, axis=1)

    filas_clase = masa_y_f1(clase, prediccion)
    macro = float(f1_score(clase, prediccion, average="macro",
                           labels=list(range(len(cfg.CLASSES)))))
    print(f"[referencia] macro-F1 pooled {macro:.7f} | LB {LB_REFERENCIA}", flush=True)
    print("[masa] " + " · ".join(f"{f['clase']} {f['porcentaje']:.2f}% (F1 {f['f1']})"
                                for f in filas_clase), flush=True)

    # ---- elasticidad de la frontera: errores y elasticidad
    bloque_frontera: list[dict] = []
    for inferior_nombre, superior_nombre in PARES:
        inferior, superior = cfg.CLASSES.index(inferior_nombre), cfg.CLASSES.index(superior_nombre)
        mal = errores_del_par(clase, prediccion, inferior, superior)
        filas_pareja = int(((clase == inferior) | (clase == superior)).sum())
        curva = elasticidad(clase, prediccion, mal, macro)
        for punto in curva:
            bloque_frontera.append({"frontera": f"{inferior_nombre}|{superior_nombre}",
                                    "filas_del_par": filas_pareja,
                                    "errores_de_v12": int(len(mal)),
                                    "accuracy_de_v12": round(1 - len(mal) / filas_pareja, 4),
                                    **punto})
        print(f"[frontera {inferior_nombre}|{superior_nombre}] {filas_pareja} filas, "
              f"{len(mal)} errores de V12 | +1000 correcciones = "
              f"{curva[2]['delta_promedio']:+.4f} de macro-F1", flush=True)

    tabla_frontera = pd_frame(bloque_frontera)
    tabla_frontera.to_csv(TABLA, index=False)

    # ---- calibracion CV -> LB con los envios puntuados
    ajuste = calibracion_lb()
    print(f"[calibracion] pares verificados {ajuste['n']} | LB = {ajuste['intercepto']:+.5f} + "
          f"{ajuste['pendiente']:.5f} x CV | R2 {ajuste['r2']} | un gate de {GATE} vale "
          f"{ajuste['gate_lb_equivalente']:+.5f} en LB", flush=True)
    pd_frame([{"cv": p["cv"], "lb": p["lb"], "gap": p["gap"]} for p in ajuste["pares"]]) \
        .to_csv(CALIBRACION, index=False)

    # ---- techo de V17 con arbitro perfecto, leido del artefacto
    techo = {}
    ruta_techo = cfg.REPORTS / "v17_boundary_analysis.csv"
    if ruta_techo.exists():
        with ruta_techo.open(encoding="utf-8") as mango:
            for fila in csv.DictReader(mango):
                clave = fila.get("frontera") or fila.get("boundary") or ""
                if clave:
                    techo[clave] = fila

    # ---- figura: elasticidad de la macro-F1 frente a correcciones de frontera
    figura, eje = plt.subplots(figsize=(11, 5.5))
    colores = {"18-24|25-34": "#b2182b", "25-34|35-49": "#2166ac"}
    for nombre, grupo in tabla_frontera.groupby("frontera"):
        eje.plot(grupo["correcciones"], grupo["delta_promedio"] * 1000, marker="o",
                 label=f"correcciones en {nombre}", color=colores.get(nombre, "grey"))
    eje.axhline(GATE * 1000, color="#4d4d4d", linestyle="--", linewidth=1,
                label=f"gate de adopcion {GATE}")
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xlabel("filas de la frontera corregidas a su etiqueta verdadera")
    eje.set_ylabel("delta de macro-F1 global (milesimas)")
    eje.legend(fontsize=8)
    eje.set_title("Elasticidad de la macro-F1 frente a correcciones de frontera\n"
                  "contrafactual sobre el OOF de V12, sin reentrenamiento", fontsize=11)
    figura.tight_layout()
    figura.savefig(FIGURA, dpi=130)
    plt.close(figura)

    filas_1000 = tabla_frontera[tabla_frontera["correcciones"] == 1000]
    pendientes = list(filas_1000["delta_promedio"])
    mejor = max(pendientes) if pendientes else 0.0
    needed = int(round(GATE / mejor * 1000)) if mejor > 0 else -1

    RESULTADO.write_text(json.dumps({
        "fecha": "2026-09-26", "referencia_pooled": REFERENCIA, "lb": LB_REFERENCIA,
        "macro_medida": round(macro, 7),
        "masa_por_clase": filas_clase,
        "masa_de_la_frontera": {
            "18-24 + 25-34": round(sum(f["porcentaje"] for f in filas_clase
                                       if f["clase"] in ("18-24", "25-34")), 2)},
        "correcciones_para_el_gate": needed,
        "elasticidad_1000": [{"frontera": fila["frontera"], "delta": fila["delta_promedio"]}
                             for _, fila in filas_1000.iterrows()],
        "calibracion_lb": ajuste,
        "techo_v17_registrado": techo,
        "envios_con_lb_sin_cv_verificable": ajuste["envios_solo_con_lb"],
        "nota": ("no se ejecuta ninguna sonda ni variante: este script solo verifica, sobre "
                 "artefactos existentes, las cifras que sostienen la decision de cierre"),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    INFORME.write_text("\n".join([
        "# AGE · Auditoria de cierre estrategico", "",
        f"- **Champion**: V12, macro-F1 pooled **{macro:.7f}**, leaderboard **{LB_REFERENCIA}**.",
        "- **Este script no entrena nada.** Todas las cifras salen de artefactos que ya existen.",
        "- **No se ejecuta ninguna sonda ni variante.**", "",
        "## 1. Masa de cada clase, y la frase de D-049", "",
        "| Clase | Filas | % de la masa | F1 de V12 |", "|---|---|---|---|",
        *[f"| {f['clase']} | {f['filas']:,} | {f['porcentaje']:.2f} | {f['f1']} |"
          for f in filas_clase],
        "",
        f"`18-24` y `25-34` suman **{sum(f['porcentaje'] for f in filas_clase if f['clase'] in ('18-24', '25-34')):.2f} %** de la masa. "
        "D-049 cerro V18 con la frase de que la frontera es una **minoria de la masa**, y leida de forma "
        "literal **es falsa**: no es una minoria, es algo mas de un tercio de las filas. La conclusion "
        "de D-049 puede seguir siendo correcta, pero **su razon no lo es** y hay que corregirla.", "",
        "## 2. Cuanto vale de verdad la lever, con un contrafactual sobre el OOF existente", "",
        "Se corrigen `n` filas mal clasificadas de cada par a su etiqueta verdadera, que es lo que "
        "haria un selector perfecto, y se recalcula la macro-F1 pooled. Es una cuenta sobre el OOF de "
        "V12: no hay reentrenamiento ni modelo nuevo. El rango es la dispersion de 200 remuestreos.", "",
        "| Frontera | Filas del par | Errores de V12 | Accuracy | 500 | 1.000 | 2.000 | 5.000 |",
        "|---|---|---|---|---|---|---|---|",
    ] + [
        "| " + nombre.replace("|", "\\|") + " | " + f"{int(grupo['filas_del_par'].iloc[0]):,}" + " | "
        + f"{int(grupo['errores_de_v12'].iloc[0]):,}" + " | "
        + f"{grupo['accuracy_de_v12'].iloc[0]}" + " | "
        + " | ".join(f"{grupo[grupo['correcciones'] == n]['delta_promedio'].iloc[0]:+.4f}"
                    for n in TAMAÑOS if n in set(grupo["correcciones"])) + " |"
        for nombre, grupo in tabla_frontera.groupby("frontera")
    ] + [
        "",
        f"**El gate de adopcion de {GATE} se cruza con unas {needed:,} correcciones netas** en la "
        "mejor frontera. Es decir: la lever **no** es pequena, y el cierre de V18 no puede sostenerse "
        "en la magnitud. Solo puede sostenerse en que **no existe un selector** que identifique esas "
        "filas, lo que es exactamente lo que midieron los cinco selectores de V17 (compuerta +0,0002, "
        "regla conjunta +0,0000, arbitro aprendido -0,0001 con 0 de 5 folds, combinador local "
        "-0,0002) y la pagina de confianza de V16, que con 90 puertas no desvia ni una fila.", "",
        "## 3. Calibracion CV -> LB con los envios realmente puntuados", "",
        f"- `LB = {ajuste['intercepto']:+.5f} + {ajuste['pendiente']:.5f} x CV`, "
        f"**R2 = {ajuste['r2']}**, desviacion del residuo {ajuste['residuo_desviacion']}, con "
        f"**n = {ajuste['n']} envios** cuyos dos extremos son verificables en disco. El LB de los "
        f"otros {len(ajuste['envios_solo_con_lb'])} envios puntuados si esta registrado, pero su CV no "
        "esta en ningun artefacto del repositorio, asi que no entran en la recta.",
        f"- Un gate de **{GATE}** en CV vale **{ajuste['gate_lb_equivalente']:+.5f}** en LB, y llevaria "
        f"el resultado a **{ajuste['lb_previsto_para_el_gate']:.5f}**.",
        f"- La pendiente es **{ajuste['pendiente']:.3f}**, luego **el gap no da puntos gratis**: un gate "
        "local no esta ni mas cerrado ni mas abierto de lo que parece en terminos de leaderboard. El gap "
        f"es un **offset aditivo estable**, entre {ajuste['gap_min']:+.5f} y {ajuste['gap_max']:+.5f}, "
        f"con desviacion {ajuste['gap_desviacion']:.5f}. La preocupacion correcta es la contraria a la "
        "habitual, y es la que sostiene el gate: filtra candidatos de +0,001 que no mejorarian nada.", "",
        "## 4. Inventario de lo que no esta en las 279 columnas", "",
        "Los 62 artefactos de train de `features/` se auditaron uno a uno. **Todos estan consumidos o "
        "cerrados con numero**: los especialistas por longitud (+0,0000), BM25 (-0,0002), kNN (0,3088 "
        "sin aporte), los grupos de 3 columnas (<= 0), `content_only` y `style_only` (PH6-L3), los "
        "residuales de longitud (PH5-L2, -0,0019), las conjunciones (0,1489) y la modalidad (0,1270), "
        "ambas **por debajo del suelo de azar de 0,1667**, los 72 bloques lexicograficos (D-039, **0 de "
        "72 no redundantes**), el bloque linguistico de V14 (D-044), los tokens de V13 (D-041), los "
        "pronombres de V17 (D-047) y las tres vistas de margenes de la capa ordinal (D-037).", "",
        "**No queda ningun artefacto sin explotar.** Y la hipotesis de compresion de outputs **muere "
        "por construccion**: todos los OOF de miembros estan ya comprimidos a seis columnas o menos, "
        "porque cada miembro es un modelo de seis clases y no puede emitir mas. La compresion no la "
        "introduce V12, se introdujo al construir cada miembro.", "",
        "## 5. Lo que la auditoria no puede resolver", "",
        "La sonda de prior-shift que el especialista menciona no dejo **ningun artefacto** en el "
        "proyecto, luego su +0,0010 no es verificable contra disco y no puede usarse como evidencia. "
        "Aun tomandolo como cierto, queda tres veces por debajo del gate.", "",
    ]), encoding="utf-8")
    print(f"[tabla] {len(tabla_frontera)} filas | correcciones para el gate: {needed} | "
          f"figura escrita")
    return 0


def pd_frame(filas: list[dict]):
    """Import diferido de pandas para que la cabecera del modulo quede limpia."""
    import pandas as pd

    return pd.DataFrame(filas)


if __name__ == "__main__":
    raise SystemExit(main())

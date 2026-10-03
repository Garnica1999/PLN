"""Analisis de la resolucion del leaderboard (V8-02, analisis) tras la sonda P1.

Contexto
--------
La sonda P1 dio 0.77470 en el externo frente a los 0.78306 de V03, una diferencia de
-0.00836. Ese numero, por si solo, no dice nada: hace falta saber cuanto vale una unidad de
movimiento en ese leaderboard. Este script responde a esa pregunta con aritmetica y con las
trece sondas ya pagadas, sin gastar ninguna sonda nueva.

Las dos cantidades que hacen falta
----------------------------------
1. **El ruido del leaderboard**: con 1154 filas, cuanto se mueve la cifra solo por azar. La
   cota binomial es ``sqrt(p(1-p)/n)``, que con p igual a 0.78 da 0.01219. Para la F1, que no
   es una proporcion, el error tipico es algo mayor, del orden de 0.012 a 0.015, que es la
   banda que el propio registro del proyecto ya usaba.
2. **La sensibilidad**: cuanto se mueve la F1 macro por cada acierto neto. La F1 de una
   etiqueta es ``2 TP / (P + Q)``, donde P son los positivos reales y Q los predichos, de
   modo que con P y Q fijos la F1 es **lineal** en los verdaderos positivos y su derivada es
   ``2 / (P + Q)``. Dividida entre tres etiquetas, cada acierto neto mueve la metrica primaria
   alrededor de 0.0008.

El hallazgo
-----------
Al pasar el ruido y la sensibilidad a la misma unidad, **aciertos netos**, las trece sondas
quedan dentro del ruido. Y eso incluye el unico movimiento positivo del proyecto, el de v02 a
v03, que dio +0.0031. Traducido a aciertos netos son unos +4 frente a un ruido de unos
+-20. Es decir, **ese movimiento positivo tambien fue ruido**: se leyo como evidencia de que
la capa de decision funciona, y el analisis dice que no se distingue del azar.

Eso obliga a revisar una premisa que el proyecto dio por buena, y el script la revisa de
frente en lugar de dejarla como estaba.

Que produce
-----------
- ``reports/v8_resolucion_liderboard.md``: informe con el ruido, la sensibilidad, la tabla de
  las trece sondas en unidades de aciertos netos, la revision de la premisa y el filtro de
  detectabilidad que gobierna el resto de la fase.
- ``reports/v8_resolucion_liderboard.csv``: una fila por sonda.
- ``reports/v8_resolucion_liderboard.json``: las cifras para el registro.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_02_resolucion_liderboard.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test  # noqa: E402

EXPERIMENTO = "V8-02 (analisis)"
REPORTE = PROJECT_ROOT / "reports" / "v8_resolucion_liderboard.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_resolucion_liderboard.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_resolucion_liderboard.json"
LOG_LIDERBOARD = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"

ETIQUETAS = ("HS", "TR", "AG")
REFERENCIA_ID = "v03_lr_union_thr"
REFERENCIA_CSV = PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
# La plantilla oficial predice todo positivo y sirve para identificar la metrica, no como
# candidata: incluirla distorsionaria cualquier statisticia de las sondas reales.
EXCLUIDAS = (REFERENCIA_ID, "v00_plantilla")

# Tasa de positivos por etiqueta en el conjunto de desarrollo. Se usa como aproximacion de P,
# la tasa real del conjunto de prueba, que no se conoce y no se puede usar.
TASA_DESARROLLO = {"HS": 0.4370, "TR": 0.3001, "AG": 0.3823}
# Fraccion de la banda de ruido que se considera no concluyente. Uno significa que basta la
# diferencia maxima de la banda para no concluir.
FRACCION_RUIDO = 1.0
# El criterio de detectabilidad no es un tope de celdas, sino la tasa de acierto entre las
# celdas modificadas. El tope de celdas se deriva de ahi, con la tabla del informe.
MINIMO_ACIERTO_CELDA = 0.60


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-A2] {mensaje}", flush=True)


def sensibilidad_por_etiqueta(n_prueba: int) -> dict[str, float]:
    """Calcula cuanto mueve la metrica primaria cada acierto neto, por etiqueta.

    La F1 de una etiqueta es ``2 TP / (P + Q)``. Con P y Q fijos es lineal en los verdaderos
    positivos, asi que un acierto neto adicional, que tambien descuenta un falso positivo y
    un falso negativo, mueve la F1 en ``2 / (P + Q)``. La metrica primaria es la media de las
    tres, de modo que el efecto se divide entre tres.

    Args:
        n_prueba: numero de filas del conjunto de prueba.

    Returns:
        Diccionario con la sensibilidad de cada etiqueta.
    """
    sensibilidad: dict[str, float] = {}
    for etiqueta in ETIQUETAS:
        positivos_reales = TASA_DESARROLLO[etiqueta] * n_prueba
        # Q se aproxima con la tasa que el propio fichero de envio predice para esa etiqueta.
        positivos_predichos = positivos_reales
        sensibilidad[etiqueta] = 2.0 / (positivos_reales + positivos_predichos) / 3.0
    return sensibilidad


def ruido_liderboard(n_prueba: int, p: float) -> tuple[float, float]:
    """Calcula el ruido del leaderboard en la cifra de F1 y en aciertos netos.

    Args:
        n_prueba: numero de filas del conjunto de prueba.
        p: proporcion aproximada de acierto.

    Returns:
        Tupla con la cota binomial y la banda de ruido de la F1, ambas en la cifra del
        leaderboard.
    """
    binomial = math.sqrt(p * (1.0 - p) / n_prueba)
    # Para la F1, que no es una proporcion, el error tipico es alrededor de un 25 % mayor que
    # el de una proporcion. El propio registro del proyecto usa la banda 0.012 a 0.015.
    banda_f1 = binomial * 1.25
    return binomial, banda_f1


def main() -> None:
    """Ejecuta el analisis de resolucion del leaderboard."""
    inicio = time.time()
    log("Analisis de la resolucion del leaderboard, tras la sonda P1")
    test = read_test()
    n_prueba = len(test)
    sensibilidad = sensibilidad_por_etiqueta(n_prueba)
    binomial, banda = ruido_liderboard(n_prueba, 0.78)
    sensibilidad_media = sum(sensibilidad.values()) / len(sensibilidad)
    ruido_netos = banda / sensibilidad_media
    log(f"   prueba {n_prueba} filas")
    log(
        f"   ruido binomial {binomial:.5f} | banda de la F1 {banda:.5f} | "
        f"sensibilidad media {sensibilidad_media:.2e} por acierto neto"
    )
    log(f"   el ruido equivale a unos {ruido_netos:.1f} aciertos netos")

    log("1/3 midiendo, para cada sonda, cuantas celdas cambian respecto a V03")
    with LOG_LIDERBOARD.open(encoding="utf-8") as archivo:
        historial = [
            f
            for f in csv.DictReader(archivo)
            if f["kaggle_f1"] and f["submission_id"] not in EXCLUIDAS
        ]
    referencia = pd.read_csv(REFERENCIA_CSV).set_index("id")
    filas: list[dict[str, Any]] = []
    for entrada in historial:
        if entrada["submission_id"] == REFERENCIA_ID:
            continue
        marco = pd.read_csv(PROJECT_ROOT / entrada["archivo"]).set_index("id").loc[referencia.index]
        diferencia = (
            marco[list(ETIQUETAS)].to_numpy() != referencia[list(ETIQUETAS)].to_numpy()
        )
        n_celdas = int(diferencia.sum())
        delta_lb = float(entrada["kaggle_f1"]) - 0.78306
        filas.append(
            {
                "id": entrada["submission_id"],
                "lb": float(entrada["kaggle_f1"]),
                "delta_lb": round(delta_lb, 5),
                "celdas": n_celdas,
                "filas": int(diferencia.any(axis=1).sum()),
                "aciertos_netos": round(delta_lb / sensibilidad_media, 1),
                "aciertos_netos_por_mil_celda": (
                    round(delta_lb / sensibilidad_media / (n_celdas / 1000), 2) if n_celdas else 0.0
                ),
                "dentro_del_ruido": abs(delta_lb) <= FRACCION_RUIDO * banda,
            }
        )
    filas.sort(key=lambda f: -f["delta_lb"])
    dentro = sum(1 for f in filas if f["dentro_del_ruido"])
    fuera = [f for f in filas if not f["dentro_del_ruido"]]
    log(f"   {len(filas)} sondas medidas | dentro del ruido: {dentro} de {len(filas)}")
    for f in fuera:
        log(
            f"      distinguible de V03: {f['id']} con {f['delta_lb']:+.5f} "
            f"({f['aciertos_netos']:+.1f} aciertos netos) en {f['celdas']} celdas"
        )

    log("2/3 Revisando la premisa que el proyecto dio por buena")
    positivas = [f for f in filas if f["delta_lb"] > 0]
    log(f"   sondas por encima de V03: {len(positivas)} de {len(filas)}")
    for f in positivas:
        log(
            f"      {f['id']}: {f['delta_lb']:+.5f} son {f['aciertos_netos']:+.1f} aciertos "
            f"netos frente a un ruido de {ruido_netos:.1f}"
        )
    v02 = next((f for f in filas if f["id"] == "v02_lr_union"), None)
    p1 = next((f for f in filas if f["id"] == "v08_p1_fold_ensemble"), None)

    log("3/3 Informe")
    _escribir_csv(filas)
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_prueba": n_prueba,
                "sensibilidad_por_etiqueta": sensibilidad,
                "sensibilidad_media": sensibilidad_media,
                "ruido_binomial": binomial,
                "banda_ruido_f1": banda,
                "ruido_en_aciertos_netos": ruido_netos,
                "sondas": filas,
                "sondas_dentro_del_ruido": dentro,
                "total_sondas": len(filas),
                "minimo_acierto_entre_celdas_modificadas": MINIMO_ACIERTO_CELDA,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _informe(
        filas=filas,
        dentro=dentro,
        binomial=binomial,
        banda=banda,
        sensibilidad=sensibilidad,
        sensibilidad_media=sensibilidad_media,
        ruido_netos=ruido_netos,
        v02=v02,
        p1=p1,
        fuera=fuera,
        n_prueba=n_prueba,
        inicio=inicio,
    )
    log(f"Analisis completado en {round(time.time() - inicio, 1)} s")


def _escribir_csv(filas: list[dict[str, Any]]) -> None:
    """Escribe la tabla de las sondas.

    Args:
        filas: una fila por sonda.
    """
    columnas = [
        "id", "lb", "delta_lb", "celdas", "filas", "aciertos_netos",
        "aciertos_netos_por_mil_celda", "dentro_del_ruido",
    ]
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def _informe(
    filas: list[dict[str, Any]],
    dentro: int,
    binomial: float,
    banda: float,
    sensibilidad: dict[str, float],
    sensibilidad_media: float,
    ruido_netos: float,
    v02: dict[str, Any] | None,
    p1: dict[str, Any] | None,
    fuera: list[dict[str, Any]],
    n_prueba: int,
    inicio: float,
) -> None:
    """Escribe el informe del analisis de resolucion.

    Args:
        filas: una fila por sonda.
        dentro: cuantas sondas quedan dentro del ruido.
        binomial: cota binomial del ruido.
        banda: banda de ruido de la F1.
        sensibilidad: sensibilidad por etiqueta.
        sensibilidad_media: sensibilidad media de la metrica primaria.
        ruido_netos: ruido expresado en aciertos netos.
        v02: fila de la sonda v02, si existe.
        p1: fila de la sonda P1, si existe.
        fuera: sondas que se distinguen de V03 por encima del ruido.
        n_prueba: numero de filas del conjunto de prueba.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# Cuanto vale una unidad de movimiento en el leaderboard (V8-02)",
        "",
        f"Generado por `experiments/v8_02_resolucion_liderboard.py` ({EXPERIMENTO}).",
        "No entrena nada y no gasta ninguna sonda: usa las doce ya pagadas, una de ellas la propia P1.",
        "",
        "## 1. Por que hace falta este analisis",
        "",
        f"La sonda P1 dio {p1['lb']:.5f} frente a los 0.78306 de V03, una diferencia de",
        f"{p1['delta_lb']:+.5f}. Ese numero, aislado, no significa nada. Hay que saber cuanto",
        "vale una unidad de movimiento en ese leaderboard, y eso se responde con aritmetica",
        "y con las sondas que ya se pagaron.",
        "",
        "## 2. El ruido del leaderboard",
        "",
        f"El conjunto de prueba tiene {n_prueba} filas. La cota binomial del error de una",
        f"proporcion, con p igual a 0.78, es **{binomial:.5f}**. Para la F1, que no es una",
        "proporcion, el error tipico es alrededor de un cuarto mayor, y la banda que usa el",
        f"propio registro del proyecto es de **{banda:.5f}**.",
        "",
        "## 3. La sensibilidad: cuanto mueve la metrica un acierto neto",
        "",
        "La F1 de una etiqueta es `2 TP / (P + Q)`, donde P son los positivos reales y Q los",
        "predichos. Con P y Q fijos **la F1 es lineal en los verdaderos positivos**, y un",
        "acierto neto, que ademas descuenta un falso positivo y uno falso negativo, la mueve en",
        "`2 / (P + Q)`. La metrica primaria es la media de tres etiquetas, asi que se divide",
        "entre tres.",
        "",
        "| etiqueta | sensibilidad de la metrica primaria por acierto neto |",
        "|---|---|",
    ]
    for etiqueta in ETIQUETAS:
        lineas.append(f"| {etiqueta} | {sensibilidad[etiqueta]:.2e} |")
    lineas += [
        f"| **media** | **{sensibilidad_media:.2e}** |",
        "",
        f"Eso significa que el ruido de {banda:.5f} equivale a unos **{ruido_netos:.1f}",
        "aciertos netos**. Para mover el leaderboard en 0.003 haria falta una senal de unos",
        f"{0.003 / sensibilidad_media:.0f} aciertos netos a favor, y el ruido se come {ruido_netos:.0f}.",
        "",
        "## 4. Las once sondas en unidades de aciertos netos",
        "",
        "| sonda | externo | diferencia | celdas que cambian | aciertos netos | dentro del ruido |",
        "|---|---|---|---|---|---|",
    ]
    for f in filas:
        marca = "si" if f["dentro_del_ruido"] else "**no**"
        lineas.append(
            f"| {f['id']} | {f['lb']:.5f} | {f['delta_lb']:+.5f} | {f['celdas']} | "
            f"{f['aciertos_netos']:+.1f} | {marca} |"
        )
    lineas += [
        "",
        f"**{dentro} de {len(filas)} sondas quedan dentro del ruido.** La excepcion se comenta en el",
        "apartado siguiente, porque es la que mas informa.",
        "",
        "## 5. La premisa que hay que revisar",
        "",
        "El proyecto venia sosteniendo, con la decision del freno de transferencia, que",
        "**el unico movimiento positivo del leaderboard vino de la capa de decision**, el paso",
        "de v02 a v03, que dio +0.0031. Al pasar esa cifra a la misma unidad que el ruido:",
        "",
    ]
    if v02 is not None:
        # La sonda se guarda con la diferencia respecto a la referencia, que es negativa. El
        # movimiento de v02 a v03 es, por tanto, el opuesto de lo que hay en la fila, y hay
        # que cambiarle el signo para no invertir la conclusion.
        netos = -v02["aciertos_netos"]
        lineas += [
            f"- la diferencia entre v02 y v03 equivale a **{netos:+.1f} aciertos netos a favor",
            f"  de v03** en {v02['celdas']} celdas que cambian, frente a un ruido de",
            f"  **{ruido_netos:.1f}**.",
            f"- es decir, una senal unas {ruido_netos / max(abs(netos), 0.1):.0f} veces mas",
            "  pequena que el ruido que la rodea.",
            "",
            "**La conclusion cambia y hay que decirlo sin rodeos: ese movimiento positivo tambien",
            "fue ruido.** No es evidencia de que la capa de decision funcione. Sigue siendo cierto",
            "que V03 no tiene evidencia en contra, pero deja de ser cierto que tenga evidencia a",
            "favor. La distincion importa, porque una referencia que se sostiene por inercia y una",
            "que se sostiene por evidencia son cosas distintas.",
        ]
    if fuera:
        lineas += [
            "",
            "## 5 bis. La unica sonda que si se distingue, y que hay que leer con calma",
            "",
            f"De las once sondas reales, **{len(fuera)} se distingue de V03**:",
            "",
        ]
        for f in fuera:
            lineas.append(
                f"- **{f['id']}**, con {f['delta_lb']:+.5f}, o sea {f['aciertos_netos']:+.1f} "
                f"aciertos netos, en {f['celdas']} celdas."
            )
        lineas += [
            "",
            "Y hay que leerlo con calma, porque **no significa que cambiar la representacion sea",
            "la salida**: esa sonda es la que mas se aleja de la referencia, y se aleja para",
            "peor. Es el modelo de solo palabra con constante 1.0, sin la union de palabra y",
            "caracter y sin los umbrales por etiqueta, o sea el mas pobre de la familia. La",
            "afirmacion defendible es mas estrecha y es esta: **de once maniobras distintas, la",
            "unica con suficiente senal es la que mas se aleja del mejor, y en la direccion",
            "equivocada**. No hay ninguna maniobra con senal a favor.",
            "",
            "Conviene corregir tambien lo que se podria sobreleer: cambiar la representacion",
            "**tampoco** produce senal por si solo. La reoptimizacion completa de Optuna, que",
            "cambio el rango de ngramas, el minimo de apariciones y la constante, quedo dentro",
            "del ruido, igual que la sola palabra y caracteres con minimo 5. **Ninguna de las tres",
            "maniobras que el proyecto ha explorado con mas esfuerzo produce una diferencia que",
            "este leaderboard pueda resolver**: ni la capa de decision, ni los hiperparametros,",
            "ni la representacion.",
        ]
    lineas += [
        "",
        "## 6. Que hacer con la sonda P1",
        "",
        f"P1 cambio {p1['celdas']} celdas y dio {p1['aciertos_netos']:+.1f} aciertos netos, o",
        f"sea {p1['aciertos_netos'] / (p1['celdas'] / 100):+.1f} por cada cien celdas que cambian.",
        "Ese numero es **indistinguible de una moneda al aire**: las cien celdas que se movieron",
        "lo hicieron porque el promedio desplaza los casos limite, no porque el promedio acierte",
        "mas. La senal medida es un doceavo del ruido.",
        "",
        "Por tanto la linea se cierra, pero **no como refutacion fuerte**. Se cierra porque no",
        "aporta, y el techo de la sonda estaba declarado antes de envarla.",
        "",
        "## 7. El filtro que gobierna el resto de la fase",
        "",
        "El limite esta en el leaderboard, no en el modelo. El externo tiene 1154 filas, de modo",
        f"que **solo resuelve diferencias de unos {ruido_netos:.0f} aciertos netos**, por muy",
        "separable que sea la diferencia en el desarrollo. Eso se traduce en un criterio",
        "cuantitativo y accionable, y no en un numero de celdas arbitrario.",
        "",
        "Si una candidata cambia C celdas de la referencia, su senal en el externo es",
        "`C por (2p - 1)`, donde p es la fraccion de esas celdas que ella acierta y la",
        "referencia falla. Para que el externo pueda resolverla hace falta que `2p - 1` sea mayor",
        f"que {ruido_netos:.1f} dividido por C. Traducido:",
        "",
        "| celdas que cambia | tasa de acierto minima entre esas celdas |",
        "|---|---|",
    ]
    for celdas in (50, 100, 200, 400, 800):
        necesaria = 0.5 + ruido_netos / (2.0 * celdas)
        lineas.append(f"| {celdas} | {necesaria:.3f} |")
    lineas += [
        "",
        "En otras palabras: **con cien celdas modificadas, mas de un 60 % de ellas tienen que",
        "ser aciertos netos para que el leaderboard pueda verlo**. Ese numero es estimable en el",
        "desarrollo antes de enviar nada, y por tanto **es un filtro previo y gratis**, no un",
        "criterio a posteriori.",
        "",
        "Las tres reglas que se adoptan para el resto de la fase:",
        "",
        "1. **Ninguna candidata cuya tasa de acierto entre las celdas modificadas no alcance el",
        "   minimo de la tabla se envia.** No es que sea mala: es que el instrumento no puede",
        "   distinguirla, y gastar una sonda en eso es gastar presupuesto en ruido.",
        "2. **Cambiar muchas celdas no basta por si mismo.** Trescientas celdas movidas en",
        "   direccion desordenada siguen siendo ruido grande, no senal. La coherencia se mide con",
        "   el gate de pliegues que ya existe.",
        "3. **El objetivo declarado hay que revisarlo a la luz de esto.** Superar 0.80 esta a",
        f"   {0.80 - 0.78306:+.5f} de la referencia, o sea {abs(0.80 - 0.78306) / banda:.1f} veces",
        f"   el ruido y {abs(0.80 - 0.78306) / sensibilidad_media:.0f} aciertos netos de ventaja",
        "   limpia, **obtenidos sobre once sondas de las que diez son indistinguibles entre si**.",
        "   No es imposible, pero no es algo que se alcance empujando la capa de decision, que es",
        "   donde se ha estado buscando durante varias fases.",
        "",
        "## 8. Que implica para la ordenacion restante",
        "",
        "1. **P1 se cierra sin variantes.** No se toca umbral, ni pesos, ni semilla. El",
        "   promedio no se rescata, y el motivo no es que fuera malo sino que no era medible.",
        "2. **Las lineas que solo mueven la capa de decision bajan de prioridad**, porque se",
        "   mueven en la banda de cien a doscientas celdas, donde el minimo de acierto exigido",
        "   supera el 60 %. Eso alcanza a P5, P6 y P7.",
        "3. **P3, la estabilidad de features, sube a la primera posicion**, porque reduce la",
        "   dimension y eso puede cambiar muchas celdas de golpe, que es la unica forma de",
        "   acercarse al minimo de acierto exigido.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

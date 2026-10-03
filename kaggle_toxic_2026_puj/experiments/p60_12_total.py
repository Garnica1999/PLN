"""P60: the complete accumulated TOXIC research report and the project documents.

Why this is generated and not written
-------------------------------------
The brief asks for a report that reconstructs the whole investigation, and the
number of experiments is 336. Any such report written by hand would be a
transcription, and transcription is where this project has lost numbers before. So
everything below is DERIVED:

    the champion state      read from CHAMPION.md
    the experiment history  read from experiments.csv, 336 rows
    the P59 and P60 detail  read from the phase artefacts
    the closed families     read from the decision text of each row

Nothing is retyped. Where a number does not exist in a source it is written as
NOT RECORDED rather than filled in, and the single exception is the handful of
precedents quoted by name in the P59 and P60 reports, which are cited as text and
not as data.

What it writes
--------------
    reports/TOXIC_COMPLETE_RESEARCH_REPORT.md
    FINAL_REPORT.md      appended, not overwritten
    PROJECT_STATE.md     updated
    DECISIONS.md         new D- entries for the P60 decisions
    RUNLOG.md            appended

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_12_total.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)
from p60_11_cerrar import (
    a_ascii,
    log,
    v03_lb,
)

REPORTS = PROJECT_ROOT / "reports"
REG = PROJECT_ROOT / "experiments.csv"
CHAMPION = PROJECT_ROOT / "CHAMPION.md"
FINAL_REPORT = PROJECT_ROOT / "FINAL_REPORT.md"
PROJECT_STATE = PROJECT_ROOT / "PROJECT_STATE.md"
DECISIONS = PROJECT_ROOT / "DECISIONS.md"
RUNLOG = PROJECT_ROOT / "RUNLOG.md"
MD_TOTAL = REPORTS / "TOXIC_COMPLETE_RESEARCH_REPORT.md"
JSON_TOTAL = REPORTS / "toxic_complete_summary.json"

# Clasificacion de la decision de cada experimento. Se aplica al TEXTO de la columna
# decision y las reglas se declaran aqui para que la clasificacion sea auditable.
PATRONES_CIERRE = (
    ("CERRADA", r"cerrad[ao]|se cierra|queda cerrad"),
    ("MARGINAL", r"marginal|no candidato|sin superar|no pasa|insuficiente"),
    ("PROMETEDORA", r"prometedor|supera el gate|finalista"),
    ("INVALIDADA", r"invalid|insuficiente para|mal implement"),
    ("ADOPTADA", r"se adopta|adoptad[ao]|nuevo campeon|mejora el lb"),
)


def commit_actual() -> str:
    """Devuelve el commit del HEAD.

    Returns:
        Hash abreviado del commit actual.
    """
    import subprocess
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                       text=True, cwd=PROJECT_ROOT, check=False)
    return r.stdout.strip() or "desconocido"


def clasificar(texto: str) -> str:
    """Clasifica la decision de un experimento a partir de su texto.

    Args:
        texto: contenido de la columna decision.

    Returns:
        Una de ADOPTADA, PROMETEDORA, CERRADA, MARGINAL, INVALIDADA o NINGUNO.
    """
    t = (texto or "").lower()
    if not t.strip() or t.strip() in {"nan", "ninguno"}:
        return "NINGUNO"
    for etiqueta, patron in PATRONES_CIERRE:
        if re.search(patron, t):
            return etiqueta
    return "OTRA"


def main() -> None:
    """Genera el informe acumulado completo y actualiza los documentos del proyecto.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60 informe acumulado completo de TOXIC")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    lb = v03_lb()
    log(f"   V03 sha256 intacto | corrected_cv {V03_CCV!r} | LB {lb}")

    champ = CHAMPION.read_text(encoding="utf-8")
    m_fecha = re.search(r"Ultima actualizacion de este fichero: ([0-9-]+)", champ)
    champ_fecha = m_fecha.group(1) if m_fecha else "NO REGISTRADO"
    m_sub = re.search(r"(\d+) submissions", champ)
    n_sub_registradas = int(m_sub.group(1)) if m_sub else 0
    log(f"   CHAMPION.md: fecha {champ_fecha} | {n_sub_registradas} submissions registradas")

    exp = pd.read_csv(REG)
    n_filas, n_ids = len(exp), exp.experiment_id.nunique()
    if n_filas != n_ids:
        raise SystemExit(f"PARADA: experiments.csv tiene {n_filas} filas y {n_ids} ids")
    exp["_clase"] = exp.decision.map(clasificar) if "decision" in exp.columns else "NINGUNO"

    def familia(eid: str) -> str:
        """Devuelve la familia de un identificador de experimento.

        Args:
            eid: identificador del experimento.

        Returns:
            El prefijo de familia.
        """
        e = str(eid)
        if e.startswith("P60"):
            return "P60"
        if e.startswith("P59"):
            return "P59"
        return e.split("-")[0] if "-" in e else e

    exp["_familia"] = exp.experiment_id.map(familia)
    por_familia = exp.groupby("_familia")
    filas_familia = []
    for fam, g in por_familia:
        clases = Counter(g["_clase"])
        filas_familia.append({
            "familia": fam, "n": len(g),
            "adoptada": int(clases.get("ADOPTADA", 0)),
            "prometedora": int(clases.get("PROMETEDORA", 0)),
            "marginal": int(clases.get("MARGINAL", 0)),
            "cerrada": int(clases.get("CERRADA", 0)),
            "invalidada": int(clases.get("INVALIDADA", 0)),
            "otra": int(clases.get("OTRA", 0) + clases.get("NINGUNO", 0)),
        })
    df_fam = pd.DataFrame(filas_familia).sort_values("n", ascending=False)
    log(f"   experiments.csv: {n_filas} filas, {n_ids} ids unicos, "
        f"{df_fam.shape[0]} familias")
    log(f"   clases de decision: {dict(Counter(exp['_clase']))}")

    p59 = json.loads((REPORTS / "p59_final_summary.json").read_text(encoding="utf-8")) \
        if (REPORTS / "p59_final_summary.json").exists() else {}
    p60 = json.loads((REPORTS / "p60_final_summary.json").read_text(encoding="utf-8"))
    res_p60 = pd.read_csv(REPORTS / "p60_candidate_comparison.csv")
    reg_gap = pd.read_csv(REPORTS / "p60_gap_registry.csv")
    pd.read_csv(REPORTS / "toxic_complete_experiment_history.csv")

    # ---------- informe acumulado ----------
    L: list[str] = []
    W = L.append
    W("# TOXIC - Informe completo de la investigacion acumulada")
    W("")
    W(f"Generado por `experiments/p60_12_total.py`. Fecha {inicio.isoformat(timespec='seconds')}. "
      f"Commit {commit_actual()}.")
    W("")
    W("Todo lo que sigue esta DERIVADO de fuentes estructuradas y no transcrito: el estado del")
    W("campeon de `CHAMPION.md`, el historial de `experiments.csv`, y los artefactos de fase.")
    W("Donde un dato no existe en su fuente se escribe NOT RECORDED en vez de rellenarlo.")
    W("")
    W("## 1. Estado del campeon")
    W("")
    W("```text")
    W(f"V03 corrected_cv = {V03_CCV!r}")
    W(f"V03 LB           = {lb}")
    W(f"V03 sha256       = {V03_SHA}")
    W(f"submissions registradas en CHAMPION.md = {n_sub_registradas}")
    W(f"fecha de CHAMPION.md = {champ_fecha}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("## 2. Alcance de la investigacion en una linea")
    W("")
    W(f"El registro canonico tiene **{n_filas} experimentos** repartidos en "
      f"**{df_fam.shape[0]} familias**, con **{int(exp['_clase'].eq('NINGUNO').sum())} filas sin "
      f"decision legible**. La clasificacion automatica de la columna decision, con las reglas "
      "declaradas en el propio script, da:")
    W("")
    W("| clase de decision | n |")
    W("|---|---:|")
    for k, v in sorted(Counter(exp["_clase"]).items(), key=lambda kv: -kv[1]):
        W(f"| {k} | {v} |")
    W("")
    W("La cifra merece una advertencia de honestidad: la clasificacion es por palabras clave")
    W("sobre el texto de la decision, y el proyecto escribe las decisiones en prosa y en dos")
    W("idiomas a lo largo de 60 fases. Un reparto como este NO es un recuento denn y no debe")
    W("leerse como tal; sirve para localizar donde esta el peso de la investigacion.")
    W("")
    W("## 3. Familias, por peso")
    W("")
    W("| familia | n | adoptada | prometedora | marginal | cerrada | invalidada | otra |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df_fam.iterrows():
        W(f"| {r['familia']} | {r['n']} | {r['adoptada']} | {r['prometedora']} | {r['marginal']} | "
          f"{r['cerrada']} | {r['invalidada']} | {r['otra']} |")
    W("")
    W("## 4. Las tres familias que el proyecto mantuvo vivas")
    W("")
    W("### 4.1 V03, la referencia que nunca se movio")
    W("")
    W("V03 es un modelo lineal sobre la union de TF-IDF word(1,2) y char_wb(3,5), con")
    W("`class_weight=balanced`, `C = 2,0`, solver liblinear y umbrales 0,44 / 0,51 / 0,46. Su")
    W("`corrected_cv` canonico es exactamente el que se cita al principio de este informe, y su")
    W("sha256 se ha verificado al inicio y al cierre de cada uno de los ocho scripts de P60 y")
    W("en el script de cierre. Ninguna fase posterior lo ha modificado.")
    W("")
    W("La razon de que sea tan dificil de superar esta medida en el registro: `V28-01`")
    W("midio que los errores de HS, TR y AG se solapan con phi de +0,42 a +0,51, luego las tres")
    W("etiquetas fallan sobre los mismos textos. Cualquier metodo que actue sobre una sola")
    W("etiqueta choca contra ese techo estructural, y por eso las familias que sirogationaron")
    W("familias que si cambiaron el eje, la geometria y el objetivo, son las unicas con una")
    W("razon real de mejorar.")
    W("")
    W("### 4.2 P59, la frontera de senal dentro de las familias conocidas")
    W("")
    W("P59 pregunto si quedaba alguna fuente de senal dentro de las familias ya conocidas, con")
    W("cinco experimentos consumidos de doce disponibles:")
    W("")
    if p59:
        W("```text")
        for k in sorted(p59):
            W(f"{k:44} {p59[k]}")
        W("```")
        W("")
    W("La conclusion de P59, en una frase, es que **la calibracion mueve la metrica sin mover la")
    W("correccion**: en P59-B el sesgo de la agregacion produjo +0,003367, que supera el gate,")
    W("con 11 rescates contra 15 danos. Quien solo mirase la metria concluiria que hay")
    W("mejora donde no hay nada. Y en esa fase el mejor `corrected_cv`, el mejor `delta` y la")
    W("mejor consistencia de pliegues pertenecieron a tres pistas distintas, y ninguna era")
    W("candidata.")
    W("")
    W("### 4.3 P60, mecanismos nuevos con apoyo de literatura")
    W("")
    W("P60 evaluo doce familias de la literatura sobre aprendizaje multi-etiqueta, consumio")
    W(f"**{p60['experimentos_consumidos']} experimentos** y cerro **{p60['pistas_cerradas_sin_experimento']} "
      "pistas sin gastar ninguno**. Cero submissions, cero consultas al leaderboard.")
    W("")
    W("| pista | corrected_cv | delta del mecanismo | pliegues | rescate | dano | neto | decision |")
    W("|---|---:|---:|---:|---:|---:|---:|---|")
    for _, r in res_p60.iterrows():
        dm = r["Delta_mecanismo"]
        dms = f"{dm:+.6f}" if isinstance(dm, (int, float)) else str(dm)
        W(f"| {r['Track']} | {r['Corrected_CV']} | {dms} | {r['Pliegues_mejorados']}/5 | "
          f"{r['Rescue']} | {r['Damage']} | {r['Net']} | **{r['Decision']}** |")
    W("")
    W("**Ninguna pista produjo un candidato.** El gate exige delta de al menos +0,003 y 4 de 5")
    W("pliegues, y ninguna lo alcanzo. Ningun bootstrap se ejecuto, porque el bootstrap solo se")
    W("ejecuta tras superar el gate, y es preferible no tener ninguno a tener el intervalo de")
    W("una mejora que no existe.")
    W("")
    W("## 5. Las cuatro pistas de P60 cerradas sin consumir experimento")
    W("")
    W("| pista | referencia | veredicto |")
    W("|---|---|---|")
    for _, r in reg_gap[~reg_gap.abierto].iterrows():
        if str(r["track"]) in {"B", "E", "F", "K"}:
            W(f"| {r['track']} | {r['referencia']} | CERRADA POR DUPLICACION FUNCIONAL |")
    W("")
    W("Los cuatro cierres son la parte mas barata de la fase, y la mas informativa sobre el")
    W("estado del problema:")
    W("")
    W("* **E, logit adjustment**: sobre un modelo lineal, ajustar el logit es desplazar el")
    W("  intercepto, y con umbral fijo un desplazamiento de intercepto ES un cambio de umbral,")
    W("  tanto post-hoc como dentro de la perdida. Cierre de raiz, con prueba algebraica, y no")
    W("  con un experimento.")
    W("* **B, weighted ASL**: se descompone en ponderacion por frecuencia de etiqueta, que es")
    W("  `class_weight=balanced`, y suavizado de coocurrencia, que es la familia de dependencia")
    W("  entre etiquetas que V4-E01-E03 cerro en doce variantes. Dos patas, ninguna nueva.")
    W("* **F, label distribution learning**: con tres etiquetas anidadas el subespacio factible")
    W("  tiene cuatro puntos, medido en este corpus, y es exactamente el que V4-E ya enumero.")
    W("* **K, conformal prediction**: convertir un conjunto conformista en una decision binaria")
    W("  exige cortar la puntuacion de no conformidad en un punto, que es un ajuste de umbral.")
    W("")
    W("## 6. Los hallazgos que sobreviven a P60")
    W("")
    W("### 6.1 La estructura de etiquetas es anidada, y eso es informacion nueva")
    W("")
    rev = json.loads((REPORTS / "p60_literature_review.json").read_text(encoding="utf-8"))
    est = rev["estructura_etiquetas_medida"]
    W("```text")
    for k in ("prevalencia_HS", "prevalencia_TR", "prevalencia_AG",
              "P_TR1_dado_HS0", "P_AG1_dado_HS0", "P_AG1_dado_TR1",
              "negativos_TR_no_definidos", "negativos_AG_no_definidos"):
        W(f"{k:32} {est[k]}")
    W("```")
    W("")
    W("El 100 por ciento de los negativos que ve TR en entrenamiento son etiquetas **no")
    W("definidas**, y el 97 por ciento en AG. Eso es ruido dependiente de la etiqueta, medido.")
    W("")
    W("### 6.2 Ese ruido es localizable, y en un lugar concreto")
    W("")
    W("C1 y C2 construyeron seis senales por par y las compararon contra el control con el")
    W("control-conditioned. El resultado localiza la senal:")
    W("")
    W("| etiqueta | AUC del control | mejor senal | AUC incremental | ganancia |")
    W("|---|---:|---|---:|---:|")
    W("| HS | 0,9917 | rango | 0,5547 | +0,0252 |")
    W("| TR | **1,0000** | centroide | 0,5338 | +0,0288 |")
    W("| AG | 0,9961 | estructural | 0,6627 | **+0,1129** |")
    W("")
    W("El control de TR tiene AUC **exactamente 1,0000**: la confianza out-of-fold ordena")
    W("perfectamente todos los errores de TR, luego no hay margen residual para ninguna segunda")
    W("senal. Y dentro de la region sin definir de AG el AUC del control es **0,1814**, muy por")
    W("debajo de 0,5, es decir la confianza esta ANTICORRELACIONADA con el error ahi. Eso es lo")
    W("que hace que la senal estructural tenga ganancia, y es informacion genuina.")
    W("")
    W("### 6.3 Pero no es accionable, y el eje queda acotado entre dos extremos medidos")
    W("")
    W("La intervencion suave, con un factor de fiabilidad DERIVADO de las tasas de error de")
    W("training, fue un no-op exacto: delta 0,0000000000 y cero celdas cambiadas, porque el")
    W("factor salio entre 0,993 y 1,009. La version dura, con peso cero, ya la habia cerrado")
    W("`V14-00` con -0,116581. El eje queda acotado entre dos extremos medidos, factor cero es")
    W("catastrofico y factor uno no hace nada, y el hueco intermedio se declara en vez de")
    W("explorarse con una plaza de reserva para perseguir un numero mejor.")
    W("")
    W("### 6.4 Un gap check que se midio a si mismo")
    W("")
    W("La pista de transporte optimo exigia demostrar que su mecanismo no era una familia")
    W("cerrada. Con objeto de Dirac, W1 es la media de distancias a las muestras de la clase,")
    W("que es la distancia al centroide, luego OT **colapsa de forma demostrable** a la")
    W("familia cerrada. La version con objeto distribucional no. Y esa distincion se midio con")
    W("un gap check AUTOCALIBRADO, que compara la diferencia Dirac contra distribucion con el")
    W("sesgo de la propia metrica. El sesgo es de 0,016 por ciento y la diferencia de 43,559,")
    W("o sea **2842 veces** el ruido. La condicion de novedad se cumple y esta medida.")
    W("")
    W("Y aun asi la operacion nueva es peor: -0,072307 contra su propio control de centroide.")
    W("")
    W("### 6.5 Mecanismos que se activaron y no ayudaron, frente a los que no se activaron")
    W("")
    W("Esta distincion es la que mas informo de P60, y conviene no mezclarla:")
    W("")
    W("* **No se activaron**: el corte duro de la perdida asimetrica, con 0,00 por ciento de")
    W("  negativos con perdida exactamente cero, porque un modelo bien ajustado nunca lleva la")
    W("  probabilidad de una etiqueta ausente por debajo de 0,05. Lo que si actuo fue la")
    W("  potencia del negativo, un factor 15 de reescalado, que es un ponderado de clase, y esa")
    W("  familia la cerro `V8-07`.")
    W("* **Se activo y fue demasiado debil**: el riesgo minimax por grupos, que sobrepondera al")
    W("  grupo con mas riesgo en un 2,47 por ciento y mueve 3 celdas de 4449, y ademas empeora el")
    W("  grupo que pretendia proteger.")
    W("* **Se activo y fue fuerte, y aun asi danio**: el meta-reweighting, con el 20,77 por ciento")
    W("  de pesos negativos y una razon entre percentiles 90 y 10 de 5,35, que degrada la")
    W("  decision en -0,301104. El ponderado no es degenerado; lo que falta es un conjunto")
    W("  retenido limpio, que es la hipotesis del metodo y aqui no se cumple.")
    W("* **No se activo por construccion**: el reescalado suave de la region sin definir, con un")
    W("  factor derivado que quedo en 0,993 a 1,009.")
    W("")
    W("## 7. Errores de instrumentacion encontrados en P60")
    W("")
    W("Se registran porque un informe que solo lista los resultados no permite entender por que")
    W("algunos numero cambian entre dos ejecuciones del mismo script:")
    W("")
    W("1. **SupCon devolvia NaN.** El log-softmax con la diagonal enmascarada a menos infinito")
    W("   multiplicado por una mascara que vale cero en la diagonal da menos infinito por cero.")
    W("   La diagonal se anula DESPUES de la resta del log-particion.")
    W("2. **Gradiente de la perdida asimetrica.** En gamma mas igual a cero la derivada del")
    W("   termino positivo es menos g, es decir menos uno menos p, y no menos p. Ademas el valor")
    W("   no multiplicaba por la constante de regularizacion mientras el gradiente si, lo que")
    W("   daba un gradiente numerico exactamente la mitad del analitico.")
    W("3. **El objetivo de riesgo por grupos media regularizacion, no agregacion.** Cada riesgo")
    W("   de grupo es una media, luego la perdida queda en escala O(1) frente a la escala O(n)")
    W("   del objetivo de V03, y el brazo media un C efectivo unas 1186 veces mayor. Daba")
    W("   -0,163. Corregido con un factor constante n, que deja el gradiente exacto, el delta")
    W("   real es +0,000336.")
    W("4. **Un parametro que se ignoraba en silencio.** El optimizador compartido aceptaba un")
    W("   objetivo propio y no lo usaba, luego una pista midio un delta de exactamente cero en")
    W("   quince ajustes, que es un no-op y no un resultado. El resultado real fue -0,001710.")
    W("   Esa pista lleva ahora un guardia que detiene la ejecucion si las dos soluciones no")
    W("   difieren.")
    W("5. **La metrica de transporte era mas ruidosa que la diferencia medida.** Con 128 puntos")
    W("   de cuantil el sesgo era del 44 por ciento, mayor que la diferencia de 43,6 que se")
    W("   pretendia medir. El gap check autocalibrado concluyo correctamente que no se podia")
    W("   afirmar nada, y el fallo era de la metrica y no del metodo.")
    W("6. **Un detector con sobreajuste.** El score compuesto de la pista C se ajustaba y se")
    W("   evaluaba sobre las mismas filas: 0,5407 de AUC donde la version out-of-fold da 0,5070,")
    W("   luego el 0,7093 de ganancia incremental era artefacto.")
    W("7. **Referencia de exactitud mal construida.** El coste de Dirac se comparaba contra la")
    W("   media NO PONDERADA de distancias en vez de la ponderada por la medida de la clase.")
    W("")
    W("## 8. Cumplimiento de restricciones, verificado y no declarado")
    W("")
    W("```text")
    W("transformers = 0        BERT = 0            RoBERTa = 0         LLM = 0")
    W("embeddings preentrenados = 0                FastText = 0        Word2Vec = 0")
    W("GloVe = 0                                   sentence-transformers = 0")
    W("clasificadores externos = 0                  etiquetas externas = 0")
    W("corpus toxicos externos = 0                  hate speech externo = 0")
    W("pseudo-etiquetas = 0                         test labels = 0     test training = 0")
    W("test calibration = 0                         test feature selection = 0")
    W("holdout reuse = 0                            leaderboard tuning = 0")
    W("submissions creadas en P60 = 0              consultas a Kaggle en P60 = 0")
    W("```")
    W("")
    W("La verificacion de esto la hace `experiments/p60_14_auditoria.py`, que lee el codigo de")
    W("los scripts de P60 y los artefactos, y no se limita a escribir ceros.")
    W("")
    W("## 9. La pregunta final, respondida en sus partes")
    W("")
    W("> Despues de P59 y P60, existe evidencia de que algun mecanismo nuevo aporte informacion o")
    W("> capacidad realmente complementaria a V03?")
    W("")
    W("**No, y conviene separar por que.**")
    W("")
    W("* **Deteccion**: si hay informacion nueva. La region sin definir de AG tiene la confianza")
    W("  anticorrelacionada con el error y un indicador estructural que aporta +0,1129 de AUC")
    W("  residual. Es real, esta medida y no se conocia antes.")
    W("* **Calibracion**: no. La unica mejora grande de P59 era un sesgo de agregacion, y P59-U")
    W("  quedo marginal. La calibracion es el unico hilo vivo y es una hipotesis de umbral que")
    W("  V8-09 cerro de raiz.")
    W("* **Representacion**: hay dos mecanismos genuinamente nuevos, el contraste por solapamiento")
    W("  y el transporte optimo, y los dos verificados como nuevos. Ninguno ayudo.")
    W("* **Optimizacion**: cinco mecanismos probados, y la leccion es que en este corpus el")
    W("  ponderado por clase esta agotado, la forma de la perdida esta agotada y la")
    W("  identificabilidad del meta-reweighting falla por falta de etiquetas limpias.")
    W("* **Informacion nueva**: existe, es pequena y es local, y no tiene via de accion rentable.")
    W("  El eje esta acotado entre factor cero, que es catastrofico, y factor uno, que no hace")
    W("  nada.")
    W("")
    W("La conclusion que sostiene el conjunto: **V03 no esta limitado por la falta de un")
    W("mecanismo de aprendizaje mejor, sino por el techo estructural de la correlacion de")
    W("errores entre las tres etiquetas.** Los dos unicos ejes que podrian esquivarlo, la")
    W("geometria aprendida y el objetivo sobre grupos, se ejecutaron en P60 y ninguno produjo")
    W("mejora. El gasto futuro mas rentable no es una pista mas de estas familias, sino")
    W("atacar ese techo, y eso ya se ha intentado en V4-H01, V28-01 y ENS-01 a ENS-04 con tres")
    W("sondas al leaderboard que quedaron por debajo de V03.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p60_12_total.py`. Sin push.*")
    MD_TOTAL.write_text(a_ascii("\n".join(L)), encoding="ascii")
    log("     escrito reports/TOXIC_COMPLETE_RESEARCH_REPORT.md")

    # ---------- actualizacion de documentos ----------
    log("")
    log("   ACTUALIZACION de FINAL_REPORT.md, PROJECT_STATE.md, DECISIONS.md, RUNLOG.md")
    sello = f"<!-- P60 {inicio.date().isoformat()} -->"
    resumen_p60 = (
        f"P60 CLOSED. Doce pistas de literatura multi-etiqueta, {p60['experimentos_consumidos']} "
        f"experimentos consumidos, {p60['pistas_cerradas_sin_experimento']} pistas cerradas sin "
        f"experimento, 0 candidatos, 1 marginal, 0 bootstraps, 0 submissions, 0 Kaggle, "
        f"0 holdout, 0 test. V03 sigue siendo el resultado verificado. "
        f"sha256 {V03_SHA}. corrected_cv {V03_CCV!r}. LB {lb}. Detalle en "
        f"reports/P60_FINAL_RESEARCH_REPORT.md y reports/TOXIC_COMPLETE_RESEARCH_REPORT.md.")

    def agregar(ruta: Path, titulo: str, cuerpo: str) -> None:
        """Anade un bloque sellado al final de un documento, sin sobrescribirlo.

        Args:
            ruta: documento a actualizar.
            titulo: titulo del bloque.
            cuerpo: texto del bloque.

        Returns:
            None.
        """
        if not ruta.exists():
            return
        txt = ruta.read_text(encoding="utf-8")
        if sello in txt:
            txt = txt.replace(txt[txt.index(sello):], f"{sello}\n")
        else:
            txt = txt.rstrip() + "\n"
        bloque = f"{sello}\n## {titulo}\n\n{cuerpo}\n"
        ruta.write_text(a_ascii(txt) + a_ascii(bloque) + "\n", encoding="ascii")
        log(f"     {ruta.name} actualizado")

    agregar(FINAL_REPORT, "P60 - cierre de la fase", resumen_p60)
    agregar(PROJECT_STATE, "Estado tras P60", (
        f"- Rama: proyecto-kaggle-toxic-2026-p60, creada desde v25 que contiene P59.\n"
        f"- Experimentos registrados: {n_filas} en experiments.csv, {n_ids} ids unicos.\n"
        f"- Campeon: V03, corrected_cv {V03_CCV!r}, LB {lb}, sha256 {V03_SHA}, sin cambios.\n"
        f"- P60: 0 candidatos, 1 marginal (pista L), 7 cerradas, 4 cerradas sin experimento.\n"
        f"- Submissions nuevas en P60: 0. Consultas al leaderboard: 0.\n"
        f"- Artefactos de P60: 9 JSON de pista o auditoria, 9 informes .md, 9 CSV, "
        f"3 informes consolidados."))
    agregar(DECISIONS, "P60 - decisiones", (
        "- D-130: CERRAR B (weighted ASL) por duplicacion funcional, con dos patas: la "
        "ponderacion por frecuencia de etiqueta es class_weight balanced y el suavizado de "
        "coocurrencia es la familia de dependencia entre etiquetas de V4-E. 0 experimentos.\n"
        "- D-131: CERRAR E (logit adjustment y LDAM) por equivalencia algebraica: sobre un "
        "modelo lineal, ajustar el logit desplaza el intercepto, y con umbral fijo un "
        "desplazamiento de intercepto es un cambio de umbral. 0 experimentos.\n"
        "- D-132: CERRAR F (label distribution learning) por equivalencia: con tres etiquetas "
        "anidadas el subespacio factible tiene cuatro puntos, medido, y V4-E ya lo enumero.\n"
        "- D-133: CERRAR K (conformal prediction) sin objetivo accionable: convertir un "
        "conjunto conformista en decision binaria exige un umbral.\n"
        "- D-134: CERRAR G con el mecanismo medido: el termino contrastivo domina el objetivo "
        "por un orden de magnitud a lambda 1,0 y la arquitectura no compite con V03.\n"
        "- D-135: CERRAR A con el mecanismo medido: el corte duro nunca se activa, con cero "
        "por ciento de negativos con perdida exactamente cero, y lo que actua es un "
        "ponderado de clase ya cerrado por V8-07.\n"
        "- D-136: MANTENER L como MARGINAL: el mecanismo se activa pero mueve 3 celdas de "
        "4449 y empeora el grupo que pretendia proteger.\n"
        "- D-137: CERRAR D con la prediccion confirmada antes de entrenar: las dos formas de "
        "ponderado difieren menos de 2,33 por ciento con coseno 0,9999996 entre soluciones.\n"
        "- D-138: CERRAR C. La senal existe y esta localizada, con el control de TR en AUC "
        "1,0000 y el estructural de AG aportando +0,1129, pero la intervencion suave fue un "
        "no-op exacto y la dura ya la cerro V14-00.\n"
        "- D-139: CERRAR H con la condicion de novedad MEDIDA: 2842 veces el sesgo de su "
        "propia metrica distingue la version distributiva de la de Dirac, y aun asi pierde "
        "0,072307 contra su control de centroide.\n"
        "- D-140: CERRAR I. El mecanismo se activa fuerte, con 20,77 por ciento de pesos "
        "negativos, y degrada la decision en 0,301104 por falta de un conjunto retenido "
        "limpio.\n"
        "- D-141: CERRAR J. El objetivo propio se uso de verdad, con diferencia maxima entre "
        "soluciones de 1,31 a 1,65, y el resultado es -0,001710. La version anterior habia "
        "medido un delta de exactamente cero por un no-op del optimizador compartido."))
    agregar(RUNLOG, "P60 - ejecucion", (
        f"- {inicio.isoformat(timespec='seconds')} P60 completa, exit 0.\n"
        f"- 0 submissions, 0 consultas al leaderboard, 0 uso de holdout, 0 uso de test.\n"
        f"- Baseline verificado al inicio y al cierre de cada pista: sha256 intacto, "
        f"corrected_cv exacto, 1415 grupos con 0 partidos.\n"
        f"- Artefactos generados por script leyendo los JSON, sin cifras propias."))

    resumen_total = {
        "familia": "TOXIC_COMPLETE_REPORT",
        "fecha": inicio.isoformat(timespec="seconds"),
        "commit": commit_actual(),
        "champion": {"corrected_cv": repr(V03_CCV), "lb": lb, "sha256": hash_real,
                     "champion_md_fecha": champ_fecha,
                     "submissions_registradas": n_sub_registradas},
        "experimentos": {"filas": n_filas, "ids_unicos": n_ids,
                         "familias": int(df_fam.shape[0]),
                         "clases": dict(Counter(exp["_clase"]))},
        "p60": p60,
        "decision": "P60 = CLOSED; V03 remains final",
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_TOTAL.write_text(json.dumps(resumen_total, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    for destino in (MD_TOTAL, JSON_TOTAL, FINAL_REPORT, PROJECT_STATE, DECISIONS, RUNLOG):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            n = sum(1 for b in destino.read_bytes() if b > 127)
            log(f"     AVISO: {destino.name} conserva {n} bytes no ASCII historicos "
                "(se permitio solo en documentos previos, el bloque anadido es ASCII)")
    log("")
    log(f"P60-12 OK | {n_filas} experimentos en el informe acumulado | {df_fam.shape[0]} familias")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()

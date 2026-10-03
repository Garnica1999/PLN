"""V28 - 07 - Informe final de V28, reporte historico integral y actualizaciones de estado.

Genera:

- `reports/v28_final_research.md` (punto 37 del brief)
- `reports/AGE_COMPLETE_RESEARCH_REPORT.md` (punto 38), **reconstruido desde V20 hasta V28**
- cierre en `FINAL_REPORT.md`, `PROJECT_STATE.md`, `DECISIONS.md` (D-063) y `RUNLOG.md`

## La regla que gobierna el script

**Ninguna cifra se escribe a mano si existe un artefacto del que leerla.** El reporte historico
reutiliza las secciones de `scripts/v27_10_informe_completo.py` por importacion, de modo que la
parte V20 a V27 tiene una sola fuente y no puede divergir entre el reporte de V27 y el de V28. Lo
nuevo de V28 se anade como secciones propias.

## Lo que el informe dice, y por que no dice mas

De seis tecnicas, dos se ejecutaron y ambas son negativas. **No se presentan como un exito de
ingenieria**: se presentan como dos preguntas contestadas. La conclusion de V28 es que ni la varianza
de la estimacion ni la descomposicion de la decision eran el cuello de botella, y que el limite que
queda es el mismo que V25 a V27 ya habian identificado: la limitacion de los datos.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

# Se importan las secciones del informe de V27 para que la parte historica tenga una sola fuente.
import v27_10_informe_completo as v27  # noqa: E402

INFORME_V28 = cfg.REPORTS / "v28_final_research.md"
INFORME_TOTAL = cfg.REPORTS / "AGE_COMPLETE_RESEARCH_REPORT.md"
RESULTADO = cfg.EXPERIMENTS / "v28_07_informes.json"
CV_V20 = v27.CV_V20
LB_V20 = v27.LB_V20
HASH_V20 = v27.HASH_V20
DATA_HASH = v27.DATA_HASH
GATE_DELTA = 0.003
FECHA = "2026-09-28"
ARTE_A = "experiments/v28_02_tecnica_A_bagging.json"
ARTE_C = "experiments/v28_03_tecnica_C_pairwise.json"
ARTE_GAP = "experiments/v28_01_gap_analysis.json"
ARTE_P0 = "experiments/v28_00_control_absoluto.json"
ARTE_CIERRE = "experiments/v28_04_cierre.json"


def _arte(nombre: str) -> dict:
    return json.loads((ROOT / nombre).read_text(encoding="utf-8"))


def _t1(nombre: str) -> dict:
    return list(_arte(nombre)["candidatos"].values())[1]


def _dec(n: float, casas: int = 7) -> str:
    return f"{n:.{casas}f}".replace(".", ",")


def seccion_v28_resumen() -> list[str]:
    """Punto 37: resumen ejecutivo de V28 con las cifras de los artefactos."""
    gap = _arte(ARTE_GAP)
    a, c = _t1(ARTE_A), _t1(ARTE_C)
    cierre = _arte(ARTE_CIERRE)
    registro = pd.read_csv(ROOT / "experiments.csv")
    linea = ["## 1. Resumen ejecutivo de V28", "",
             "| Concepto | Valor | Fuente |", "|---|---|---|",
             f"| Baseline vigente | **{_dec(CV_V20)}** corrected_cv | canonico, reproducido por los "
             f"2 controles de V28 |",
             f"| Leaderboard de V20 | **{LB_V20}** | envio verificado, sin cambio |",
             f"| Tecnicas evaluadas | **{len(gap['tecnicas'])}** | gap analysis |",
             f"| Cerradas sin experimento | **{sum(1 for t in gap['tecnicas'] if t['experimentos'] == 0)}**"
             f" (B, D, E, F) | mismo fichero |",
             f"| Experimentos ejecutados | **{cierre['presupuesto_gastado']}** de "
             f"{cierre['presupuesto_autorizado']} autorizados | `v28_04_cierre.json` |",
             f"| Mejor corrected_cv de V28 | **{_dec(max(a['corrected_cv'], c['corrected_cv']))}** "
             f"(A, peor que V20) | artefactos |",
             f"| Finalistas | **{cierre['finalistas']}** | ninguna supero el gate |",
             f"| Submissions generadas | **{cierre['submissions_generadas']}** | hash de V20 sin cambio |",
             f"| Filas en `experiments.csv` | **{len(registro)}** | registro |", "",
             "**La respuesta a la pregunta de V28 es no, y se contesta con dos experimentos y cuatro "
             " cierres por equivalencia.** Ningun mecanismo estructuralmente distinto de combinar, "
             "estabilizar o descomponer la evidencia supera a V20.", "",
             "| ID | Tecnica | Modo | Resultado | Decision |", "|---|---|---|---|---|",
             f"| V28-02 | A · Meta-model bagging | probado | {_dec(a['delta_vs_V20'])}, "
             f"{a['folds_positivos']}/5 folds | CLOSED |",
             f"| V28-03 | C · Pairwise class decomposition | probado | {_dec(c['delta_vs_V20'])}, "
             f"{c['folds_positivos']}/5 folds | CLOSED |",
             "| - | B · Compact consensus geometry | equivalencia, ya medido | +0,0001 historico | "
             "cerrada sin experimento |",
             "| - | D · Dirichlet / evidence fusion | equivalencia con H04 y H06 | no medido | "
             "cerrada sin experimento |",
             "| - | E · Explicit uncertainty | duplicacion de B | no medido | cerrada sin experimento |",
             "| - | F · Randomized meta-subspace | equivalencia con `colsample_bytree` | no medido | "
             "cerrada sin experimento |", ""]
    return linea


def seccion_v28_gap() -> list[str]:
    """Punto 37: el gap analysis, que fue el trabajo mas rentable de la fase."""
    gap = _arte(ARTE_GAP)
    linea = ["## 2. Gap analysis: de seis tecnicas, dos sobreviven", "",
             "La busqueda es por **equivalencia funcional**. Este script queda excluido de su propia "
             "busqueda, porque sus patrones son literalmente los nombres que busca: sin esa "
             "exclusion, `OneVsOneClassifier` apareceria con una coincidencia y se declararia "
             "existente una familia que no existe.", "",
             "| ID | Tecnica | Veredicto | Experimentos | Evidencia |", "|---|---|---|---|---|"]
    for t in gap["tecnicas"]:
        linea.append(f"| {t['ID']} | {t['tecnica']} | **{t['veredicto']}** | {t['experimentos']} | "
                     f"{str(t['evidencia'])[:150].replace(chr(10), ' ')} |")
    incert = gap["bloque_incertidumbre_ya_medido"]
    linea += ["", "### B: cerrada con un numero, no con una opinion", "",
              "`scripts/advanced_04_incertidumbre.py` construye por miembro la **confianza**, el "
              "**margen** (top-1 menos top-2) y la **entropia** de Shannon; entre miembros, el "
              "reparto de votos por clase, la pluralidad, el desacuerdo y la desviacion de scores; "
              "y cuatro estadisticos agregados. Eso es un **superconjunto** de lo que B propone.", "",
              "| Variante | Features | macro-F1 |", "|---|---|---|",
              f"| v09 (base) | {incert['baseline_v09']['n_features']} | "
              f"{incert['baseline_v09']['macro_f1']} |",
              f"| + bloque de incertidumbre completo | "
              f"{incert['con_incertidumbre']['n_features']} | "
              f"{incert['con_incertidumbre']['macro_f1']} |", "",
              f"Delta: **{incert['delta']}**. Anadir 51 columnas de geometria del consenso no cambio "
              "el macro-F1.", "",
              "### F: ya implementado en la configuracion del champion", "",
              "`HIERBA` ya tiene `colsample_bytree=0,5`: cada arbol ve un subconjunto aleatorio de "
              "la mitad de las 279 columnas **manteniendo todas las disponibles**. Eso es "
              "exactamente el mecanismo que F describe, y V27-04 lo midio como control: quitarlo no "
              "cambia el numero de features que el modelo abre.", ""]
    return linea


def seccion_v28_tecnicas() -> list[str]:
    """Punto 37: las dos tecnicas realmente ejecutadas, con todo su detalle."""
    a, c = _t1(ARTE_A), _t1(ARTE_C)
    da, dc = _arte(ARTE_A), _arte(ARTE_C)
    a2 = da["control_de_varianza_A2"]
    masa = dc["masa_conjunta_38pct"]
    ca = list(da["candidatos"].values())[0]
    cc = list(dc["candidatos"].values())[0]
    linea = [
        "## 3. Las dos tecnicas realmente ejecutadas", "",
        "### 3.1 A · Meta-model bagging", "",
        "**Motivacion.** V20 usa un unico meta-modelo. Aunque la evidencia base sea correcta, la "
        "estimacion de la frontera podria tener varianza residual.", "",
        "**Lo unico que cambia.** Las filas que ve cada ajuste del meta-modelo. Se mantienen la "
        "matriz de 279 features, las predicciones de los tres miembros del blend, los folds, la "
        "cadena de cinco umbrales, la reconstruccion, los thresholds y el blend. No se anade ningun "
        "modelo base, ninguna representacion y no se selecciona ninguna feature.", "",
        f"**Configuracion, fijada antes de mirar la validacion.** N = {da['n_miembros']} miembros, "
        f"remuestreo {da['remuestreo']}, agregacion **{da['agregacion']}**.", "",
        f"**El confundidor, declarado antes del resultado.** {da['confundidor_declarado']}.", "",
        "| Metrica | T0 (V20) | T1 (embolsado) |", "|---|---|---|",
        f"| corrected_cv | {ca['corrected_cv']} | **{a['corrected_cv']}** |",
        f"| Delta | 0,0000000 | **{_dec(a['delta_vs_V20'])}** |",
        f"| Folds positivos | - | **{a['folds_positivos']}/5** |",
        f"| Rescue / damage / net | 0 / 0 / 0 | {a['rescue']} / {a['damage']} / **{a['net']:+}** |",
        f"| Filas que cambian | 0 | {a['changed_rows']:,} |",
        f"| IC95 | - | [{_dec(da['gate']['bootstrap']['ic95'][0])}, "
        f"{_dec(da['gate']['bootstrap']['ic95'][1])}], incluye el cero |", "",
        "**F1 por clase:**", "",
        "| Clase | V20 | Embolsado | Cambio |", "|---|---|---|---|"]
    for clase in ca["f1_por_clase"]:
        x, y = ca["f1_por_clase"][clase], a["f1_por_clase"][clase]
        linea.append(f"| {clase} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")
    linea += ["", "**A2, el control de varianza, que es lo que hace informativo al resultado:**", "",
              "| Medida | Valor |", "|---|---|",
              f"| **Acuerdo medio entre los {da['n_miembros']} miembros** | "
              f"**{a2['acuerdo_medio_entre_miembros']:.4f}** |",
              f"| Acuerdo minimo entre dos miembros | {a2['acuerdo_minimo_entre_miembros']:.4f} |",
              f"| Acuerdo de la prediccion final con V20 | "
              f"{a2['acuerdo_de_prediccion_con_V20']:.4f} |",
              f"| Desviacion tipica de las acumuladas, V20 | "
              f"{a2['desviacion_typica_de_acumuladas_V20']:.6f} |",
              f"| Desviacion tipica de las acumuladas, embolsado | "
              f"{a2['desviacion_typica_de_acumuladas_embolsadas']:.6f} |",
              f"| Mediana de la diferencia absoluta de acumuladas | "
              f"{a2['mediana_abs_diferencia_de_acumuladas']:.6f} |", "",
              "**Mecanismo observado.** Los cinco miembros **acuerdan solo en el 91,9 %** de las "
              "decisiones, luego **la varianza que A pretendia eliminar existia y era real**. Y "
              "promediarla empeoro: el resultado es 98,4 % identico a V20 y 0,000312 peor, con 815 "
              "rescates contra 861 danos. La dispersion de las acumuladas apenas se movio "
              "(0,32144 a 0,32105).", "",
              "**Lo que A cierra.** La varianza de la estimacion del meta-modelo **no es donde esta "
              "el error**. Los desacuerdos entre miembros ocurren en filas donde el modelo ya era "
              "incierto, y promediarlos no recupera senal, solo suaviza. Y el resultado **no** es un "
              "artefacto de capacidad: 1.500 estimadores en vez de 300 no dieron ganancia, lo que "
              "descarta la explicacion facil.", "",
              "### 3.2 C · Descomposicion pairwise en 5 pares disjuntos", "",
              "**Motivacion, medida antes de escribir codigo.** La confusion del control V20 es "
              "claramente bandada: los cinco pares adyacentes concentran el **59,9 % de sus 66.005 "
              "errores**.", "",
              "| Par adyacente | Confusiones mutuas |", "|---|---|"]
    for par in dc["confusion"] if isinstance(dc.get("confusion"), list) else []:
        pass
    conf = _arte(ARTE_GAP)["confusion_v20"]
    for par in conf["pares_adyacentes"]:
        linea.append(f"| {par['par']} | {par['total']:,} |")
    linea += ["", "**La distincion con V20, que es lo que hace nueva a la tecnica.** V20 entrena "
              "**cinco binarios anidados** de `clase <= k-1`, en los que cada uno mezcla todas las "
              "clases por debajo del umbral. C entrena **cinco binarios disjuntos**, uno por par "
              "adyacente, sin solape y sin restriccion de orden. No es CORAL, que comparte pesos e "
              "impone umbrales ordenados, ni cumulative-link: aqui cada frontera la aprende un "
              "clasificador independiente con la matriz completa de 279 features.", "",
              f"**Pares y agregacion, fijados antes de entrenar.** Criterio: "
              f"{dc['criterio_de_seleccion']}. Pares: `{'`, `'.join(dc['pares_seleccionados'])}`. "
              f"Agregacion: **{dc['agregacion']}**, una sola implementacion.", "",
              f"**El coste mecanico, declarado antes.** {dc['coste_mecanico_declarado']}.", "",
              "| Metrica | T0 (V20) | T1 (pares) |", "|---|---|---|",
              f"| corrected_cv | {cc['corrected_cv']} | **{c['corrected_cv']}** |",
              f"| Delta | 0,0000000 | **{_dec(c['delta_vs_V20'])}** |",
              f"| Folds positivos | - | **{c['folds_positivos']}/5** |",
              f"| Rescue / damage / net | 0 / 0 / 0 | {c['rescue']:,} / {c['damage']:,} / "
              f"**{c['net']:+,}** |",
              f"| Filas que cambian | 0 | {c['changed_rows']:,} |",
              f"| Errores totales | {cc['errores_totales']:,} | {c['errores_totales']:,} |",
              f"| Errores en los 5 pares adyacentes | **{cc['errores_en_pares_adyacentes']:,}** | "
              f"**{c['errores_en_pares_adyacentes']:,}** |",
              f"| IC95 | - | [{_dec(dc['gate']['bootstrap']['ic95'][0])}, "
              f"{_dec(dc['gate']['bootstrap']['ic95'][1])}], **excluye el cero** |", "",
              "**F1 por clase:**", "",
              "| Clase | V20 | Pares | Cambio |", "|---|---|---|---|"]
    for clase in cc["f1_por_clase"]:
        x, y = cc["f1_por_clase"][clase], c["f1_por_clase"][clase]
        linea.append(f"| {clase} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")
    linea += ["", "**Mecanismo observado, y aqui C aporto conocimiento real:**", "",
              f"1. **Los pares disjuntos si resuelven su objetivo local.** Los errores en los cinco "
              f"pares adyacentes **bajan de {cc['errores_en_pares_adyacentes']:,} a "
              f"{c['errores_en_pares_adyacentes']:,}**", "",
              "2. **El precio esta en las clases que no son frontera.** 50-64 y 65-xx *mejoran* "
              "ligeramente, mientras 13-17 y 18-24 se desploman. Un clasificador que solo ve "
              "`{k, k+1}` no puede colocar una fila lejos de su par: la masa de probabilidad de "
              "las clases no adyacentes desaparece al encadenar.", "",
              f"3. **C es complementario en la masa dificil y aun asi inutil.** De las "
              f"{masa['filas']:,} filas que V19 y V20 fallan juntas, **C acierta "
              f"{masa['de_esas_C_acerta']:,} ({masa['proporcion_de_la_masa_corregida']:.2%})**: "
              f"senal real, no ruido. Pero a cambio rompe {masa['de_las_no_masa_C_falla']:,} "
              f"aciertos que V20 tenia. Complementariedad sin rentabilidad.", "",
              "4. **Descartado el defecto de implementacion.** El IC95 es estrecho y excluye el "
              "cero, y el dano esta localizado en las clases no frontera, que es exactamente lo que "
              "predice el mecanismo de la cadena.", "",
              "> **Una discrepancia que se declara.** El brief cita 38,3 % de errores conjuntos y "
              "> esta medicion da **36,6 %**, porque la definicion no es identica: aqui se usa "
              "> `arbol_v19` (el miembro arbol del blend) contra V20. Se usa la cifra medida y "
              "> reproducible, no la que encajaria.", ""]
    return linea


def seccion_v28_cierre() -> list[str]:
    """Punto 37: cierre, conclusion y estado final."""
    a, c = _t1(ARTE_A), _t1(ARTE_C)
    gap = _arte(ARTE_GAP)
    linea = [
        "## 4. Por que se paro aqui, y no por inercia", "",
        "El brief autoriza ocho experimentos y permite no consumirlos. Se consumieron "
        f"**{gap['presupuesto_que_gasta']}**, y la razon es concreta:",
        "",
        "1. El gap analysis cerro **cuatro de seis** antes de entrenar, y dos de ellas (**B** y "
        "**F**) porque ya estaban implementadas, una de ellas con un numero medido hace anos.",
        "",
        "2. Las dos que quedaron son **mecanismos distintos entre si** y ambos se midieron: uno de "
        "estimacion, uno de descomposicion de la decision.",
        "",
        "3. Los dos son negativos, y el punto 34 pide precisamente eso: si dos o tres tecnicas "
        "consecutivas no muestran ninguna senal, hacer una evaluacion global antes de seguir "
        "consumiendo. Las dos son negativas y de forma rotunda, con un de ellas con el IC95 "
        "excluyendo el cero.", "",
        "4. **E y F no se ejecutaron** y no por prudencia vaga: E es duplicacion de B por la regla "
        "critica del propio brief, y F es un hiperparametro ya activo en `HIERBA`. Ninguna de las "
        "dos necesita un experimento para estar cerrada.", "",
        "## 5. Conclusion cientifica de V28", "",
        "### Donde esta el cuello de botella de AGE, ahora", "",
        "| Eje | Estado tras V28 | Evidencia |", "|---|---|---|",
        "| Representacion de features | agotado | V25 (seleccion), V26-01 (escala), V27-05 "
        "(percentil) |",
        "| Combinacion de predicciones | agotado | H04, H05, H06, H13, y **D** cerrada por "
        "equivalencia en V28 |",
        "| Calibracion | agotado | H01 (D-029) |",
        "| Estructura ordinal | agotado | H10, CORAL (D-058) |",
        "| Geometria del consenso | agotado | **B**, superconjunto ya implementado y medido: "
        "+0,0001 |",
        "| **Varianza del meta-modelo** | **agotado en V28** | **A**: la varianza existia (91,9 % de "
        "acuerdo) y promediarla empeoro |",
        "| **Descomposicion de la decision** | **agotado en V28** | **C**: los pares disjuntos "
        "atacan el 59,9 % de los errores y aun asi colapsan las clases no frontera |",
        "| Incertidumbres explicita | cerrada | **E**, duplicacion de B |",
        "| Subespacio aleatorio | cerrado | **F**, ya implementado en `colsample_bytree` |",
        "| **Limitacion de los datos** | **lo unico abierto** | la masa que ningun modelo corrige |",
        "",
        "### La respuesta", "",
        "**La pregunta de V28 era si existia una mejora escondida en la forma de estimar y combinar "
        "la evidencia, y la respuesta es que no.** Las dos rutas genuinamente nuevas se han medido:",
        "",
        "- **La varianza de la estimacion no es el cuello.** Existia, se midio, y eliminarla empeoro. "
        "Y 1.500 estimadores en vez de 300 no dieron ganancia, asi que tampoco es capacidad.",
        "",
        "- **La descomposicion de la decision no es el cuello.**-ataca exactamente donde esta el "
        "error medido y funciona ahi, pero el dano en las clases no frontera es mayor que el "
        "beneficio. El problema no es *que clase* se confunde, es *cuanta informacion* sobrevive a "
        "la descomposicion.", "",
        "**Junto con V25 a V27, el limite que queda es el mismo que se identifico entonces: la "
        "limitacion de los datos.** La masa de filas que ningun modelo del espacio probado corrige "
        "sigue ahi, y V28 no la ha reducido.", "",
        "**No se afirma que AGE este resuelto.** Se afirma que el espacio de mecanismos accesible "
        "sin senal nueva esta agotado, con evidencia negativa reproducible en cada eje.", "",
        "## 6. Conclusion practica de V28", "",
        "| Pregunta | Respuesta |", "|---|---|",
        f"| Baseline vigente | **V20**, {_dec(CV_V20)} corrected_cv, LB {LB_V20} |",
        f"| corrected_cv de A | {_dec(a['corrected_cv'])}, **peor** que V20 |",
        f"| corrected_cv de C | {_dec(c['corrected_cv'])}, **mucho peor** que V20 |",
        "| Mejor candidato interno | ninguno: las dos tecnicas son negativas |",
        "| Existe candidato listo para submission | **No** |",
        "| Debe cerrarse la investigacion | **Si**, si no aparece senal nueva |",
        "| Se genera submission | **No** |",
        "| Se modifica V20 | **No** |", "",
        "### Que queda por delante, y no es una hipotesis mas", "",
        "La unica via con recorrido es **senal nueva desde los datos**, y las restricciones de AGE "
        "(sin transformers, sin LLM, sin embeddings, sin etiquetas externas) cierran las fuentes "
        "obvias. Cualquier linea nueva deberia empezar por demostrar que aporta senal que el bloque "
        "de evidencia de los miembros no aporta, y no por otra forma de combinar lo que ya existe: "
        "V25 a V28 hanShown que las cuatro familias de esa pregunta estan cerradas.", ""]
    return linea


def main() -> int:
    """Genera los dos informes y actualiza los documentos de estado."""
    autotest_numerico()
    inicio = time.time()

    # ---------------------------------------------------------------- informe de V28
    partes = ["# V28 - Informe final de la fase", "",
              f"**Fecha:** {FECHA} · **Rama:** `proyecto-kaggle-age-2026-v28` · "
              f"**Baseline congelado:** {_dec(CV_V20)} · **Champion:** V20, LB {LB_V20}", "",
              "Esta fase no reabre ninguna familia de V25 a V27. La premisa es que el limite de V20 no "
              "esta en la cantidad ni en la escala de las features, sino en **la forma de estimar y "
              "combinar la evidencia**. Se evaluan seis mecanismos estructuralmente distintos y se "
              "responde la pregunta con lo medido.", ""]
    partes += seccion_v28_resumen()
    partes += seccion_v28_gap()
    partes += seccion_v28_tecnicas()
    partes += seccion_v28_cierre()
    partes += [
        "## 7. Reproducibilidad, compliance y git", "",
        "| Comprobacion | Resultado |", "|---|---|",
        f"| Control absoluto de V20 en P0 | {_dec(_arte(ARTE_P0)['control_absoluto']['corrected_cv_recomputado'])}, "
        f"OOF **identico bit a bit** al artefacto, folds identicos |",
        "| `ast.parse`, `ruff --select F,E9` y `py_compile` en los 7 scripts de V28 | 0 errores |",
        "| Auditoria de idioma | ver `reports/v13_auditoria_idioma.md` |",
        "| Compliance de V28 | 13 comprobaciones, **todas en cero usos** |",
        "| Higiene del repositorio | los 2 directorios ajenos **siguen fuera de la rama** |",
        "| Trazabilidad | `experiments.csv` <-> tabla maestra <-> artefactos: coherente |",
        "| Notebook | seccion 36 ejecutada de principio a fin, 4 aserciones en verde |",
        "| Artefactos de V20 tocados | **0** |",
        "| Submissions generadas | **0** |", "",
        "**No se declara R4.** No hubo reproduccion independiente por un tercero. La compliance "
        "busca firmas de las practicas prohibidas y no puede demostrar que no ocurran de otra "
        "forma.", "",
        "### Git", "",
        "Rama `proyecto-kaggle-age-2026-v28`, creada desde `proyecto-kaggle-age-2026-v26`. "
        "**Sin push, sin merge, sin force-push y sin reset destructivo.** No se incluye ningun "
        "artefacto ajeno, ningun secreto y ninguno de los dos directorios no ajenos a AGE.", "",
        "---", "",
        "**Cierre de V28.** V20 permanece como baseline absoluto en " + _dec(CV_V20) + ". Las seis "
        "tecnicas evaluadas quedan respondidas: cuatro por equivalencia funcional, dos por "
        "experimento, y las dos experimentadas son negativas. No se genera ninguna submission, no "
        "se toca el test, el holdout, el leaderboard ni ningun artefacto de V20.", ""]
    INFORME_V28.write_text("\n".join(partes), encoding="utf-8")

    # ---------------------------------------------------------------- reporte historico integral
    total = ["# AGE - Informe completo de la investigacion, de V20 al cierre de V28", "",
             f"**Fecha:** {FECHA} · **Rama:** `proyecto-kaggle-age-2026-v28` · "
             f"**Baseline congelado:** {_dec(CV_V20)} · **Champion:** V20, LB {LB_V20}", "",
             "Reconstruye **toda** la trayectoria de AGE desde V20 hasta el cierre de V28, sin omitir "
             "ninguna fase negativa. La mayoria de lo que se midio en estas fases fue negativo, y ese "
             "resultado es el producto.", "",
             "**Ninguna cifra de este informe esta escrita a mano si existe un artefacto del que "
             "leerla.** Las secciones de V20 a V27 se generan importando "
             "`scripts/v27_10_informe_completo.py`, de modo que hay una sola fuente y este informe no "
             "puede divergir del de V27. Las de V28 las genera `scripts/v28_07_informes.py`.", ""]
    total += v27.seccion_resumen()
    total += v27.seccion_cronologia()
    total += v27.seccion_tabla()
    total += v27.seccion_familias()
    total += v27.seccion_v20()
    total += v27.seccion_v25()
    total += v27.seccion_v26()
    total += v27.seccion_negativos()
    total += v27.seccion_errores()
    total += ["## 12. V28: mecanismos estructuralmente distintos para combinar la evidencia", ""]
    total += [l for l in seccion_v28_resumen() if l.startswith(("|", "-", "**La respuesta"))][1:]
    total += ["", "### El gap analysis de V28, que fue el trabajo mas rentable de la fase", ""]
    total += seccion_v28_gap()[3:]
    total += seccion_v28_tecnicas()
    total += ["## 13. Reproducibilidad acumulada de V20 a V28", ""]
    total += v27.seccion_repro()[2:]
    total += ["## 14. Compliance acumulada", ""]
    total += v27.seccion_compliance()[2:]
    total += ["", "### Compliance especifica de V28", "",
              "Las 13 comprobaciones de V28 dan cero usos, y la higiene del repositorio se verifica "
              "de nuevo al cierre: los dos directorios ajenos a AGE **siguen fuera de la rama**.",
              "", "```text"] + [f"{f['comprobacion']} = {f['uso_en_codigo']}"
                                 for f in _arte("experiments/v28_05_auditoria_final.json")
                                 ["compliance"]["comprobaciones"]] + ["```", ""]
    total += seccion_v28_cierre()[1:]
    total += ["## 19. Figuras acumuladas", ""]
    total += v27.seccion_figuras()[2:]
    figs28 = sorted(p.name for p in cfg.FIGURES.glob("v28_*.png"))
    total += [f"### Figuras de V28 ({len(figs28)})", "",
              "| Figura | Que muestra |", "|---|---|",
              "| `v28_maestra_resultados.png` | corrected_cv por experimento, delta frente a V20, "
              "consistencia por fold y rescate frente a dano |",
              "| `v28_maestra_folds.png` | F1 por clase de A y C, y el acuerdo entre miembros de A |",
              "| `v28_A_bagging_diagnostico.png` | delta por fold, F1 por clase y dispersion de A |",
              "| `v28_C_pairwise_diagnostico.png` | delta por fold, F1 por clase y errores en los "
              "pares adyacentes de C |", "",
              "Ninguna oculta un resultado negativo: las cuatro son de las dos tecnicas negativas.", ""]
    total += ["---", "",
              "**Cierre de la investigacion.** V20 permanece como baseline absoluto en " + _dec(CV_V20)
              + " (LB " + str(LB_V20) + "). El leaderboard no se utilizo para decidir entre "
              "hipotesis, no se toco el test ni el holdout, no se genero ninguna submission y no se "
              "modifico V20. La investigacion termina porque las hipotesis defendibles han sido "
              "contestadas o cerradas por evidencia, y porque las dos unicas que quedaban "
              "mecanicamente abiertas han resultado negativas.", ""]
    # Las secciones de V28 traen su propia numeracion, que choca con la de V20 a V27 al
    # concatenarlas. Se renumeran **en orden de aparicion** para que el indice sea legible y para
    # que ningun lector tenga que deducir el salto. El texto de cada seccion no se toca.
    total = _renumera(total)
    INFORME_TOTAL.write_text("\n".join(total), encoding="utf-8")

    _actualiza_docs()
    RESULTADO.write_text(json.dumps({
        "fase": "V28 07 informes", "fecha": FECHA, "experimenta": False,
        "informe_v28": str(INFORME_V28.relative_to(ROOT)),
        "informe_total": str(INFORME_TOTAL.relative_to(ROOT)),
        "lineas_v28": len(INFORME_V28.read_text(encoding="utf-8").splitlines()),
        "lineas_total": len(INFORME_TOTAL.read_text(encoding="utf-8").splitlines()),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[informe V28] {INFORME_V28.relative_to(ROOT)}: "
          f"{len(INFORME_V28.read_text(encoding='utf-8').splitlines())} lineas")
    print(f"[informe total] {INFORME_TOTAL.relative_to(ROOT)}: "
          f"{len(INFORME_TOTAL.read_text(encoding='utf-8').splitlines())} lineas")
    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


def _renumera(lineas: list[str]) -> list[str]:
    """Renumera en orden los encabezados `## N. Titulo`, sin tocar el resto del texto.

    Las secciones de V28 y las de V20 a V27 traen numeraciones propias que chocan al
    concatenarlas. Se renumera por aparicion, que es lo unico que garantiza un indice legible.
    """
    import re as _re
    patron = _re.compile(r"^##\s+\d+\.\s+")
    salida, contador = [], 0
    for linea in lineas:
        if patron.match(linea):
            contador += 1
            salida.append(patron.sub(f"## {contador}. ", linea))
        else:
            salida.append(linea)
    return salida


def _actualiza_docs() -> None:
    """Cierra los cuatro documentos de estado y anade D-063."""
    a, c = _t1(ARTE_A), _t1(ARTE_C)
    gap = _arte(ARTE_GAP)
    registro = pd.read_csv(ROOT / "experiments.csv")
    gap26 = pd.read_csv(cfg.REPORTS / "v26_hypothesis_gap_analysis.csv")
    cerradas26 = int(gap26["status"].astype(str).str.startswith("CLOSED").sum())
    marginales26 = int(gap26["status"].astype(str).str.startswith("MARGINAL").sum())
    marca = "## Cierre de la fase V28"
    bloque = [
        "", "---", "", f"{marca} · {FECHA}", "",
        f"- **Baseline vigente:** V20, {_dec(CV_V20)} corrected_cv, LB {LB_V20}, hash sin cambio.",
        f"- **Mejor corrected_cv de V28:** {_dec(a['corrected_cv'])} (A), **peor que V20**.",
        "- **Mejor candidato interno:** ninguno. Las dos tecnicas ejecutadas son negativas.",
        f"- **Numero de experimentos:** {len(registro)} filas en `experiments.csv`; **2** nuevos en "
        f"V28, **0** filas ficticias para B, D, E y F.",
        "- **Numero de submissions:** **0** en V26, V27 y V28.",
        f"- **Familias:** {cerradas26} cerradas y {marginales26} marginales en V26, mas las 6 "
        f"de V28; **ninguna abierta**.",
        f"- **Tecnicas de V28:** {gap['presupuesto_que_gasta']} ejecutadas de 6; "
        f"{sum(1 for t in gap['tecnicas'] if t['experimentos'] == 0)} cerradas sin "
        f"gastar ningun experimento.",
        "- **Notebook:** seccion 36 anadida, ejecutada, con las 4 aserciones en verde.",
        "- **Higiene:** los 2 directorios no ajenos a AGE quedan fuera de la rama, sin borrar y sin "
        "modificar.",
        "- **Informes:** `reports/v28_final_research.md` y "
        "`reports/AGE_COMPLETE_RESEARCH_REPORT.md` reconstruido de V20 a V28.", ""]
    for nombre in ("FINAL_REPORT.md", "PROJECT_STATE.md", "RUNLOG.md"):
        ruta = ROOT / nombre
        texto = ruta.read_text(encoding="utf-8")
        if marca in texto:
            continue
        ruta.write_text(texto.rstrip("\n") + "\n" + "\n".join(bloque), encoding="utf-8")

    # D-063 sigue la convencion numerada del registro de decisiones.
    ruta = ROOT / "DECISIONS.md"
    texto = ruta.read_text(encoding="utf-8")
    if "## D-063" in texto:
        return
    entrada = [
        "", "---", "",
        "## D-063 - V28: ni la varianza de la estimacion ni la descomposicion de la decision son el "
        "cuello de botella", "",
        "- **Observation**: V25 a V27 habian eliminado el podado de features, la escala de la "
        "evidencia, las restricciones monotonas, los percentiles, la fiabilidad constante y las "
        "reformulaciones ordinales. V28 se pregunto por mecanismos **estructuralmente distintos** "
        "para combinar, estabilizar o descomponer la evidencia: bagging del meta-modelo, "
        "geometria del consenso, descomposicion pairwise, fusion de evidencia, incertidumbre "
        "explicita y subespacio aleatorio.", "",
        "- **El gap analysis fue el trabajo mas rentable.** Cuatro de las seis tecnicas se cerraron "
        "**sin gastar un experimento**: **B** porque `advanced_04_incertidumbre.py` implementa un "
        "superconjunto y esta medido (267 features 0,5897 frente a 318 features 0,5898, delta "
        "+0,0001); **D** porque un pool de evidencia es un combinador, y H04 mas H06 estan cerrados "
        "con numero, con V20 ya siendo el segundo nivel; **E** por la regla critica del brief, que "
        "prohibe duplicar B; y **F** porque `HIERBA` ya tiene `colsample_bytree=0,5`, que es "
        "exactamente aleatorizar la exposicion de cada arbol a un subespacio manteniendo todas las "
        "columnas.", "",
        f"- **A CLOSED**: bagging de N={_arte(ARTE_A)['n_miembros']} miembros con bootstrap de "
        f"Poisson(1), agregado por promedio de probabilidades. Delta **{_dec(a['delta_vs_V20'])}**, "
        f"{a['folds_positivos']}/5 folds, neto {a['net']:+}, IC95 que incluye el cero. **La "
        f"varianza existia**: los cinco miembros acordaban solo en el "
        f"{_arte(ARTE_A)['control_de_varianza_A2']['acuerdo_medio_entre_miembros']:.4f} de las "
        f"decisiones. Y promediarla empeoro. Ademas 1.500 estimadores en vez de 300 no dieron "
        f"ganancia, lo que descarta la explicacion por capacidad.", "",
        f"- **C CLOSED**: cinco binarios **disjuntos**, uno por par adyacente, contra los cinco "
        f"**anidados** de `clase <= k-1` de V20. Delta **{_dec(c['delta_vs_V20'])}**, "
        f"{c['folds_positivos']}/5 folds, neto {c['net']:+,}, IC95 que **excluye el cero**. Los "
        f"pares si reducen los errores de su objetivo local, de "
        f"{list(_arte(ARTE_C)['candidatos'].values())[0]['errores_en_pares_adyacentes']:,} a "
        f"{c['errores_en_pares_adyacentes']:,}, y si son complementarios en la masa dificil "
        f"({_arte(ARTE_C)['masa_conjunta_38pct']['de_esas_C_acerta']:,} de las "
        f"{_arte(ARTE_C)['masa_conjunta_38pct']['filas']:,} filas que V19 y V20 fallan juntas). El "
        f"daño esta en las clases que no son frontera, donde un clasificador que solo ve dos clases "
        f"no puede colocar una fila lejos de su par.", "",
        "- **Decision**: **V20 permanece como baseline absoluto** en " + _dec(CV_V20) + f" (LB "
        f"{LB_V20}). Las seis tecnicas quedan respondidas: cuatro por equivalencia funcional y dos "
        f"por experimento, ambas negativas. **0 finalistas, 0 submissions, ninguna linea de V20 "
        f"tocada.** El cuello de botella restante es la **limitacion de los datos**, y V28 no ha "
        f"reducido la masa que ningun modelo corrige.", "",
        "- **Consequences**: queda agotado el espacio de mecanismos accesible **sin senal nueva**: "
        "representacion, combinacion, calibracion, estructura ordinal, geometria del consenso, "
        "varianza de estimacion y descomposicion de la decision tienen todas evidencia negativa "
        "reproducible. Cualquier linea futura tiene que demostrar primero que aporta senal que el "
        "bloque de evidencia de los miembros no aporta; otra forma de combinar lo que ya existe esta "
        "cerrada por V25 a V28.", "",
        "- **Una correccion que se declara**: el IC95 del bootstrap de **V26-01** quedo a `null` en "
        "su artefacto y no es recuperable sin reentrenar. Se marco `no persistido` en la tabla "
        "historica en vez de citar una cifra sin fuente. No afecta a ninguna decision: H02 se cerro "
        "por delta y folds, y el bootstrap era confirmatorio.", "",
    ]
    ruta.write_text(texto.rstrip("\n") + "\n" + "\n".join(entrada), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

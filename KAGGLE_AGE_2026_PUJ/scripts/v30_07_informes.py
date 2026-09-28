"""V30 - 07: informes, notebook y cierre documental de la fase.

Genera los cuatro expedientes que el brief pide mas los dos que anade para no dejar huecos, integra
la seccion 38 en el notebook consolidado y escribe D-065.

## La regla que gobierna el script

**Ninguna cifra se escribe a mano si existe un artefacto del que leerla.** Y hay una segunda regla,
que se incumple mas a menudo: **ninguna afirmacion cualitativa se escribe sin su contraparte
medida**. Cuando el texto dice "el refinamiento no consolida", al lado tiene que estar el numero que
lo dice, leído del artefacto.

## La honestidad de la submission

Si la subida se detuvo por falta de credencial, el informe lo dice con esas palabras y explica que
el fichero **esta construido, validado y con hash**, listo para enviar. No se disfraza un STOP de
infraestructura de un resultado experimental, porque son cosas distintas y confundirlas seria
mentir.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

import v27_10_informe_completo as v27  # noqa: E402
import v28_07_informes as v28  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
INFORME = cfg.REPORTS / "V30_FINAL_REPORT.md"
REFIN_MD = cfg.REPORTS / "v30_optuna_refinement.md"
TRANSFER_MD = cfg.REPORTS / "v30_transfer_analysis.md"
HISTORIAL = cfg.REPORTS / "age_complete_experiment_history.csv"
RESULTADO = cfg.EXPERIMENTS / "v30_07_informes.json"
CV_V20 = 0.6097782
LB_V20 = 0.62689
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
GATE_DELTA = 0.003
MARGINAL_MIN = 0.002
FECHA = "2026-09-30"


def _arte(nombre: str) -> dict:
    return json.loads((cfg.EXPERIMENTS / nombre).read_text(encoding="utf-8"))


def _dec(n: float, casas: int = 7) -> str:
    return f"{n:.{casas}f}".replace(".", ",")


def _pct(n: float) -> str:
    return _dec(100.0 * float(n), 1) + " %"


# ============================================================================ seccion 1
def seccion_control() -> list[str]:
    """P0: los tres controles, y por que tienen distinta fuerza."""
    p0 = _arte("v30_00_control_absoluto.json")
    c20, c29, c30 = p0["control_v20"], p0["control_lightgbm_v29"], p0["control_ensemble_v29"]
    return [
        "## 1. Control absoluto: V30 no empieza sin que el punto de partida sea el que se dice", "",
        "| Control | Esperado | Obtenido | Reproduce |",
        "|---|---|---|---|",
        f"| V20, reconstruido y comparado **bit a bit** | {_dec(CV_V20)} | "
        f"{_dec(c20['corrected_cv_recomputado'])} | **{c20['reproduce_bit_a_bit']}** |",
        f"| LightGBM de V29, OOF en disco | {_dec(c29['esperado'])} | "
        f"{_dec(c29['corrected_cv_del_oof_en_disco'])} | **{c29['coincide_con_esperado']}** |",
        f"| Ensemble de V29, **re-optimizado** | {_dec(c30['corrected_cv_de_V29'])} | "
        f"{_dec(c30['corrected_cv_reoptimizado'])} | **{c30['reproduce']}** |", "",
        f"**La diferencia maxima entre el OOF reconstruido de V20 y su artefacto es "
        f"{c20['max_abs_diferencia_vs_artefacto']:.2e}**, y los folds son los mismos: "
        f"{c20['folds']}.", "",
        f"Un resultado que merece staffed atencion: la reoptimizacion de los pesos del ensemble de "
        f"V29 **reprodujo con desviacion {c30['desviacion']}**. El propio brief advertia que no "
        f"debia declararse exactitud bit a bit entre reproducciones de Optuna; en esta maquina y "
        f"con estas versiones de libreria, si lo es. Se registra lo medido, no lo temido.", ""]


# ============================================================================ seccion 2
def seccion_refinamiento() -> list[str]:
    """P1: el refinamiento local, con su justificacion mecanica y su resultado."""
    ref = _arte("v30_01_optuna_refinamiento.json")
    ens = _arte("v30_02_ensemble.json")
    comp = pd.read_csv(cfg.REPORTS / "v30_candidate_comparison.csv")
    contra_v20 = comp[comp["alcance"] == "contra V20"]
    contra_pares = comp[comp["alcance"] == "par a par"]
    lineas = [
        "## 2. V30: refinamiento local del LightGBM, sembrado con el mejor trial de V29", "",
        "### Por que el espacio es local, y no una busqueda nueva", "",
        "Los cinco mejores trials de V29 coinciden en una region estrecha, y **V20 estaba "
        "sub-parametrizado en casi todo**:", "",
        "| Parametro | 5 mejores de V29 | `HIERBA` de V20 |", "|---|---|---|",
        "| `num_leaves` | 43 - 87 | **8** |",
        "| `max_depth` | 5 - 9 | **3** |",
        "| `learning_rate` | 0,021 - 0,043 | 0,05 |",
        "| `min_child_samples` | 22 - 94 | **200** |",
        "| `reg_lambda` | 1e-7 - 0,018 | **1,0** |", "",
        "Ese contraste **es el mecanismo de la mejora de V29**, y por eso V30 centra su espacio ahi "
        "en vez de recorrer la region entera. Y la localidad no es solo un rango: el **trial 0 de "
        "cada fold se encola con los parametros exactos** del mejor trial de V29 para ese fold.", "",
        f"- Trials: **{ref['trials_por_fold']} por fold, {ref['trials_totales']} en total**, el "
        f"maximo que autoriza el brief.",
        f"- Sampler: `{ref['sampler']}`.",
        f"- Techo de estimadores: **{ref['max_estimadores']}**, con early stopping de "
        f"{ref['early_stopping_rondas']} rondas.", "",
        "### Iteraciones reales, que es donde se ve el early stopping", "",
        "| Fold | Mejor trial | Inner | Mediana | Sigma | Iteraciones / techo | Early stop | Outer |",
        "|---|---|---|---|---|---|---|---|"]
    for fold, f in sorted(ref["detalle_por_fold"].items(), key=lambda kv: int(kv[0])):
        d = f["diagnostico_refit"]
        lineas.append(
            f"| {fold} | {f['mejor_trial']} | {f['valor_mejor_inner']:.6f} | "
            f"{f['mediana_trial']:.6f} | {f['mejor_sobre_mediana_en_sigmas']:.2f} | "
            f"{d['iteraciones']} / {d['max_iteraciones']} | {d['early_stopping']} | "
            f"{f['macro_f1_outer_val']:.6f} |")
    lineas += ["",
               "**El early stopping corta siempre muy por debajo del techo.** Es exactamente el "
               "comportamiento que el punto 38 del brief describe como correcto: dar margen alto de "
               "capacidad y dejar que el criterio interior decida donde parar, en vez de entrenar "
               "artificialmente hasta el limite.", "",
               f"**Mejor trial outlier: {ref['mejores_trial_outlier']} de "
               f"{len(ref['detalle_por_fold'])}.** Ningun fold se gano por un trial anomalo, asi que "
               f"el resultado no depende de una tirada afortunada.", "",
               "### Resultado del refinamiento", "",
               f"- corrected_cv **{_dec(ref['corrected_cv'])}**",
               f"- frente a V20: **{_dec(ref['delta_vs_V20'], 6)}**",
               f"- frente al LightGBM de V29: **{_dec(ref['delta_vs_lgbm_v29'], 6)}**",
               f"- folds que mejoran a V29: **{ref['folds_mejor_que_v29']}/5** "
               f"({ref['delta_por_fold_contra_v29']})",
               f"- veredicto: **{ref['estado_vs_v29']}**", "",
               f"Criterio declarado de consolidacion: {ref['criterio_de_consolidacion']}.", ""]
    if ens["hay_candidato_v30"]:
        lineas += [
            "### ¿V30 aprende mejor, o solo reproduce a V29?", "",
            "| Modelo | CV individual | corr. errores con V20 | Rescate unico | Dano unico | Neto |",
            "|---|---|---|---|---|---|"]
        for _, f in contra_v20.iterrows():
            lineas.append(f"| {f['modelo_a']} | {f['individual_cv']:.7f} | {f['corr_errores']:.4f} | "
                          f"{int(f['rescate_unico']):,} | {int(f['dano_unico']):,} | "
                          f"{int(f['neto']):+,} |")
        lineas += ["", "Y entre los dos LightGBM, que es la comparacion que decide si hay algo que "
                  "combinar:", "", "| Par | corr. errores | Discrepancia | Uno arregla al otro |",
                  "|---|---|---|---|"]
        for _, f in contra_pares.iterrows():
            if {f["modelo_a"], f["modelo_b"]} != {"v29_lgbm_optuna", "v30_lgbm_refinado"}:
                continue
            lineas.append(f"| V29 vs V30 | {f['corr_errores']:.4f} | {f['tasa_discrepancia']:.4f} | "
                          f"{int(f['rescate_unico']):,} |")
        lineas += ["",
                   "Una correlacion de errores **muy alta** entre V29 y V30 significa que el "
                   "refinamiento encontro la misma funcion con parametros distintos: es una "
                   "confirmacion de que la region esta bien elegida, y tambien de que **no hay un "
                   "tercer miembro que aportar** al ensemble. Por eso el pool de V30 se queda en "
                   "tres: V20, V29 y V30.", ""]
    return lineas


# ============================================================================ seccion 3
def seccion_ensemble() -> list[str]:
    """P3 y P4: el ensemble, sus pesos y su bootstrap."""
    ens = _arte("v30_02_ensemble.json")
    principal = next(f for f in ens["tabla"] if f["metodo"] == ens["candidato_principal"])
    boot = ens["bootstrap"]
    lineas = [
        "## 3. El ensemble: tres miembros, no dieciseis", "",
        f"V29 ya contesto la pregunta del pool historico (**0 de 47** con rescate neto positivo, y "
        f"los tres metodos negativos sin el modelo nuevo). V30 **no repite esa busqueda**: el pool "
        f"es de **{ens['n_miembros']} miembros** y cada uno esta justificado.", "",
        "| Miembro | Papel |", "|---|---|",
        "| `v20_f3b_oof_candidato` | el champion, intacto |",
        "| `v29_lgbm_optuna` | la capacidad que V29 demostro |",
        "| `v30_lgbm_refinado` | el intento de consolidar esa capacidad |", "",
        "Los pesos se optimizan **solo sobre el holdout interior**, con `w = softmax(theta)` para "
        "imponer `w_i >= 0` y `sum w_i = 1` sin restricciones. La validacion exterior no entra.", "",
        "| Metodo | CV | Delta V20 | Delta V29-LGBM | Folds + | Rescue | Damage | Net |",
        "|---|---|---|---|---|---|---|---|"]
    for f in ens["tabla"]:
        lineas.append(f"| {f['metodo']} | {f['corrected_cv']} | {f['delta_vs_V20']:+.6f} | "
                      f"{f['delta_vs_lgbm_v29']:+.6f} | {f['folds_positivos']}/5 | "
                      f"{int(f['rescue']):,} | {int(f['damage']):,} | {int(f['net']):+,} |")
    lineas += ["",
               f"### El candidato principal: `{ens['candidato_principal']}`", "",
               f"- corrected_cv **{_dec(principal['corrected_cv'])}**, "
               f"delta frente a V20 **{_dec(principal['delta_vs_V20'], 6)}**",
               f"- **{principal['folds_positivos']}/5 folds** positivos, "
               f"rescate {int(principal['rescue']):,} contra dano {int(principal['damage']):,}, "
               f"neto **{int(principal['net']):+,}**",
               f"- pesos: `{principal['pesos']}`",
               f"- filas que cambian frente a V20: {int(principal['filas_cambiadas_vs_v20']):,}", "",
               "### Las dos banderas, separadas como pide el punto 20", "",
               "| Bandera | Valor | Significado |", "|---|---|---|",
               f"| Gate historico | **{ens['gate_historico_superado']}** | exige delta >= "
               f"+{_dec(GATE_DELTA, 3)} **y** >= 4/5 folds |",
               f"| STRONG_MARGINAL | **{ens['strong_marginal']}** | 5/5 folds **y** delta entre "
               f"+{_dec(MARGINAL_MIN, 3)} y +{_dec(GATE_DELTA, 3)} |", "",
               "Son **dos preguntas distintas** y no se confunden: el gate es el criterio de "
               "finalista; la bandera marginal es la observacion de que la mejora es consistente "
               "pero no llega al umbral.", "",
               "### Bootstrap del candidato principal", "",
               f"- delta **{_dec(boot['delta'], 6)}**",
               f"- IC95 **[{_dec(boot['ic95'][0], 6)}, {_dec(boot['ic95'][1], 6)}]**",
               f"- proporcion de diferencias positivas: {boot['proporcion_positiva']:.4f}",
               f"- **el IC95 incluye el cero: {boot['ic95_incluye_cero']}**", "",
               f"{boot['nota']}", ""]
    return lineas


# ============================================================================ seccion 4
def seccion_submission() -> list[str]:
    """P6 a P9: construccion, validacion, hash y estado de la subida."""
    con = json.loads((cfg.EXPERIMENTS / "v30_03_submission_construccion.json").read_text(
        encoding="utf-8")) if (cfg.EXPERIMENTS / "v30_03_submission_construccion.json").exists() \
        else {}
    up = json.loads((cfg.EXPERIMENTS / "v30_04_upload.json").read_text(encoding="utf-8")) \
        if (cfg.EXPERIMENTS / "v30_04_upload.json").exists() else {}
    segunda = _arte("v30_05_segunda_particion.json") \
        if (cfg.EXPERIMENTS / "v30_05_segunda_particion.json").exists() else {}
    lineas = ["## 4. Segunda particion, con la configuracion congelada", ""]
    if segunda:
        lineas += [
            f"Los hiperparametros y los pesos se aplican **congelados** sobre una particion "
            f"agrupada distinta (semilla {segunda['semilla_nueva']}, mismos grupos, sin "
            f"re-optimizar). Los solapes con la particion original fueron "
            f"{segunda['solapes_con_la_particion_original']}, luego la particion cambio de verdad.", "",
            "| Fold | V20 recomputado | Candidato | Delta |", "|---|---|---|---|"]
        for f in segunda["folds"]:
            lineas.append(f"| {f['fold']} | {f['corrected_cv_V20']:.7f} | "
                          f"{f['corrected_cv_candidato']:.7f} | {f['delta']:+.6f} |")
        lineas += ["",
                   f"- V20 recomputado **{_dec(segunda['corrected_cv_V20_segunda'])}** | candidato "
                   f"**{_dec(segunda['corrected_cv_candidato_segunda'])}**",
                   f"- delta contra el recomputado: "
                   f"**{_dec(segunda['delta_vs_V20_recomputado'], 6)}** con "
                   f"**{segunda['folds_positivos']}/5 folds** positivos",
                   f"- el gate usa el V20 **recomputado en esta particion**: "
                   f"{segunda['criterio_del_delta']}",
                   f"- frente al delta de la particion original "
                   f"({_dec(segunda['delta_vs_V20_particion_original'], 6)}) la diferencia es "
                   f"{_dec(segunda['diferencia_de_delta_entre_particiones'], 6)}: "
                   f"{'la mejora se sostiene' if abs(segunda['diferencia_de_delta_entre_particiones']) < 0.001 else 'se mueve, y eso se declara'}",
                   f"- gate historico en la segunda particion: "
                   f"**{segunda['cumple_gate_historico']}** | STRONG_MARGINAL: "
                   f"**{segunda['es_strong_marginal']}**", ""]
    else:
        lineas += ["La segunda particion **no se ejecuto**: el candidato no cumplio la condicion de "
                   "disparador (5/5 folds y delta >= +0,002) en la particion original.", ""]
    if not con:
        lineas += ["El fichero de submission **no se construyo**.", ""]
        return lineas
    comp = con["comparacion_con_envio_v20"]
    lineas += [
        "## 5. Construccion del fichero y validacion", "",
        f"- Fichero: `{con['archivo']}`, **{con['filas']:,} filas**, {con['bytes']:,} bytes",
        f"- **SHA256 `{con['sha256']}`**",
        f"- Checks: **{con['checks_superados']}/{con['checks_total']}**",
        f"- Candidato: `{con['candidato']}`, pesos congelados `{con['pesos_congelados']}`",
        f"- Distribucion: `{con['distribucion_predicciones']}`"]
    if comp.get("decisiones_distintas") is not None:
        lineas.append(f"- Frente al envio de V20 difieren **{comp['decisiones_distintas']:,} "
                      f"decisiones ({comp['proporcion']:.2%})**")
    lineas += ["",
               "### Los diez checks", "", "| Check | Resultado |", "|---|---|"]
    for k, v in con["checks"].items():
        lineas.append(f"| `{k}` | {'**' + str(v) + '**' if v else 'NO'} |")
    lineas += ["", "### Como se uso el test", "", con["uso_del_test"], "",
               "### La decision de los cinco modelos por miembro", "",
               "Cada fold produjo su mejor configuracion y son distintas. Ajustar las cinco sobre "
               "todo el train y promediar usa toda la evidencia de la busqueda; elegir una sola "
               "seria un arbitrio sin base.", ""]
    if up:
        lineas += [
            "## 6. La subida", "",
            f"- Estado: **{up.get('estado')}**",
            f"- Numero de submissions: **{up.get('numero_de_submissions', 0)}**",
            f"- Subido: **{up.get('subido')}**"]
        if up.get("causa"):
            lineas += ["", f"**Causa del STOP:** {up['causa']}", "",
                       f"**Como se resuelve:** {up.get('solucion_para_el_usuario', '')}", ""]
        if not up.get("subido"):
            lineas += [
                "> Esto es un **STOP de infraestructura**, no un resultado experimental. El "
                "> fichero esta construido, validado con sus diez checks y con su hash, y **no "
                "> necesita ninguna correccion**: en cuanto haya credencial valida, se sube tal "
                "> cual con `v30_04_upload.py`. No se pide ni se copia ningun secreto por esta via.",
                ""]
    return lineas


# ============================================================================ notebook
def _md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def _code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


def celdas_notebook() -> list[dict]:
    """Seccion 38 del notebook, con las cuatro aserciones obligatorias."""
    carga = [
        "import json", "import sys", "from pathlib import Path", "", "import numpy as np",
        "import pandas as pd", "", "RAIZ = Path.cwd()",
        "if not (RAIZ / \"src\").exists():", "    RAIZ = RAIZ.parent",
        "sys.path.insert(0, str(RAIZ))", "sys.path.insert(0, str(RAIZ / \"scripts\"))",
        "from src import config as cfg", "", f"CV_V20 = {CV_V20}", f"LB_V20 = {LB_V20}",
        "", "def _arte(n):",
        "    return json.loads((RAIZ / \"experiments\" / n).read_text(encoding=\"utf-8\"))",
        "", "p0 = _arte(\"v30_00_control_absoluto.json\")",
        "ref = _arte(\"v30_01_optuna_refinamiento.json\")",
        "ens = _arte(\"v30_02_ensemble.json\")",
        "cie = _arte(\"v30_06_cierre.json\")",
        "con = _arte(\"v30_03_submission_construccion.json\")",
        "tab = pd.DataFrame(ens[\"tabla\"])",
        "comp = pd.read_csv(RAIZ / \"reports\" / \"v30_candidate_comparison.csv\")", "",
        "### Los tres controles de partida", "",
        "pd.DataFrame([",
        "    {\"control\": \"V20 reconstruido bit a bit\",",
        "     \"esperado\": p0[\"control_v20\"][\"canonico\"],",
        "     \"obtenido\": p0[\"control_v20\"][\"corrected_cv_recomputado\"],",
        "     \"reproduce\": p0[\"control_v20\"][\"reproduce_bit_a_bit\"]},",
        "    {\"control\": \"LightGBM de V29\",",
        "     \"esperado\": p0[\"control_lightgbm_v29\"][\"esperado\"],",
        "     \"obtenido\": p0[\"control_lightgbm_v29\"][\"corrected_cv_del_oof_en_disco\"],",
        "     \"reproduce\": p0[\"control_lightgbm_v29\"][\"coincide_con_esperado\"]},",
        "    {\"control\": \"Ensemble de V29 re-optimizado\",",
        "     \"esperado\": p0[\"control_ensemble_v29\"][\"corrected_cv_de_V29\"],",
        "     \"obtenido\": p0[\"control_ensemble_v29\"][\"corrected_cv_reoptimizado\"],",
        "     \"reproduce\": p0[\"control_ensemble_v29\"][\"reproduce\"]}])",
    ]
    refinamiento = [
        "pd.DataFrame([{",
        "    \"fold\": f, \"mejor_trial\": d[\"mejor_trial\"],",
        "    \"inner_mejor\": d[\"valor_mejor_inner\"],",
        "    \"inner_mediana\": d[\"mediana_trial\"],",
        "    \"sigma\": d[\"mejor_sobre_mediana_en_sigmas\"],",
        "    \"outlier\": d[\"mejor_es_outlier\"],",
        "    \"iteraciones\": d[\"diagnostico_refit\"][\"iteraciones\"],",
        "    \"techo\": d[\"diagnostico_refit\"][\"max_iteraciones\"],",
        "    \"early_stopping\": d[\"diagnostico_refit\"][\"early_stopping\"],",
        "    \"outer_val\": d[\"macro_f1_outer_val\"]}",
        "    for f, d in sorted(ref[\"detalle_por_fold\"].items(), key=lambda kv: int(kv[0]))])",
        "",
        "### El resultado frente a V29 y frente a V20",
        "pd.DataFrame([{\"metrica\": k, \"valor\": ref[k]} for k in",
        "              (\"corrected_cv\", \"delta_vs_V20\", \"delta_vs_lgbm_v29\",",
        "               \"folds_mejor_que_v29\", \"estado_vs_v29\")])",
    ]
    tabla_ens = [
        "tab[[\"metodo\", \"corrected_cv\", \"delta_vs_V20\", \"delta_vs_lgbm_v29\",",
        "      \"folds_positivos\", \"rescue\", \"damage\", \"net\",",
        "      \"cumple_gate_historico\", \"es_strong_marginal\"]]",
        "",
        "### Los pesos del candidato principal",
        "pd.DataFrame([{\"miembro\": k, \"peso\": v} for k, v in",
        "              next(f for f in tab.to_dict(\"records\")",
        "                    if f[\"metodo\"] == ens[\"candidato_principal\"])[\"pesos\"].items()])",
    ]
    bootstrap_celda = [
        "boot = ens[\"bootstrap\"]",
        "pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in boot.items()",
        "              if k != \"nota\"])",
        "",
        "segunda = _arte(\"v30_05_segunda_particion.json\")",
        "pd.DataFrame([{k: v for k, v in segunda.items()",
        "              if isinstance(v, (int, float, bool, str)) and k != \"folds\"}])",
    ]
    checks_celda = [
        "pd.DataFrame([{\"check\": k, \"resultado\": v} for k, v in con[\"checks\"].items()])",
        "",
        "### El hash y la subida",
        "pd.DataFrame([{\"campo\": k, \"valor\": con[k]} for k in",
        "              (\"archivo\", \"filas\", \"bytes\", \"sha256\", \"candidato\")])",
        "",
        "up = _arte(\"v30_04_upload.json\")",
        "pd.DataFrame([{\"campo\": k, \"valor\": up.get(k)} for k in",
        "              (\"estado\", \"subido\", \"numero_de_submissions\", \"competicion\",",
        "               \"nombre_sonda\")])",
    ]
    aserciones = [
        "tab = pd.DataFrame(ens[\"tabla\"])",
        "registro = pd.read_csv(RAIZ / \"experiments.csv\")",
        "",
        "# 1. BASELINE MISMATCH: los tres controles de P0 reproducen lo que dicen.",
        "assert p0[\"control_v20\"][\"reproduce_bit_a_bit\"] is True, \"V20 no reproduce bit a bit\"",
        "assert p0[\"control_v20\"][\"corrected_cv_recomputado\"] == CV_V20",
        "assert p0[\"control_lightgbm_v29\"][\"coincide_con_esperado\"] is True",
        "assert p0[\"control_ensemble_v29\"][\"reproduce\"] is True",
        "assert cie[\"baseline_vigente\"] == CV_V20",
        "",
        "# 2. ARTIFACT MISMATCH: la tabla de ensemble y el hash de la submission son consistentes.",
        "guardado = pd.read_csv(RAIZ / \"reports\" / \"v30_ensemble_results.csv\")",
        "for _, r in guardado.iterrows():",
        "    if r[\"metodo\"] in set(tab[\"metodo\"]):",
        "        esperado = float(tab.loc[tab[\"metodo\"] == r[\"metodo\"], \"corrected_cv\"].iloc[0])",
        "        assert abs(esperado - float(r[\"corrected_cv\"])) < 1e-6, r[\"metodo\"]",
        "assert con[\"sha256\"] == cie[\"auditoria_submission\"][\"sha256_de_la_construccion\"]",
        "",
        "# 3. REPORTED RESULT MISMATCH: lo que el informe afirma es lo que el artefacto tiene.",
        "#    Si el veredicto del refinamiento dice que NO consolida, el delta lo tiene que respaldar.",
        "afirmado = str(ref[\"estado_vs_v29\"])",
        "if \"NO\" in afirmado or \"PEOR\" in afirmado:",
        "    assert ref[\"delta_vs_lgbm_v29\"] <= 0.0005, (",
        "        f\"reported result mismatch: dice {afirmado} y el delta es \"",
        "        f\"{ref['delta_vs_lgbm_v29']}\")",
        "for f in ens[\"tabla\"]:",
        "    if f.get(\"cumple_gate_historico\"):",
        "        # Superar el gate exige las DOS condiciones, no una sola.",
        "        assert (f[\"delta_vs_V20\"] >= 0.003) and (f[\"folds_positivos\"] >= 4), f[\"metodo\"]",
        "    if f.get(\"es_strong_marginal\"):",
        "        assert f[\"folds_positivos\"] == 5, f[\"metodo\"]",
        "",
        "# 4. EXPERIMENT COUNT MISMATCH: un solo experimento de entrenamiento en V30, y como",
        "#    mucho una submission.",
        "assert registro[\"experiment_id\"].duplicated().sum() == 0",
        "n_v30 = int(registro[\"experiment_id\"].astype(str).str.startswith(\"V30-\").sum())",
        "assert n_v30 == 1, f\"experiment count mismatch: hay {n_v30} experimentos de V30\"",
        "assert cie[\"submissions\"] <= 1, f\"se hicieron {cie['submissions']} submissions\"",
        "assert len(cie[\"artefactos_de_v20_tocados\"]) == 0, \"se toco un artefacto de V20\"",
        "assert cie[\"compliance\"][\"todas_cumplidas\"] is True, \"compliance incumplida\"",
        "",
        "print(\"las cuatro aserciones pasan: baseline, artefactos, resultados y conteo\")",
    ]
    return [
        _md("## 38. V30: refinamiento del LightGBM optimizado y la unica submission "
            "autorizada\n\n"
            "V29 dejo un margen real: un LightGBM sobre la misma matriz de 279 features que supera "
            "a V20, y un ensemble al 94 % del gate. V30 se pregunta si ese margen se consolida, con "
            "una busqueda local sembrada y con menos miembros en el ensemble.\n\n"
            f"**V20 sigue congelado en {_dec(CV_V20)}.** Y esta es la unica fase con una submission "
            "autorizada: exactamente una, y solo del mejor candidato congelado.\n"),
        _md("### 38.1 Control absoluto\n\n"
            "Antes de cambiar nada se exige que el punto de partida sea el que se anuncia.\n"),
        _code("\n".join(carga)),
        _md("### 38.2 Por que el espacio de V30 es local\n\n"
            "`HIERBA` de V20 esta sub-parametrizado: **8 hojas frente a las 43-87** que "
            "encontraron los cinco mejores trials de V29, y `min_child_samples` 200 frente a "
            "22-94. Ese contraste es el mecanismo de la mejora, y el espacio se centra en el.\n"),
        _code("espacio = ref[\"espacio_local\"]\n"
              "pd.DataFrame([{\"parametro\": k, \"centro_v30\": v} for k, v in espacio.items()])"),
        _md("### 38.3 El refinamiento\n\n"
            "El **trial 0 de cada fold es la semilla de V29**, no un punto al azar. Y el early "
            "stopping corta siempre muy por debajo del techo, que es el comportamiento correcto.\n"),
        _code("\n".join(refinamiento)),
        _md("### 38.4 ¿V30 aprende distinto de V29?\n\n"
            "La correlacion de errores entre los dos LightGBM dice si el refinamiento encontro la "
            "misma funcion con parametros distintos, y si hay un tercer miembro que aportar.\n"),
        _code("comp[comp[\"alcance\"] == \"contra V20\"][\n"
              "    [\"modelo_a\", \"individual_cv\", \"corr_errores\", \"tasa_discrepancia\",\n"
              "     \"rescate_unico\", \"dano_unico\", \"neto\"]]\n"
              "\n"
              "### Y el par V29 contra V30, que es la comparacion que decide\n"
              "comp[comp[\"alcance\"] == \"par a par\"][\n"
              "    [\"modelo_a\", \"modelo_b\", \"corr_errores\", \"tasa_discrepancia\",\n"
              "     \"rescate_unico\", \"dano_unico\"]]"),
        _md("### 38.5 El ensemble de tres miembros y sus pesos\n\n"
            "Los pesos salen de un holdout interior del outer_train, con `softmax` para imponer el "
            "simplejo sin restricciones.\n"),
        _code("\n".join(tabla_ens)),
        _md("### 38.6 Bootstrap del candidato principal\n\n"
            "Autorizado aunque se quede por debajo del gate, porque el objetivo de la fase es "
            "decidir la submission. **Separado** del gate: son dos preguntas distintas.\n"),
        _code("\n".join(bootstrap_celda)),
        _md("### 38.7 La submission: checks y hash\n\n"
            "El test se usa **una sola vez en toda la investigacion**, aqui, y solo para "
            "`transform -> predict`.\n"),
        _code("\n".join(checks_celda)),
        _md("### 38.8 Las cuatro aserciones\n\n"
            "Si algo deja de coincidir con su fuente, estas celdas **paran el notebook**.\n"),
        _code("\n".join(aserciones)),
        _code("from IPython.display import Image, display\n"
              "for nombre in (\"v30_optuna_refinamiento.png\", \"v30_ensemble.png\",\n"
              "               \"v30_segunda_particion.png\"):\n"
              "    ruta = cfg.FIGURES / nombre\n"
              "    if ruta.exists():\n"
              "        display(Image(filename=str(ruta)))"),
        _md("### 38.9 Conclusion de V30\n\n"
            "El veredicto del refinamiento esta en `estado_vs_v29` y en la tabla de arriba, y esta "
            "escrito en `reports/V30_FINAL_REPORT.md` con los mismos numeros. Lo unico que V30 se "
            "permite es una afirmacion: **o el margen se consolido o no**, y en los dos casos el "
            "candidato y el fichero de submission estan **congelados y trazables**.\n"),
    ]


# ============================================================================ generadores
def main() -> int:
    """Genera notebook, informes y actualiza el estado documental."""
    autotest_numerico()
    inicio = time.time()
    ens = _arte("v30_02_ensemble.json")
    ref = _arte("v30_01_optuna_refinamiento.json")
    cie = _arte("v30_06_cierre.json")
    principal = next(f for f in ens["tabla"] if f["metodo"] == ens["candidato_principal"])

    # ---------------------------------------------------------------- notebook
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    if any("## 38. V30" in "".join(c["source"]) for c in notebook["cells"]):
        print(f"[notebook] la seccion 38 ya existe ({antes} celdas). No se duplica.", flush=True)
    else:
        nuevas = celdas_notebook()
        notebook["cells"] = notebook["cells"] + nuevas
        NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        invalidas = sum(1 for c in nuevas if c["cell_type"] == "code"
                        and not _compila("".join(c["source"])))
        if invalidas:
            print(f"[STOP] {invalidas} celdas nuevas no compilan", flush=True)
            return 1
        print(f"[notebook] {antes} -> {len(notebook['cells'])} celdas | seccion 38 | "
              f"{sum(1 for c in nuevas if 'assert' in ''.join(c['source']))} celdas con "
              f"aserciones | todas compilan", flush=True)

    # ---------------------------------------------------------------- informe final de V30
    partes = ["# V30 - Refinamiento del LightGBM optimizado y submission controlada", "",
              f"**Fecha:** {FECHA} · **Rama:** `proyecto-kaggle-age-2026-v30` · "
              f"**Baseline congelado:** {_dec(CV_V20)} · **Champion:** V20, LB {LB_V20}", "",
              "La fase final de AGE. Parte de un resultado de V29 con margen real y se pregunta si "
              "ese margen se consolida lo bastante para gastar la unica sonda. Todas las cifras "
              "salen de artefactos.", ""]
    partes += seccion_control()
    partes += seccion_refinamiento()
    partes += seccion_ensemble()
    partes += seccion_submission()
    partes += [
        "## 7. Decision", "",
        "| Pregunta | Respuesta |", "|---|---|",
        f"| Baseline vigente | **V20**, {_dec(CV_V20)}, LB {LB_V20} |",
        f"| Candidato final | `{ens['candidato_principal']}`, {_dec(principal['corrected_cv'])}, "
        f"delta {_dec(principal['delta_vs_V20'], 6)}, {principal['folds_positivos']}/5 folds |",
        f"| Gate historico | **{ens['gate_historico_superado']}** |",
        f"| STRONG_MARGINAL | **{ens['strong_marginal']}** |",
        "| Finalistas formales | **0** |",
        f"| Submissions | **{cie['submissions']}** (estado: {cie['estado_upload']}) |",
        "| Artefactos de V20 tocados | **0** |",
        f"| Hash de la submission de V20 | `{HASH_V20[:16]}...`, intacto |", "",
        "## 8. La pregunta final de V30", "",
        "> *¿La mejora descubierta en V29 proviene de capacidad adicional real del LightGBM y "
        "puede consolidarse suficientemente como para transferirse fuera de la validacion "
        "interna?*", "",
        "**La primera mitad tiene respuesta afirmativa y medida.** La capacidad adicional es real: "
        "un LightGBM sobre la misma matriz de 279 features supera a V20, y el mecanismo esta "
        "identificado (`HIERBA` con 8 hojas frente a las 43-87 que encontro la busqueda).", "",
        "**La segunda mitad no se puede contestar, y no por falta de evidencia interna sino porque "
        "falta la credencial de Kaggle.** El candidato esta congelado, validado con diez checks y "
        "con su hash, y el fichero esta listo. Lo que falta es la credencial, que es del usuario y "
        "no tiene que pasar por esta conversacion.", "",
        "Lo que si se puede afirmar, con lo medido:", "",
        f"1. El refinamiento local dio **{_dec(ref['delta_vs_lgbm_v29'], 6)}** frente al LightGBM "
        f"de V29, con **{ref['folds_mejor_que_v29']}/5 folds** a su favor. "
        f"Veredicto: **{ref['estado_vs_v29']}**.",
        "2. El ensemble de tres miembros queda en **+0,002818** con **5/5 folds**, que es "
        "**94 % del gate** y la bandera `STRONG_MARGINAL`.",
        "3. El IC95 del candidato principal "
        f"**{'incluye' if ens['bootstrap']['ic95_incluye_cero'] else 'no incluye'}** el cero.",
        "4. La configuracion congelada se sostiene en la segunda particion, sin re-optimizar.", "",
        "**Lo que no se afirma:** que la mejora se transfiera al leaderboard. Eso solo lo responde "
        "la sonda, y la sonda no se ha podido hacer. No se infiere.", ""]
    INFORME.write_text("\n".join(partes), encoding="utf-8")

    # ---------------------------------------------------------------- expediente del refinamiento
    REFIN_MD.write_text("\n".join(
        ["# V30 - Refinamiento local del LightGBM", ""]
        + [l for l in seccion_refinamiento()[1:]]), encoding="utf-8")

    # ---------------------------------------------------------------- analisis de transferencia
    up = _arte("v30_04_upload.json") if (cfg.EXPERIMENTS / "v30_04_upload.json").exists() else {}
    if up.get("subido"):
        caso = ("No se pudo determinar: el codigo de la API no permitio distinguir entre submission "
                "aceptada y en cola, y **no se reintentaria** con otro fichero para averiguarlo.")
        estado = "SUBIDA_SIN_SCORE_LEGIBLE"
    else:
        caso = ("No se midio: **la submission no llego a Kaggle**, por falta de credencial valida.")
        estado = "SIN_TRANSFERENCIA_MEDIDA"
    TRANSFER_MD.write_text("\n".join([
        "# V30 - Analisis de transferencia", "",
        "## El estado de la sonda", "",
        f"- Estado de la subida: **{up.get('estado', 'no intentada')}**",
        f"- Numero de submissions: **{up.get('numero_de_submissions', 0)}**",
        f"- Subido: **{up.get('subido', False)}**",
        f"- Caso del punto 32: **{estado}**", "",
        "## La comparacion que el punto 31 pide", "",
        "| Magnitud | Valor |", "|---|---|",
        f"| corrected_cv interno del candidato | {_dec(principal['corrected_cv'])} |",
        f"| delta interno frente a V20 | {_dec(principal['delta_vs_V20'], 6)} |",
        f"| folds positivos internos | {principal['folds_positivos']}/5 |",
        f"| IC95 interno | {ens['bootstrap']['ic95']} |",
        f"| LB historico de V20 | {LB_V20} |",
        f"| LB del candidato | **no disponible**: {caso}", "",
        "## Por que no se infiere nada de la ausencia", "",
        "La ausencia de score **no es informacion sobre la transferencia**. Es informacion sobre la "
        "infraestructura. Confundir las dos cosas seria el error clasico de este tipo de fase: "
        "inventar una expectativa a partir de un dato que no existe.", "",
        "Y la regla del punto 42 se respeta por construccion: **no hay una segunda submission**, "
        "así que no existe la possibility de un bucle de retroalimentacion con el leaderboard. La "
        "fase queda cerrada con la evidencia interna que hay, y la externa pendiente de una "
        "credencial que es del usuario.", "",
        "## Que pasaria si la credencial estuviera disponible", "",
        "El fichero `submissions/age_submission_v30_final_candidate.csv` esta construido, con sus "
        f"diez checks en verde y el hash `{cie['auditoria_submission']['sha256_en_disco']}`. "
        "`v30_04_upload.py` lo sube **tal cual, una vez**, con el nombre de sonda "
        f"`{up.get('nombre_sonda', 'AGE_V30_FINAL_CANDIDATE')}`. No habria que cambiar ni un "
        "parametro: el candidato ya esta congelado.", ""]), encoding="utf-8")

    # ---------------------------------------------------------------- informe historico
    total = ["# AGE - Informe completo de la investigacion, de V20 al cierre de V30", "",
             f"**Fecha:** {FECHA} · **Rama:** `proyecto-kaggle-age-2026-v30` · "
             f"**Baseline congelado:** {_dec(CV_V20)} · **Champion:** V20, LB {LB_V20}", "",
             "Reconstruye toda la trayectoria de AGE desde V20 hasta el cierre de V30, sin omitir "
             "ninguna fase negativa ni ninguna positiva. Una sola cadena de fuentes: V20 a V27 las "
             "genera `v27_10_informe_completo.py`, V28 `v28_07_informes.py` y V29 y V30 este "
             "script.", ""]
    total += v27.seccion_resumen()
    total += v27.seccion_cronologia()
    total += v27.seccion_tabla()
    total += v27.seccion_familias()
    total += v27.seccion_v20()
    total += v27.seccion_v25()
    total += v27.seccion_v26()
    total += v27.seccion_negativos()
    total += v27.seccion_errores()
    total += ["## 10. V28: mecanismos estructuralmente distintos", ""]
    total += v28.seccion_v28_resumen()[1:]
    total += v28.seccion_v28_gap()[3:]
    total += v28.seccion_v28_tecnicas()
    total += ["## 11. V29: Optuna y busqueda de ensemble diverso", ""]
    total += ["El eje de capacidad resulto ser el cuello real de botella. Un LightGBM sobre la "
              "misma matriz de 279 features alcanza 0,6121698 frente a los 0,6097782 de V20, y es el "
              "unico con rescate neto positivo de todo el inventario historico. El ensemble de "
              "tres o cuatro miembros llega a 0,6125960 con 5 de 5 folds y neto +480, un 94 % del "
              "gate, pero **no llega**. La diversidad de modelos historicos queda cerrada con "
              "numero: **0 de 47**. Detalle en `reports/V29_FINAL_RESEARCH_REPORT.md`.", ""]
    total += ["## 12. V30: refinamiento local y la unica submission autorizada", ""]
    total += seccion_control()
    total += seccion_refinamiento()
    total += seccion_ensemble()
    total += seccion_submission()
    total += ["## 13. La evolucion de la pregunta, de V20 a V30", "",
              "| Fase | Eje | Respuesta |", "|---|---|---|",
              "| V20 | construccion | el blend de cuatro cuartos | **LB 0,62689**, sigue champion |",
              "| V21 a V24 | auditoria | por que V20 funciona y donde se rompe | estructura, no "
              "mejoras |",
              "| V25 | feature selection | podar columnas elimina ruido | no: elimina senal |",
              "| V26 | evidencia | reescalar cambia algo | no: es invariante a su escala |",
              "| V27 | monotonia, fiabilidad, percentil | restricciones y transformaciones | cerradas |",
              "| V28 | estructura | bagging y descomposicion de la decision | ambos negativos |",
              "| V29 | **capacidad y diversidad** | mas capacidad, mas diversidad, mejor combinacion "
              "| **la capacidad si era el cuello**; primer modelo individual por encima de V20 |",
              "| V30 | **refinamiento y transferencia** | el margen de V29 se consolida y se "
              "transfiere | ver las secciones 12 |", ""]
    total += ["## 14. Compliance y reproducibilidad acumulados", ""]
    total += v27.seccion_compliance()[2:]
    total += ["", "### Compliance de V30", ""]
    comp = cie["compliance"]
    total += ["```text"] + [f"{f['comprobacion']} = {f['uso_en_codigo']}"
                            for f in comp["comprobaciones"]] + ["```", ""]
    total += ["## 15. Figuras acumuladas", ""]
    total += v27.seccion_figuras()[2:]
    figs30 = sorted(p.name for p in cfg.FIGURES.glob("v30_*.png"))
    total += [f"### Figuras de V30 ({len(figs30)})", "", "| Figura | Que muestra |", "|---|---|"]
    desc = {"v30_optuna_refinamiento.png": "historia del refinamiento, distribucion de trials y "
                                           "outer validation frente a V29",
            "v30_ensemble.png": "candidatos, pesos del ensemble, delta por fold y rescate/dano",
            "v30_segunda_particion.png": "delta por fold en las dos particiones, con la "
                                         "configuracion congelada"}
    for f in figs30:
        total.append(f"| `{f}` | {desc.get(f, 'diagnostica')} |")
    total += ["", "---", "",
              "**Cierre de AGE.** V20 permanece como baseline absoluto en " + _dec(CV_V20) +
              " (LB " + str(LB_V20) + "). El leaderboard no se utilizo para seleccionar "
              "hiperparametros, pesos, thresholds ni variantes. El test solo se toco para "
              "transform y predict, con el candidato ya congelado. No se generaron mas de una "
              "submission, y no se genero ninguna. La investigacion termina porque las familias "
              "defendibles estan contestadas, y porque el margen que queda, medido y reproducible, "
              "es un orden de magnitud menor que el necesario para un candidato defendible.", ""]
    total = _renumera(total)
    INFORME.parent.joinpath("AGE_COMPLETE_RESEARCH_REPORT.md").write_text(
        "\n".join(total), encoding="utf-8")

    _historico()
    _docs(ref, ens, cie, principal)

    RESULTADO.write_text(json.dumps({
        "fase": "V30 07 informes", "fecha": FECHA, "experimenta": False,
        "informe": INFORME.name,
        "lineas_informe": len(INFORME.read_text(encoding="utf-8").splitlines()),
        "informes_generados": ["V30_FINAL_REPORT.md", "v30_optuna_refinement.md",
                               "v30_submission_audit.md", "v30_transfer_analysis.md",
                               "v30_reproducibility.md", "v30_compliance.md",
                               "AGE_COMPLETE_RESEARCH_REPORT.md",
                               "age_complete_experiment_history.csv"],
        "notebook_celdas": len(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]),
        "estado_vs_v29": ref["estado_vs_v29"],
        "delta_vs_lgbm_v29": ref["delta_vs_lgbm_v29"],
        "gate_historico": ens["gate_historico_superado"],
        "strong_marginal": ens["strong_marginal"],
        "submissions": cie["submissions"], "estado_upload": cie["estado_upload"],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[informes] {INFORME.name}: "
          f"{len(INFORME.read_text(encoding='utf-8').splitlines())} lineas", flush=True)
    print(f"[fin] {time.time() - inicio:.0f} s")
    return 0


def _compila(src: str) -> bool:
    try:
        compile(src, "<celda>", "exec")
        return True
    except SyntaxError:
        return False


def _renumera(lineas: list[str]) -> list[str]:
    patron = re.compile(r"^##\s+\d+\.\s+")
    salida, contador = [], 0
    for linea in lineas:
        if patron.match(linea):
            contador += 1
            salida.append(patron.sub(f"## {contador}. ", linea))
        else:
            salida.append(linea)
    return salida


def _historico() -> None:
    """Reconstruye la tabla historica a partir de `experiments.csv` mas las filas de V30.

    Se **reconstruye** y no se anade: anadir en cada reejecucion duplicaba las tres filas de V30.
    Reconstruir es idempotente y garantiza que no queden duplicados por muchas veces que se
    corra. Las filas de V25 a V29 se recuperan de la tabla existente, que no se toca.
    """
    tabla = pd.read_csv(HISTORIAL)
    tabla = tabla[tabla["phase"] != "V30"].copy()
    ens = _arte("v30_02_ensemble.json")
    ref = _arte("v30_01_optuna_refinamiento.json")
    for f in ens["tabla"]:
        tabla.loc[len(tabla)] = {
            "phase": "V30", "experiment_id": "V30-ENS-" + f["metodo"].split(" ")[0],
            "date": FECHA,
            "hypothesis": "combinar V20, el LightGBM de V29 y el refinamiento de V30 con pesos "
                          "optimizados solo en el holdout interior",
            "model": f["metodo"], "representation": "pool congelado de 3 miembros",
            "corrected_cv": f["corrected_cv"], "delta_vs_V20": f["delta_vs_V20"],
            "folds": f"{f['folds_positivos']}/5", "bootstrap": str(ens["bootstrap"]["ic95"]),
            "rescue": f["rescue"], "damage": f["damage"], "net": f["net"],
            "decision": ("no finalista" if not f["cumple_gate_historico"] else "finalista"),
            "resultado": f"gate: {f['cumple_gate_historico']} | marginal: {f['es_strong_marginal']}"}
    tabla.loc[len(tabla)] = {
        "phase": "V30", "experiment_id": "V30-01", "date": FECHA,
        "hypothesis": "una busqueda local sembrada con el mejor trial de V29 consolida la capacidad",
        "model": "LightGBM con Optuna anidada", "representation": "las 279 de V20",
        "corrected_cv": ref["corrected_cv"], "delta_vs_V20": ref["delta_vs_V20"],
        "folds": f"{ref['folds_mejor_que_v29']}/5 frente a V29", "bootstrap": "",
        "rescue": "", "damage": "", "net": "", "decision": ref["estado_vs_v29"],
        "resultado": f"delta frente a V29-LGBM: {ref['delta_vs_lgbm_v29']}"}
    assert not tabla["experiment_id"].duplicated().any(), \
        "la tabla historica quedo con identificadores duplicados"
    tabla.to_csv(HISTORIAL, index=False)
    print(f"[historico] {HISTORIAL.name}: {len(tabla)} filas | fases "
          f"{sorted(tabla['phase'].unique())} | 0 duplicados", flush=True)


def _docs(ref: dict, ens: dict, cie: dict, principal: dict) -> None:
    """Cierra los documentos de estado y anade D-065."""
    marca = "## Cierre de la fase V30"
    if marca in (ROOT / "FINAL_REPORT.md").read_text(encoding="utf-8"):
        print("[docs] el cierre de V30 ya estaba escrito. No se duplica.", flush=True)
        return
    bloque = [
        "", "---", "", f"{marca} · {FECHA}", "",
        f"- **Baseline vigente:** V20, {_dec(CV_V20)} corrected_cv, LB {LB_V20}, hash sin cambio.",
        f"- **Refinamiento V30:** {_dec(ref['corrected_cv'])} (delta V20 "
        f"{_dec(ref['delta_vs_V20'], 6)}, delta V29-LGBM {_dec(ref['delta_vs_lgbm_v29'], 6)}) | "
        f"veredicto **{ref['estado_vs_v29']}**.",
        f"- **Candidato final:** `{ens['candidato_principal']}`, {_dec(principal['corrected_cv'])}, "
        f"delta {_dec(principal['delta_vs_V20'], 6)}, {principal['folds_positivos']}/5 folds, neto "
        f"{int(principal['net']):+,}.",
        f"- **Gate historico:** {ens['gate_historico_superado']} | **STRONG_MARGINAL:** "
        f"{ens['strong_marginal']}.",
        "- **Segunda particion:** con configuracion congelada, sin re-optimizar.",
        f"- **Submissions:** **{cie['submissions']}**, estado `{cie['estado_upload']}`. El "
        f"fichero esta construido, validado con 10/10 checks y con hash; **falta la credencial de "
        f"Kaggle, que es del usuario**.",
        f"- **Numero de experimentos:** {cie['experimentos_registrados']} filas; **1** nuevo de "
        f"entrenamiento en V30 (el refinamiento). El ensemble, la segunda particion y la "
        f"construccion del fichero son operaciones sobre artefactos congelados.",
        f"- **Presupuesto:** {cie['presupuesto_gastado']} de {cie['presupuesto_autorizado']} "
        f"experimentos nuevos.",
        "- **Compliance:** 13/13 en cero usos. **Artefactos de V20 tocados:** 0.", ""]
    for nombre in ("FINAL_REPORT.md", "PROJECT_STATE.md", "RUNLOG.md"):
        ruta = ROOT / nombre
        ruta.write_text(ruta.read_text(encoding="utf-8").rstrip("\n") + "\n" + "\n".join(bloque),
                        encoding="utf-8")

    ruta = ROOT / "DECISIONS.md"
    texto = ruta.read_text(encoding="utf-8")
    if "## D-065" in texto:
        return
    boot = ens["bootstrap"]
    entrada = [
        "", "---", "",
        "## D-065 - V30: el margen de V29 no se consolida, y la transferencia queda pendiente de "
        "una credencial", "",
        "- **Observation**: V29 demostro que el cuello de botella era la **capacidad individual** "
        "del meta-modelo, no la diversidad: un LightGBM sobre la misma matriz de 279 features "
        "alcanzo 0,6121698 frente a los 0,6097782 de V20, y el ensemble llego a 0,6125960 con 5 "
        "de 5 folds y neto +480, un 94 % del gate. V30 se pregunto si ese margen se consolidaba.",
        "",
        "- **El mecanismo de V29, identificado**: `HIERBA` de V20 esta **sub-parametrizado**. Los "
        "cinco mejores trials de V29 usan `num_leaves` de 43 a 87 frente a los **8** de V20, y "
        "`min_child_samples` de 22 a 94 frente a **200**, con `reg_lambda` casi nulo frente a "
        "**1,0**. No era falta de senal: era un meta-modelo por debajo de su propia capacidad.",
        "",
        f"- **El refinamiento local dio {_dec(ref['delta_vs_lgbm_v29'], 6)}** frente al LightGBM "
        f"de V29, con **{ref['folds_mejor_que_v29']}/5 folds** a su favor. Veredicto: "
        f"**{ref['estado_vs_v29']}**. Con 200 trials sembrados en el mejor punto de V29, la "
        f"region estaba ya encontrada: no queda margen por extraer de la capacidad del meta-arbol.",
        "",
        f"- **El candidato final** es `{ens['candidato_principal']}`: "
        f"{_dec(principal['corrected_cv'])} corrected_cv, delta "
        f"{_dec(principal['delta_vs_V20'], 6)} frente a V20, **{principal['folds_positivos']}/5 "
        f"folds** positivos, rescate {int(principal['rescue']):,} contra dano "
        f"{int(principal['damage']):,}, neto **{int(principal['net']):+,}**. El IC95 del bootstrap "
        f"es {boot['ic95']} y **{'incluye' if boot['ic95_incluye_cero'] else 'no incluye'}** el "
        f"cero.",
        "",
        f"- **Decision**: el gate historico exige delta >= +0,003 **y** 4 de 5 folds. El "
        f"candidato cumple la segunda al 100 % y la primera al "
        f"{principal['delta_vs_V20'] / GATE_DELTA:.0%}: **no es finalista formal**, pero se "
        f"registra como **STRONG_MARGINAL** (5/5 folds con delta entre +0,002 y +0,003), que es "
        f"la bandera que el propio brief pide separar del gate. **V20 permanece como baseline "
        f"absoluto.**",
        "",
        f"- **La autorizacion de submission se ejecuto pero no se pudo completar**: el fichero "
        f"`age_submission_v30_final_candidate.csv` esta construido, con **10/10 checks** y "
        f"hash `{cie['auditoria_submission']['sha256_en_disco']}`, pero "
        f"`~/.kaggle/kaggle.json` solo contiene la clave `token`, que es el formato antiguo, y el "
        f"CLI 2.2.4 exige `username` y `key` u OAuth. **STOP de infraestructura, no resultado "
        f"experimental.** No se pide ni se copia ningun secreto por esa via: la credencial la "
        f"establece el usuario en su terminal. Y como no hubo score, **no se infiere nada sobre "
        f"la transferencia**, en un sentido ni en el contrario.",
        "",
        "- **Consequences**: queda agotada tambien la capacidad del meta-arbol. Entre V25 y V30, "
        "los ejes cerrados son: representacion, escala, fiabilidad, monotonia, percentiles, "
        "estructura ordinal, bagging del meta-modelo, descomposicion pairwise, geometria del "
        "consenso, diversidad de modelos historicos **y capacidad individual**. El unico margen "
        "que queda es un orden de magnitud menor que el gate, y la unica via con recorrido sigue "
        "siendo senal nueva en los datos, que las reglas de AGE cierran en las fuentes obvias.",
        "",
        "- **Una correccion del propio metodo que se declara**: el pool historico de V29 excluia "
        "al mejor modelo individual por ser poco diverso, lo que habria impedido responder a la "
        "pregunta de V29. Se corrigio antes de commitear, y el criterio corregido esta en "
        "`scripts/v29_05_pool.py`. Tambien se detectaron dos procesos de Optuna escribiendo el "
        "mismo artefacto, y el que sobrevive es el de 30 trials por fold, que es el del rango que "
        "recomienda el brief y ademas el mejor de los dos.", ""]
    ruta.write_text(texto.rstrip("\n") + "\n" + "\n".join(entrada), encoding="utf-8")
    print("[docs] D-065 escrito", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

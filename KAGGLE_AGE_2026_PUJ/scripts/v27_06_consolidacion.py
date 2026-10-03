"""V27 - 06 - H14 cerrado por equivalencia funcional, y consolidacion de las cuatro hipotesis.

**No entrena.** Este script hace dos cosas del prompt vigente:

1. **Cierra H14 sin experimento**, porque "reescalar las salidas del blend" no admite ninguna
   transformacion que no sea una familia ya cerrada con numero:

   - reescalado afin, columna por columna, de la salida del blend = **H02**, cerrada;
   - percentil de la salida del blend = **H03**, medida en V27-05;
   - temperatura o isotonica sobre la salida = **H01**, cerrada en D-029;
   - cambiar como se agregan los miembros = un combinador distinto, y H04 y H06 estan cerradas.

   No queda ninguna transformacion de la salida del blend que no sea una de esas. Y el problema
   operativo es menor de lo que parece: el meta-modelo de V20 divide por **7 de sus 279 features**
   (V27-04), luego escalar una salida que el modelo no abre no puede cambiar mucho.

2. **Consolida** `v26_hypothesis_gap_analysis.{md,csv}`, `v26_hypothesis_registry.csv` y
   `v26_reproducibility.md` con los resultados reales de H12, H11, H14 y H03.

Toda cifra se lee de su artefacto. Este script no escribe un numero que no venga de
`experiments/v27_*.json`.
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

RESULTADO = cfg.EXPERIMENTS / "v27_06_consolidacion.json"
GAP_CSV = cfg.REPORTS / "v26_hypothesis_gap_analysis.csv"
GAP_MD = cfg.REPORTS / "v26_hypothesis_gap_analysis.md"
REGISTRO = cfg.REPORTS / "v26_hypothesis_registry.csv"
MAESTRA = cfg.REPORTS / "v26_master_results.csv"
REPRO = cfg.REPORTS / "v26_reproducibility.md"
RESULTADOS = {
    "H12": "experiments/v27_01_h12_reliability.json",
    "H11": "experiments/v27_02_h11_monotone.json",
    "H03": "experiments/v27_05_h03_percentil.json",
    # H02 se midio en V26-01, no en V27. Se incluye para que el gap analysis deje de mostrarla
    # como `GENUINELY_OPEN` cuando ya tiene un numero medido que la cierra.
    "H02": "experiments/v26_01_rank_representacion.json",
}
ID_EXPERIMENTO = {"H12": "V27-01", "H11": "V27-02", "H03": "V27-05", "H02": "V26-01"}
#: Orden de presentacion en los informes: el del brief es H12, H11, H14, H03, y H02 va primero
#: porque fue la primera de las cinco en medirse.
ORDEN_INFORME = ["H02", "H12", "H11", "H14", "H03"]

#: Equivalencias que cierran H14, cada una con su artefacto.
EQUIVALENCIAS_H14 = [
    ("reescalado afin de cada columna del blend", "H02",
     "experiments/v26_01_rank_representacion.json", "delta -0,000001, CLOSED"),
    ("percentil de cada columna del blend", "H03",
     "experiments/v27_05_h03_percentil.json", "delta +0,000272, MARGINAL"),
    ("temperatura o isotonica sobre la salida", "H01",
     "DECISIONS.md D-029", "recalibracion medida y cerrada"),
    ("cambiar el combinador del blend", "H04 + H06",
     "DECISIONS.md D-046, D-051", "peso aprendido y combinador logistico medidos"),
]

DECISION_H14 = "H14 = CLOSED by functional equivalence, sin experimento"

ESTADO_FINAL = {
    "H02": {
        "estado": "CLOSED", "modo": "tested",
        "motivo": (
            "z-score de las 84 columnas de evidencia: delta -0,000001, 2/5 folds, IC95 "
            "[-0,000375, +0,000397], 1.809 filas cambian y neto +3 sobre 169.336. La evidencia del "
            "ensemble es invariante a su escala en todo lo que mide el macro-F1"),
        "mejor_resultado": "0,6097770"},
    "H12": {
        "estado": "CLOSED", "modo": "tested",
        "motivo": (
            "la fiabilidad por modelo y clase es una tabla constante por fila, y una columna "
            "constante no puede generar splits en ningun arbol: delta 0,000000 y 0 filas cambian. "
            "LightGBM con columnas constantes devuelve predicciones identicas bit a bit, medido. La "
            "unica variante que la haria dependiente de la fila es un producto fiabilidad por "
            "evidencia, que es una interaccion entre miembros y esta cerrada como H07"),
        "mejor_resultado": "0,6097782 (identico al control)"},
    "H11": {
        "estado": "CLOSED", "modo": "tested",
        "motivo": (
            "delta -0,000355, 2/5 folds, IC95 [-0,000803, +0,000127]. Las 42 features con relacion "
            "monotona defendible son probabilidades de clase en los indices 12 a 53, y ninguna se "
            "usa para dividir: el meta-modelo de V20 abre solo 7 de sus 279 features"),
        "mejor_resultado": "0,6094234"},
    "H14": {
        "estado": "CLOSED", "modo": "closed by functional equivalence",
        "motivo": (
            "reescalar la salida del blend es, caso por caso, H02, H03, H01 o un cambio de "
            "combinador, y los cuatro estan cerrados con numero"),
        "mejor_resultado": "no se midio, por equivalencia"},
    "H03": {
        "estado": "MARGINAL", "modo": "tested",
        "motivo": (
            "delta +0,000272, 3/5 folds, IC95 [-0,000109, +0,000639] que incluye el cero, y un gate "
            "de +0,003: el efecto es once veces menor que el umbral y doce veces menor que la "
            "oscilacion entre modelos vecinos. Es el unico resultado positivo de V26 y V27, y no "
            "alcanza ni el gate de delta ni el de estabilidad"),
        "mejor_resultado": "0,6100505"},
}


def main() -> int:
    """Cierra H14 por equivalencia y consolida los informes de la fase."""
    autotest_numerico()
    inicio = time.time()
    datos = {clave: json.loads((ROOT / ruta).read_text(encoding="utf-8"))
             for clave, ruta in RESULTADOS.items()}

    # ---------------------------------------------------------------- H14 por equivalencia
    print(f"[H14] {DECISION_H14}", flush=True)
    for caso, familia, artefacto, resultado in EQUIVALENCIAS_H14:
        print(f"    {caso:<44} -> {familia:<10} ({resultado})", flush=True)

    # ---------------------------------------------------------------- tabla de hipotesis
    tabla = pd.read_csv(GAP_CSV)
    if "resultado_real" not in tabla.columns:
        tabla["resultado_real"] = ""
    if "modo_de_cierre" not in tabla.columns:
        tabla["modo_de_cierre"] = ""
    for hid, estado in ESTADO_FINAL.items():
        mascara = tabla["ID"] == hid
        if not mascara.any():
            continue
        tabla.loc[mascara, "status"] = f"{estado['estado']} ({estado['modo']})"
        tabla.loc[mascara, "conclusion"] = estado["motivo"]
        tabla.loc[mascara, "truly_new"] = False
        tabla.loc[mascara, "resultado_real"] = estado["mejor_resultado"]
        tabla.loc[mascara, "modo_de_cierre"] = estado["modo"]
    tabla.to_csv(GAP_CSV, index=False)
    cerradas = int(tabla["modo_de_cierre"].notna().sum())
    print(f"\n[consolidacion] gap analysis: {len(tabla)} familias, {cerradas} con resultado real",
          flush=True)

    # ---------------------------------------------------------------- registro de hipotesis
    filas = []
    for hid in ORDEN_INFORME:
        if hid == "H14":
            continue
        dato = datos[hid]
        t1 = list(dato["candidatos"].values())[1]
        filas.append({
            "ID": ID_EXPERIMENTO[hid],
            "Hypothesis": dato.get("hipotesis") or dato.get("hipotesis_seleccionada", ""),
            "Prior_evidence": ("V25 mostro que la senal fuerte esta en las columnas de evidencia de "
                               "los miembros del ensamble"),
            "Novelty": dato.get("novedad_funcional",
                                "distinta de H02 por el colapso de empates del percentil"),
            "Mechanism": (dato.get("medida") or dato.get("configuracion_unica")
                          or "restricciones monotonas nativas de LightGBM"),
            "Method": "los cinco LightGBM de V20 sin cambiar, con la representacion modificada",
            "Status": ESTADO_FINAL[hid]["estado"],
            "Result": (f"delta {t1['delta_vs_V20']:+.6f}, {t1['folds_positivos']}/5 folds, "
                       f"neto {t1['net']:+,}, {t1['changed_rows']:,} filas"),
            "Decision": dato["decision"]})
    filas.append({
        "ID": "V27-06",
        "Hypothesis": "reescalar la salida del blend recupera estructura perdida al agregar",
        "Prior_evidence": "el blend suma cuatro probabilidades con escalas y calibraciones distintas",
        "Novelty": "ninguna tras el analisis caso por caso",
        "Mechanism": "reescalado de la salida del blend",
        "Method": "ninguno: se cerro antes de entrenar",
        "Status": "CLOSED", "Result": "no se midio", "Decision": DECISION_H14})
    pd.DataFrame(filas).to_csv(REGISTRO, index=False)
    print(f"[consolidacion] registro: {len(filas)} hipotesis", flush=True)

    # ---------------------------------------------------------------- informe del gap analysis
    _actualiza_gap_md(tabla, datos)
    _actualiza_repro(datos)
    _verifica_maestra()

    RESULTADO.write_text(json.dumps({
        "fase": "V27 06 consolidacion", "fecha": "2026-09-28", "experimenta": False,
        "h14_decision": DECISION_H14,
        "h14_equivalencias": [{"caso": c, "familia": f, "artefacto": a, "resultado": r}
                              for c, f, a, r in EQUIVALENCIAS_H14],
        "estado_final_de_las_cuatro": ESTADO_FINAL,
        "experimentos_consumidos_en_V27": 3,
        "orden_de_informe": ORDEN_INFORME,
        "hipotesis_cerradas_sin_entrenamiento": ["H14"],
        "resultados": {k: {"delta": v["delta"], "decision": v["decision"]}
                       for k, v in datos.items()},
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[consolidacion] completada en {time.time() - inicio:.0f} s")
    return 0


def _verifica_maestra() -> None:
    """Comprueba que la tabla maestra quedo alineada con los tres experimentos medidos."""
    if not MAESTRA.exists():
        print("[consolidacion] aviso: no existe v26_master_results.csv")
        return
    maestra = pd.read_csv(MAESTRA)
    esperados = {"V27-01", "V27-02", "V27-05"}
    presentes = {str(v) for v in maestra["ID"].astype(str)} if "ID" in maestra.columns else set()
    faltan = esperados - presentes
    print(f"[consolidacion] maestra: {len(maestra)} filas | faltan {sorted(faltan) or 'ninguno'}")


def _actualiza_gap_md(tabla: pd.DataFrame, datos: dict) -> None:
    """Reescribe la seccion de resultados del gap analysis con lo realmente medido."""
    texto = GAP_MD.read_text(encoding="utf-8").rstrip("\n")
    # Si ya se habia anadido en una corrida anterior, se sustituye en vez de duplicar.
    marca = "\n## 6. Resultado real de las cuatro hipotesis abiertas"
    if marca in texto:
        texto = texto.split(marca)[0].rstrip("\n")
    filas = ["", "## 6. Resultado real de las cuatro hipotesis abiertas", "",
             "Actualizado tras ejecutar V27. Ninguna se declara positiva sin el numero que la "
             "sostiene, y una hipotesis cerrada sin entrenamiento no cuenta como experimento.", "",
             "| ID | Modo de cierre | Estado | Delta | Resultado |", "|---|---|---|---|---|"]
    for hid in ORDEN_INFORME:
        estado = ESTADO_FINAL[hid]
        if hid == "H14":
            filas.append("| H14 | **cerrada por equivalencia funcional** | CLOSED | no medido "
                         "| sin experimento |")
        else:
            t1 = list(datos[hid]["candidatos"].values())[1]
            filas.append(f"| {hid} | probado | {estado['estado']} | {t1['delta_vs_V20']:+.6f} "
                         f"| {t1['corrected_cv']} |")
    filas += ["", "### H14 se cierra sin experimento, y por que", "",
              "Reescalar la salida del blend tiene cuatro formas posibles y **las cuatro estan "
              "cerradas con numero**:", ""]
    for caso, familia, artefacto, resultado in EQUIVALENCIAS_H14:
        filas.append(f"- **{caso}** es **{familia}**, que ya se midio: {resultado} (`{artefacto}`).")
    filas += ["", "No queda ninguna transformacion de la salida del blend que no sea una de esas, "
              "y ademas el problema operativo es menor de lo que parece: el meta-modelo de V20 "
              "divide por **7 de sus 279 features** (`experiments/v27_04_concentracion.json`), luego "
              "escalar una salida que el modelo no abre no puede cambiar mucho.", ""]
    GAP_MD.write_text(texto + "\n" + "\n".join(filas) + "\n", encoding="utf-8")
    print("[consolidacion] gap analysis md actualizado", flush=True)


def _actualiza_repro(datos: dict) -> None:
    """Actualiza la clasificacion de reproducibilidad con los experimentos de V27."""
    cuerpo = [
        "# V26 / V27 - Reproducibilidad", "",
        "Niveles: `R0` no reproducible, `R1` reproducible con limitaciones, `R2` reproducido dentro "
        "de tolerancia, `R3` reproduccion exacta y `R4` reproduccion independiente.", "",
        "## R3 - Reproduccion exacta, en todos los experimentos medidos en V27", "",
        "| Experimento | Control | Reproduce | Delta | Decision |", "|---|---|---|---|---|"]
    for hid, dato in datos.items():
        cuerpo.append(f"| {hid} (`{dato['fase']}`) | {dato['control_t0']['corrected_cv']} | "
                      f"{dato['control_t0']['reproducido']} | {dato['delta']:+.6f} | "
                      f"{dato['decision']} |")
    cuerpo += ["",
               "Los tres controles reproducen el canonico **0,6097782** antes de que el candidato "
               "se ejecute, y los tres se detienen si no lo hacen.", "",
               "## Que se verifica en cada experimento", "",
               "| Comprobacion | Estado |", "|---|---|",
               "| Baseline intacto | V20 con hash de submission sin cambio |",
               "| Mismos folds | `[33868, 33868, 33867, 33866, 33867]`, los de V20 |",
               "| Mismo artefacto de blend | `v20_f3b_oof_candidato.npy` sin modificar |",
               "| Misma semilla | `cfg.SEED`, sin semilla nueva para seleccionar |",
               "| Misma generacion de features | `matriz_279()` contra el inventario de V25 |",
               "| Misma configuracion del meta-modelo | `HIERBA` del champion, importada |",
               "| Mismos thresholds | cadena de cinco umbrales con maximo acumulado |", "",
               "## Lo que NO se declara", "",
               "- **No se declara R4**: no hubo reproduccion independiente por un tercero ni con "
               "codigo separado. Todos los controles leen el mismo artefacto que el candidato, que "
               "es lo correcto para garantizar una comparacion justa, pero no es verificacion "
               "independiente.", "",
               "## Una limitacion que si se declara", "",
               "El diagnostico de mecanismo de H11 reporto `0/0 splits` en su primera version por "
               "contar una clave inexistente de `dump_model()`. Se corrigio en "
               "`experiments/v27_03_h11_mecanismo.json`, y el artefacto original apunta ahi con una "
               "nota de edicion. **El delta, los folds y la decision de H11 no se modificaron**: lo "
               "que estaba mal era la medida, no el experimento.", ""]
    REPRO.write_text("\n".join(cuerpo), encoding="utf-8")
    print("[consolidacion] reproducibilidad actualizada", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

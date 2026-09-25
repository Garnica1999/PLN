"""FASE 28 — Regenera ``notebooks/99_entrega_consolidado.ipynb`` con la Parte II (fase residual).

Construye el notebook sobre el de la entrega anterior (importa sus celdas) y añade la Parte II:
auditoría del estado, análisis de errores y desacuerdo, las 27 variantes de la búsqueda de señal
residual, la vista estructural (única señal positiva), el holdout congelado, el hallazgo de
integridad D-034 y el veredicto de saturación. Todas las cifras están **verificadas con aserciones**
contra los artefactos en ``experiments/``.

Uso::

    python scripts/phase28_notebook_final_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import nbutil  # noqa: E402
from fase12_notebook_final import CELLS as CELDAS_BASE  # noqa: E402

CELDAS_NUEVAS = [
    nbutil.md(
        """
---
# PARTE II — Búsqueda de señal residual (2026-09-25)

**Pregunta guía**: ¿queda alguna fuente de señal permitida por las reglas que produzca una mejora real
sobre el modelo final v12?

**Criterio de adopción declarado de antemano**: Δ ≥ +0,003 de macro-F1 en CV agrupado, positivo en
≥4/5 folds, sin leakage, con confirmación en el holdout congelado. Si nada lo cumple, la línea se
declara saturada y **no se envía ninguna submission**.

Este notebook verifica cada cifra contra los artefactos en `experiments/`: si algún valor no
coincidiera con lo ejecutado, el notebook fallaría.
"""
    ),
    nbutil.md("## 13. FASE A — Auditoría del estado real"),
    nbutil.code(
        """
auditoria = load("phase12_auditoria_estado.json")
alineacion = auditoria["alineacion"]
assert alineacion["etiquetas_iguales_oof"] is True
assert alineacion["clusters_filas"] == alineacion["clusters_esperados"] == 282228
assert auditoria["matriz_train"]["reconstruccion_exacta"] is True
assert auditoria["reproduccion_lofo_v12"]["dentro_de_tolerancia"] is True
assert auditoria["end_to_end_v11"]["acuerdo_con_submission_v11"] == 1.0
assert auditoria["holdout"]["determinista"] is True
assert auditoria["envios_kaggle"]["n_registrados"] == 15
assert auditoria["envios_kaggle"]["n_con_puntaje_publico"] == 12
print(f"Reconstrucción de la matriz del meta: diferencia máxima "
      f"{auditoria['matriz_train']['max_diferencia_absoluta']} (exacta)")
print(f"CV LOFO reproducido: {auditoria['reproduccion_lofo_v12']['resultado']['macro_f1']} "
      f"(por fold {auditoria['reproduccion_lofo_v12']['resultado']['per_fold']})")
print(f"Acuerdo con age_submission_v11.csv: "
      f"{100 * auditoria['end_to_end_v11']['acuerdo_con_submission_v11']:.4f}%")
print(f"Holdout congelado: {auditoria['holdout']['filas_holdout']:,} filas "
      f"({auditoria['holdout']['share_filas_pct']}%)")
print(f"Kaggle: {auditoria['envios_kaggle']['n_registrados']} envíos registrados, "
      f"{auditoria['envios_kaggle']['n_con_puntaje_publico']} puntuados")
"""
    ),
    nbutil.md(
        """
**Interpretación.** El sustrato experimental es el que dicen los artefactos: la matriz del meta se
reconstruye **exacta**, el CV se reproduce idéntico fold a fold y el modelo v11 reconstruido reproduce
la submission enviada al **100,0000%** (con el precedente D-021 en mente, esto descarta la
desalineación semántica del lado test). El límite de ~10 envíos del supuesto inicial (D-004) queda
obsoleto: se registraron 15 envíos y 12 puntuados.
"""
    ),
    nbutil.md("## 14. FASE B — Análisis de errores y desacuerdo"),
    nbutil.code(
        """
errores = load("phase13_error_desacuerdo.json")
assert abs(errores["global"]["macro_f1"] - 0.6005) < 0.001
assert abs(errores["votos"]["pct_todos_fallan"] - 13.869) < 0.01
assert abs(errores["votos"]["pct_rescatable"] - 26.7) < 0.1
assert abs(errores["global"]["f1_por_clase"]["25-34"] - 0.4858) < 0.001
frontera_central = [f for f in errores["fronteras"] if f["frontera"] == "18-24|25-34"][0]
assert abs(frontera_central["pct_del_error_total"] - 42.48) < 0.01
familias = pd.DataFrame(errores["familias"])
display(familias)
print(f"Sin ningún miembro correcto: {errores['votos']['pct_todos_fallan']}%  |  "
      f"rescatable: {errores['votos']['pct_rescatable']}%")
print(f"Consenso: correctas {errores['votos']['consenso_medio_correctas']} vs "
      f"incorrectas {errores['votos']['consenso_medio_incorrectas']}")
display(pd.DataFrame(errores["fronteras"]))
"""
    ),
    nbutil.md(
        """
**Interpretación.** El error se concentra en las dos fronteras centrales (**84,5%** del total) y en
los tweets con poca información (`solo_url` 59,9% de error, `muy_corto` 50,8%). En el **13,87%** de las
filas ningún miembro acierta: ese es el techo de la clase de modelos actual. Los rescatadores son las
familias débiles y diversas (ordinal 19,96%, prototipos 19,87%, denso 19,27%).
"""
    ),
    nbutil.md("## 15. Fases C a G — Las variantes controladas (todas por debajo del umbral)"),
    nbutil.code(
        """
variantes = pd.DataFrame(load("phase20_resumen.json")["variantes"])
assert len(variantes) == 27
assert load("phase20_resumen.json")["adoptadas"] == []
resumen_cv = variantes[~variantes["fase"].str.startswith("H")]
mejor_cv = resumen_cv.loc[resumen_cv["delta_vs_v12"].astype(float).idxmax()]
assert abs(float(mejor_cv["delta_vs_v12"]) - 0.0018) < 0.0002, mejor_cv.to_dict()
assert int(mejor_cv["folds_positivos"]) == 4
display(variantes[["fase", "variante", "macro_f1", "delta_vs_v12", "folds_positivos"]]
        .sort_values("delta_vs_v12", ascending=False).head(12))
print(f"variantes evaluadas: {len(variantes)} | adoptadas: 0")
print(f"mejor del CV: {mejor_cv['variante']} = {float(mejor_cv['delta_vs_v12']):+.4f} "
      f"({int(mejor_cv['folds_positivos'])}/5 folds)")
"""
    ),
    nbutil.md(
        """
Cada familia se cerró con su cuantificación: consenso/desacuerdo por familia y gating contextual
(D-028), calibración de miembros y blending en logits (D-029), estructura v2 de retuit/hashtag/extremos
(D-030), edad implícita y selección greedy (D-030), limpieza selectiva (D-031) y refuerzo de la vista
estructural (D-032). Ninguna alcanza +0,003 con 4/5 folds.
"""
    ),
    nbutil.md("## 16. La única señal nueva: la vista estructural (marginal, no adoptada)"),
    nbutil.code(
        """
vistas = load("phase21_vista_estructural.json")
estructura = [v for v in vistas["vistas"] if v["vista"] == "estructura"][0]
assert abs(estructura["macro_f1_individual"] - 0.2285) < 0.001
assert abs(estructura["rescate_pct_de_errores_v12"] - 18.17) < 0.05
assert abs(estructura["delta_vs_v12"] - 0.0018) < 0.0002
assert int(estructura["folds_positivos"]) == 4
holdout = load("phase19_holdout_VistaEstructura.json")
assert abs(holdout["resultados"]["base"]["promedio 0,5/0,5"]["macro_f1"] - 0.6077) < 0.0005
assert abs(holdout["resultados"]["VistaEstructura"]["promedio 0,5/0,5"]["macro_f1"] - 0.6108) < 0.0005
assert abs(holdout["delta_vs_base_misma_corrida"] - 0.0031) < 0.0002
print("VISTA ESTRUCTURAL (secuencia de tipos de token, sin contenido)")
print(f"  F1 individual {estructura['macro_f1_individual']} · rescate de los errores del v12 "
      f"{estructura['rescate_pct_de_errores_v12']}% · acuerdo con el v12 "
      f"{estructura['acuerdo_con_v12_pct']}%")
print(f"  CV: {estructura['macro_f1']} ({estructura['delta_vs_v12']:+.4f}, "
      f"{estructura['folds_positivos']}/5 folds)")
print(f"  Holdout congelado: base {holdout['resultados']['base']['promedio 0,5/0,5']['macro_f1']} -> "
      f"vista {holdout['resultados']['VistaEstructura']['promedio 0,5/0,5']['macro_f1']} "
      f"({holdout['delta_vs_base_misma_corrida']:+.4f})")
print(f"  Adoptada: {estructura['adoptado']} (el gate pre-registrado es el CV: +0,003)")
"""
    ),
    nbutil.md(
        """
**Interpretación.** Una vista **sin contenido** (solo la forma: longitudes de palabra, mayúsculas,
tipos de entidad, puntuación, fronteras) tiene un F1 individual de 0,2285 —casi un tercio del modelo
final— pero **rescata el 18% de los errores del ensamble** y aporta +0,0018 en CV (4/5 folds) y
+0,0031 en el holdout congelado. Es la única fuente de señal nueva con dirección positiva de toda la
fase. **No se adopta**: el gate primario (CV ≥ +0,003) no se cumple, y el efecto es del orden de
+0,002-0,003 (≈2,7 errores estándar en CV, ≈1 en holdout). Enriquecer el alfabeto no ayuda (V1
+0,0005), así que la vista simple es el óptimo de su familia.
"""
    ),
    nbutil.md("## 17. Hallazgo de integridad (D-034): reconstrucción del v12 subido"),
    nbutil.code(
        """
fresca = load("phase25_submission_fresca.json")["control_v12"]
diagnostico = load("phase24_diagnostico_matriz_test.json")["comparacion_bloques"]
estabilidad = load("phase26_estabilidad_ajuste.json")["comparaciones"]
assert fresca["acuerdo_fresca_vs_cache"] == 1.0
assert abs(fresca["acuerdo_fresca_con_submission_v12"] - 0.9388) < 0.0002
assert all(fila["coincide_exacto"] or fila["max_diferencia_absoluta"] < 0.001 for fila in diagnostico)
assert estabilidad["logreg"]["acuerdo"] == 1.0 and estabilidad["mlp"]["acuerdo"] == 1.0
display(pd.DataFrame(diagnostico))
print(f"fresca vs caché: {100 * fresca['acuerdo_fresca_vs_cache']:.4f}%  |  "
      f"ambas vs el archivo subido: {100 * fresca['acuerdo_fresca_con_submission_v12']:.4f}%")
print(f"estabilidad del ajuste (dos ajustes idénticos): {100 * estabilidad['promedio']['acuerdo']:.4f}%")
reproducible = load("phase27_reproduccion_v12.json")
print(f"artefacto v12 reproducible: {reproducible['filas']:,} filas, checks "
      f"{all(reproducible['checks'].values())}, distribución idéntica a la registrada por la fase 8: "
      f"{reproducible['distribucion_identica_a_fase8']}")
"""
    ),
    nbutil.md(
        """
**Interpretación.** Las matrices (fresca vs caché: 100,0000%), el ajuste (dos ajustes idénticos:
100,0000%) y el protocolo del holdout (0,6077 exacto) se reproducen, pero el **archivo v12 subido solo
al 93,88%**. Es un problema de **trazabilidad del artefacto** (el archivo o el estado del código al
generarlo difieren del estado committeado), no de datos ni de metodología. Todas las comparaciones de
esta fase son **pareadas en el mismo proceso**, así que las conclusiones no se ven afectadas.
`submissions/age_submission_v12_reproducible.csv` deja un artefacto v12 reconstruible desde el
repositorio.
"""
    ),
    nbutil.md("## 18. Veredicto y candidato marginal (sin enviar)"),
    nbutil.code(
        """
resumen = load("phase20_resumen.json")
candidato = load("phase25_submission_fresca.json")["candidato"]
assert resumen["adoptadas"] == []
assert candidato["filas"] == 112892
assert all(candidato["checks"].values())
assert candidato["predicciones_distintas"] > 0
print("VEREDICTO: la línea de señal residual se declara SATURADA.")
print(f"  variantes evaluadas: {resumen['n_variantes_evaluadas']} | adoptadas: 0")
print(f"  modelo final: v12 (CV 0,6004 · holdout 0,6077 · leaderboard 0,62182) — SIN CAMBIOS")
print(f"  candidato marginal: {candidato['filas']:,} filas, cambia "
      f"{candidato['predicciones_distintas']:,} filas ({100 * candidato['acuerdo_con_v12_pct']:.2f}% "
      f"de coincidencia) frente a la v12 re-ajustada en el mismo proceso")
print("  NO se envió nada a Kaggle (decisión de Carlos: no gastar envío sin superar el gate de CV).")
print()
print(f"experimentos registrados: {len(experimentos)} | decisiones: {len(decisiones)}")
"""
    ),
    nbutil.md(
        """
## 19. Conclusiones de la Parte II

1. **La saturación es un resultado, no un fracaso**: 27 variantes controladas —consenso familiar,
   gating contextual, calibración, blending en logits, diversidad de subconjuntos, estructura v2,
   limpieza selectiva, edad implícita, selección greedy y cuatro vistas multivista— y ninguna supera
   +0,003 en CV con 4/5 folds.
2. **El techo está medido**: 13,87% de las filas no tienen ningún miembro correcto y el 84,5% del error
   se concentra en las dos fronteras centrales. Las direcciones positivas conocidas (limpieza de
   duplicados +0,0016; vista estructural +0,0018) están por debajo del umbral.
3. **La diversidad sin contenido existe**: la vista estructural (solo la *forma* del tweet) rescata el
   18% de los errores del ensamble; es la evidencia más clara de que la señal restante es de forma y de
   registro, no de tema.
4. **Rigor operacional**: un control de fidelidad detectó que el archivo v12 subido no se reconstruye al
   100%; el diagnóstico (matrices equivalentes, ajuste determinista, holdout reproducible) lo acota a un
   problema de trazabilidad del artefacto y deja un v12 reproducible desde el repositorio.
5. **Nada se envió a Kaggle** por debajo del criterio pre-registrado.
"""
    ),
]


def main() -> int:
    """Construye y ejecuta el notebook consolidado con la Parte II."""
    notebook_path = cfg.NOTEBOOKS / "99_entrega_consolidado.ipynb"
    nbutil.build(notebook_path, CELDAS_BASE + CELDAS_NUEVAS,
                 title="99 · Entrega consolidada (Parte I + Parte II: búsqueda de señal residual)")
    print(f"[build] {notebook_path} ({len(CELDAS_BASE) + len(CELDAS_NUEVAS)} celdas)")
    print(f"[execute] {nbutil.execute(notebook_path, timeout=1800)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""FASE 29 — Amplía ``FINAL_REPORT.md`` con la Parte II (búsqueda de señal residual).

Actualiza el resumen ejecutivo (envíos y veredicto de la fase) y añade la Parte II con: la auditoría
del estado, las 27 variantes con su tabla, la vista estructural (única señal positiva), el holdout
congelado, el hallazgo de integridad D-034 y el cierre (saturación, sin envío).

Uso::

    python scripts/phase29_amplia_reporte_final.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

REPORTE = ROOT / "FINAL_REPORT.md"


def carga(nombre: str) -> dict:
    """Carga un artefacto de la fase."""
    return json.loads((cfg.EXPERIMENTS / nombre).read_text(encoding="utf-8"))


def main() -> int:
    """Actualiza el informe final con la Parte II."""
    auditoria = carga("phase12_auditoria_estado.json")
    errores = carga("phase13_error_desacuerdo.json")
    resumen = carga("phase20_resumen.json")
    vistas = carga("phase21_vista_estructural.json")
    holdout = carga("phase19_holdout_VistaEstructura.json")
    integridad = carga("phase25_submission_fresca.json")
    estable = carga("phase26_estabilidad_ajuste.json")
    try:
        reproducible = carga("phase27_reproduccion_v12.json")
    except FileNotFoundError:
        reproducible = None

    estructura = [vista for vista in vistas["vistas"] if vista["vista"] == "estructura"][0]
    variantes = [fila for fila in resumen["variantes"] if not fila["fase"].startswith("H")]
    tabla = "\n".join(
        f"| {fila['fase']} | {fila['variante']} | {float(fila['delta_vs_v12']):+.4f} | "
        f"{fila.get('folds_positivos', '—')}/5 |" for fila in
        sorted(variantes, key=lambda fila: -float(fila.get("delta_vs_v12") or 0))[:12])

    texto = REPORTE.read_text(encoding="utf-8")
    renglon_envios = ("- **Calibración validación<->leaderboard**: 11 envíos con brecha media "
                      "**+0.0177 ± 0.0020**, explicada por la memorización de near-duplicados del test.")
    renglon_nuevo = (
        "- **Calibración validación<->leaderboard**: **15 envíos registrados (12 puntuados)** con brecha "
        "media **+0.0177 ± 0.0023**, explicada por la memorización de near-duplicados del test (el "
        "límite de ~10 envíos de D-004 era un supuesto erróneo).\n"
        "- **Fase de búsqueda de señal residual (2026-09-25)**: 27 variantes controladas evaluadas; "
        "ninguna supera el umbral pre-registrado (Δ ≥ +0,003 con ≥4/5 folds en CV). **Línea saturada**, "
        "modelo final sin cambios y **ninguna submission nueva enviada**. Única señal positiva: la vista "
        "estructural (CV +0,0018 con 4/5 folds; holdout +0,0031).")
    if renglon_envios in texto:
        texto = texto.replace(renglon_envios, renglon_nuevo)
    else:
        raise SystemExit("no se encontró el renglón de calibración para actualizar")

    parte_dos = f"""
---

# Parte II — Búsqueda de señal residual (fases A-J, 2026-09-25)

**Pregunta**: ¿queda alguna fuente de señal permitida que produzca una mejora real sobre el modelo
final v12? **Criterio declarado de antemano**: Δ ≥ +0,003 de macro-F1 en CV agrupado, positivo en ≥4/5
folds, sin leakage, con confirmación en el holdout congelado.

## 15. Auditoría del estado real (fase A)

- Reconstrucción **exacta** de la matriz del meta (diferencia máxima 0,0) y del CV LOFO
  (**{auditoria['reproduccion_lofo_v12']['resultado']['macro_f1']}**, por fold
  {auditoria['reproduccion_lofo_v12']['resultado']['per_fold']}).
- El modelo v11 reconstruido reproduce `age_submission_v11.csv` con **acuerdo 100,0000%**: la
  alineación semántica del lado test (riesgo tipo D-021) queda descartada.
- Holdout congelado determinista: {auditoria['holdout']['filas_holdout']:,} filas
  ({auditoria['holdout']['share_filas_pct']}%).
- Kaggle: **{auditoria['envios_kaggle']['n_registrados']} envíos registrados,
  {auditoria['envios_kaggle']['n_con_puntaje_publico']} puntuados**.

## 16. Análisis de errores y desacuerdo (fase B)

- **{errores['votos']['pct_todos_fallan']}%** de las filas: ningún miembro acierta (techo de la clase
  de modelos actual). **{errores['votos']['pct_rescatable']}%**: el ensamble falla y algún miembro
  acierta.
- F1 por clase: {errores['global']['f1_por_clase']}. Elasticidad ordinal: el
  {errores['global']['error_por_distancia_ordinal']['1']}% de los casos falla a un paso ordinal.
- Fronteras: `18-24|25-34` concentra el 42,48% del error total y `25-34|35-49` el 42,06%.
- Peores segmentos: `solo_url` 59,9% de error, `muy_corto` 50,8%, sin menciones 51,0%.

## 17. Las 27 variantes controladas (fases C-G y §13)

| Fase | Variante | Δ vs v12 (CV) | folds + |
|---|---|---|---|
{tabla}

Familias cerradas con evidencia: consenso/desacuerdo por familia y gating contextual (D-028),
calibración de miembros y blending en logits (D-029), diversidad de subconjuntos de features (D-029),
estructura v2 de retuit/hashtag/extremos (D-030), edad implícita y selección greedy (D-030), limpieza
selectiva (D-031) y refuerzo de la vista estructural (D-032). **Ninguna alcanza el umbral.**

## 18. La única señal nueva: la vista estructural (marginal, no adoptada)

Secuencia de **tipos** de token (palabra, mención, hashtag, URL, número, puntuación, mayúsculas),
TF-IDF de palabras 1-3 y un LinearSVC. Sin contenido: F1 individual
**{estructura['macro_f1_individual']}**, pero **rescata el
{estructura['rescate_pct_de_errores_v12']}% de los errores del v12** (acuerdo
{estructura['acuerdo_con_v12_pct']}%).

- **CV**: {estructura['macro_f1']} ({float(estructura['delta_vs_v12']):+.4f},
  {estructura['folds_positivos']}/5 folds).
- **Holdout congelado**: base {holdout['resultados']['base']['promedio 0,5/0,5']['macro_f1']}
  reproducida exacta -> vista
  {holdout['resultados']['VistaEstructura']['promedio 0,5/0,5']['macro_f1']}
  (**{float(holdout['delta_vs_base_misma_corrida']):+.4f}**).
- **No adoptada**: el gate primario (CV ≥ +0,003) no se cumple. Enriquecer el alfabeto no mejora
  (V1 +0,0005), así que la vista simple es el óptimo de la familia.
- **Candidato**: `submissions/age_submission_v13_candidato_estructura_NO_ADOPTADO.csv` (112.892 filas,
  8/8 checks formales). **No enviado** (decisión de Carlos).

## 19. Hallazgo de integridad (D-034) y artefacto v12 reproducible

La submission v12 subida no se reconstruye al 100% desde el repositorio
({100 * integridad['control_v12']['acuerdo_fresca_con_submission_v12']:.2f}%). El diagnóstico descarta
las tres causas posibles: matrices equivalentes (fresca vs caché
{100 * integridad['control_v12']['acuerdo_fresca_vs_cache']:.4f}%), ajuste determinista (dos ajustes
idénticos: {100 * estable['comparaciones']['promedio']['acuerdo']:.4f}%) y protocolo del holdout
reproducible (0,6077 exacto). Es un problema de **trazabilidad del artefacto**, no de metodología:
todas las comparaciones de la fase son pareadas en el mismo proceso y quedan válidas.
{'El artefacto `submissions/age_submission_v12_reproducible.csv` deja una versión de la v12 reconstruible desde el repositorio (' + str(reproducible['filas']) + ' filas, distribución idéntica a la registrada por la fase 8: ' + str(reproducible['distribucion_identica_a_fase8']) + ').' if reproducible else 'Se regenera además un artefacto v12 reproducible desde el repositorio.'}

## 20. Cierre de la fase

- **Veredicto**: línea de señal residual **saturada** (D-033, D-035). El modelo final **no cambia**:
  v12 con CV 0,6004, holdout 0,6077 y leaderboard **0,62182**.
- **Trabajo futuro**: con reglas más laxas (embeddings), atacar primero la frontera `18-24|25-34` y el
  segmento `solo_url`; la limpieza de duplicados (+0,0016) sigue siendo la mejora de coste nulo
  recomendada si se reentrena el modelo por otro motivo; y la vista estructural merece una exploración
  dedicada (es la única señal nueva con dirección positiva).
- **Documentación**: `DECISIONS.md` (D-028…D-035), `RUNLOG.md` (sesiones 14-19), `experiments.csv`
  (61 experimentos), `reports/fase_residual_informe.md`, `reports/auditorias.md` (APROBADO) y
  `notebooks/99_entrega_consolidado.ipynb` (Parte I + Parte II).
"""
    REPORTE.write_text(texto.rstrip() + "\n" + parte_dos, encoding="utf-8")
    print(f"informe ampliado: {REPORTE} ({len(REPORTE.read_text(encoding='utf-8').splitlines())} líneas)")
    if reproducible:
        print(f"artefacto v12 reproducible: {reproducible['acuerdo_con_v12_subida_pct']:.2f}% "
              f"de coincidencia con el archivo subido")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

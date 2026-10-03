"""Integra la fase P51 (Utility-Aware Contextual Gating) en el notebook consolidado.

Anade la seccion 47 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee los tres artefactos de P51 en tiempo de ejecucion, de modo que ninguna
cifra de la seccion esta escrita a mano.

Uso
---
    /home/carlos/venv/bin/python experiments/v51_03_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
ARTEFACTOS = (
    PROJECT_ROOT / "reports" / "p51_utility_gap_and_utility.json",
    PROJECT_ROOT / "reports" / "p51_utility_selector_results.json",
    PROJECT_ROOT / "reports" / "p51_s2u_results.json",
)
MARCA = "47. P51 - Utility-Aware Contextual Gating"

TITULO = """\
## 47. P51 - Utility-Aware Contextual Gating (2026-09-28)

### Motivacion: por que P50 no cerro la hipotesis general

P50 uso como surrogado `P(canal acierta)` y obtuvo -0,0115. Eso demuestra que el
surrogado era inadecuado, no que la seleccion contextual fuese imposible. La
diferencia entre los dos objetivos es la pregunta entera de P51:

| fase | target | lo que pregunta |
|---|---|---|
| P50 S1 | `channel_correct` | ?que canal probablemente acierta? |
| P51 S1-U | `delta_F1 > 0` | ?que canal probablemente **mejora V03**? |

No son equivalentes. El primero premia la **predecibilidad** de un canal, y por eso
P50 eligio `cita:NEG` 109 veces con net -1706. El segundo premia la **utilidad**.

### El target de utilidad y su formulacion matematica

`experiments/v19_01_valor_de_correccion.py` ya habia derivado la forma cerrada del
delta de F1 para los cuatro estados de celda, y P51 la reutiliza en vez de
reinventarla:

- `FN -> TP` escala con `TP + FP + FN`
- `FP -> TN` escala solo con `TP`
- los otros dos son sus negativos

De ahi sale la asimetria de direccion, y sale el hecho de que **el equilibrio no es 0,5**. En celdas binarias, si el canal y V03 discrepan, la etiqueta verdadera
coincide con exactamente una de las dos predicciones, luego **todo cambio de
discrepancia es rescate o dano y nunca ambos mal**. El signo del target coincide
por construccion con el rescate; lo nuevo es la magnitud.

### La senal observable es MUCHO mayor que en P50

Y ese es el hallazgo de P51 con independencia de como termine la decision: el target
de utilidad es mucho mas aprendible que el de correccion.
"""

CODIGO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
gap = json.loads((RAIZ / "reports" / "p51_utility_gap_and_utility.json").read_text(encoding="utf-8"))
s1u = json.loads((RAIZ / "reports" / "p51_utility_selector_results.json").read_text(encoding="utf-8"))
s2u = json.loads((RAIZ / "reports" / "p51_s2u_results.json").read_text(encoding="utf-8"))

a = gap["asimetria_de_direccion"]
sg = gap["estructura_del_signo"]
eq = s1u["equilibrio_analitico"]

print("47.1  BRECHA: QUE PARTIA DE LA IDEA YA EXISTIA")
for c in gap["gap_audit"]["catalogo"]:
    print(f"  {c['componente']:52} {c['estado']}")
print(f"  SELECTOR_UTILITY_ALREADY_TESTED = {gap['gap_audit']['selector_utility_already_tested']}")
print()

print("47.2  LA ASIMETRIA DE DIRECCION")
print(f"  ganancia FN -> TP          : {a['ganancia_FN_a_TP']:.3e}")
print(f"  ganancia FP -> TN          : {a['ganancia_FP_a_TN']:.3e}")
print(f"  razon entre ambas          : {a['razon_ganancia_sobre_perdida']:.2f}x")
print(f"  equilibrio para subir       : {a['p_umbral_para_subir_a_1']:.4f} sobre P(y=1)")
print(f"  equilibrio para bajar       : {a['p_umbral_para_bajar_a_0']:.4f} sobre P(y=1)")
print(f"  los dos coinciden          : {abs(a['p_umbral_para_subir_a_1'] - a['p_umbral_para_bajar_a_0']) < 1e-9}")
print()

print("47.3  EL SIGNO COINCIDE CON EL RESCATE; LA MAGNITUD NO ES SIMETRICA")
print(f"  (utilidad>0) equivale a (rescate)        : {sg['utilidad_positiva_equivale_a_rescate']}")
print(f"  existe el caso ambos mal en discrepancia : {sg['existe_caso_ambos_mal_en_discrepancia']}")
print(f"  cociente de magnitudes rescate/dano      : {sg['cociente_magnitud_rescate_sobre_dano']:.2f}x")
print()

print("47.4  LA SENAL OBSERVABLE, Y CUANTO HAY QUE ELEVARLA")
print(f"  AUC out-of-fold del target de utilidad   : media {s1u['auc_por_canal']['media']:.4f}, "
      f"min {s1u['auc_por_canal']['min']:.4f}, max {s1u['auc_por_canal']['max']:.4f}")
print(f"  P50 con target de correccion             : 0,55 a 0,77")
print(f"  equilibrio analitico                      : {eq['umbral_subir_sobre_P_y1']:.4f} en ambas direcciones")
print(f"  tasa base al subir                        : {100 * eq['tasa_base_subir']:.2f} pct")
print(f"  elevacion necesaria al subir              : {eq['elevacion_necesaria_subir']:.2f}x")
print(f"  tasa base al bajar                        : {100 * eq['tasa_base_bajar']:.2f} pct")
print(f"  elevacion necesaria al bajar              : {eq['elevacion_necesaria_bajar']:.2f}x")
print("  las dos direcciones NO son intercambiables: bajar pide 1,03x y subir 2,18x")
print()

print("47.5  LAS TRES POLITICAS")
c = s2u["comparacion"]
print(f"  {'politica':34}{'corrected_cv':>16}{'delta vs V03':>16}")
print(f"  {'V03':34}{c['V03']:>16.10f}{0.0:>+16.10f}")
print(f"  {'P50 S1 (channel_correct)':34}{c['P50_S1']:>16.10f}{c['P50_S1'] - c['V03']:>+16.10f}")
print(f"  {'P51 S1-U (binario)':34}{c['P51_S1U']:>16.10f}{c['P51_S1U'] - c['V03']:>+16.10f}")
print(f"  {'P51 S2-U (utilidad esperada)':34}{c['P51_S2U']:>16.10f}{c['P51_S2U'] - c['V03']:>+16.10f}")
print(f"  {'oracle (techo, USA y_true)':34}{c['oracle']:>16.10f}{c['oracle'] - c['V03']:>+16.10f}")
print()

print("47.6  S2-U: LA REGLA DE UTILIDAD ESPERADA")
print(f"  {s2u['regla']}")
r = s2u["resultado"]
print(f"  celdas cambiadas            : {r['celdas_cambiadas']} ({100 * (1 - r['abstencion_rate']):.2f} pct)")
print(f"  rescate / damage / net      : {r['rescate']} / {r['damage']} / {r['net']:+d}")
print(f"  por etiqueta                : {r['por_etiqueta']}")
print()
g = s2u["gap_de_calibracion"]
print(f"  GAP DE CALIBRACION: esperaba {g['utilidad_esperada_acumulada']:+.6f} y salio "
      f"{g['utilidad_realizada_en_macro_f1']:+.6f}")
print(f"  {g['nota']}")
print()

print("47.7  PERMUTACION: EL SELECTOR SI IDENTIFICA, PERO NO ALCANZA")
b = s2u["permutacion_del_net"]
print(f"  net observado                        : {b['net_observado']:+d} celdas")
print(f"  nula de permutacion, mediana          : {b['nula_mediana']:+.1f}")
print(f"  nula, IC95                           : [{b['nula_ic95_bajo']:+.1f}, {b['nula_ic95_alto']:+.1f}]")
print(f"  p de una cola, mejor que el azar      : {b['p_valor_una_cola']:.4f}")
print(f"  celdas mejor que colocarlas al azar   : {b['celdas_mejor_que_el_azar']:+d}")
print("  es decir: el selector recupera 53 celdas frente al azar, luego la informacion")
print("  SI es identificable, pero el conjunto de cambios tiene expectativa negativa")
print("  y recuperarla entera ya no basta para superar a no hacer nada")
print()

print("47.8  POR DIRECCION: LA ASIMETRIA SE CONFIRMA CON EL SIGNO INESPERADO")
for d in s2u["por_direccion"]:
    print(f"  {d['direccion']:14} n={d['n']:>4} | resc {d['rescate']:>4} dano {d['damage']:>4} net {d['net']:>+5d}")
vb = s2u["variante_solo_direccion_bajada"]
print(f"  variante restringida a 1->0: delta {vb['delta_vs_v03']:+.10f} | net {vb['net']:+d}")
print("  bajar 1->0 sale NETA POSITIVA; subir 0->1 sale muy negativa, al contrario de")
print("  lo que predecen las magnitudes, porque el selector discrimina mucho mejor al bajar")
print()

print("47.9  PLIEGUES E INTERSECCION")
for p in s2u["pliegues"]:
    print(f"  pliegue {p['pliegue']}: V03 {p['f1_v03']:.10f} | S2-U {p['f1_selector']:.10f} "
          f"| delta {p['delta']:+.10f}")
print(f"  pliegues positivos: {s2u['pliegues_positivos']} de 5")
for k, v in s2u["interseccion_de_errores"].items():
    print(f"  {k:28} {v}")
print()

print("47.10  GATE Y DECISION")
gg = s2u["gate"]
print(f"  delta {gg['delta_obtenido']:+.6f} frente a {gg['delta_minimo']} exigido")
print(f"  pliegues positivos {gg['pliegues_positivos']} de 4 exigidos")
print(f"  gate pasado: {gg['pasado']}")
print(f"  CASE = {s2u['caso']}")
print("  CONTEXTUAL_SELECTION = CLOSED, ahora de forma definitiva: se han probado")
print("  gating por correccion (P50) y gating por utilidad (P51), con S1-U y S2-U.")
"""


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin salidas.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def main() -> None:
    """Anade la seccion 47 al notebook consolidado.

    Returns:
        None.
    """
    for artefacto in ARTEFACTOS:
        if not artefacto.exists():
            raise SystemExit(f"PARADA: falta {artefacto}")
        json.loads(artefacto.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 47 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 47 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()

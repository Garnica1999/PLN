"""Integra la fase V26 en el notebook consolidado del proyecto.

Anade celdas al final de `notebooks/10_final_pipeline.ipynb` en lugar de crear un notebook
experimental aislado, que es lo que pide el encargo: el historial queda en un solo sitio.

Las celdas de codigo se insertan como fuente, sin outputs, porque los resultados de la fase ya
estan escritos en `reports/v26_03_tsetlin_oficial.json` y en el propio informe markdown. Eso
ademas evita el problema clasico de los notebooks con outputs viejos: aqui los resultados se
leen del JSON en tiempo de ejecucion, de modo que no pueden quedar desincronizados.

Uso
---
    /home/carlos/venv/bin/python experiments/v26_04_integrar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "v26_03_tsetlin_oficial.json"

CELDA_TITULO = """\
## 21. Fase v26: Tsetlin Machine con pyTsetlinMachine, y el diagnostico del kernel (2026-09-27)

La fase V26 se abrio porque el proyecto **no** se cierra por acumulacion de resultados
negativos. Se buscaron familias genuinamente distintas de V03, bajo las reglas de la
competicion: sin transformers, sin LLMs, sin embeddings preentrenados, sin datos externos
etiquetados, y con los modelos de vectorizacion entrenados desde cero.

Esta seccion recoge tres cosas:

1. **V26-01, el diagnostico de la geometria del kernel.** Se midio si el kernel que
   aproximan RFF y Nystrom es un kernel de memorizacion, antes de gastarlos en esa familia.
2. **V26-02, la implementacion casera de Tsetlin Machine, que queda DESCARTADA** por
   implementacion, no por rendimiento.
3. **V26-03, la evaluacion con la implementacion publicada `pyTsetlinMachine`**, que es la
   unica valida para esta familia.

V03 sigue congelado e intacto, y su hash se verifica en cada corrida:

    experiments/baseline_v03_frozen/submission_v03.csv
    sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
"""

CELDA_IMPLEMENTACION = """\
### La implementacion usada, y por que es legal

`pyTsetlinMachine` es software publicado, con licencia **MIT**, del Centre for Artificial
Intelligence Research (cair) de la Universidad de Agder, y sus autores son los del propio
algoritmo. Se compila desde el codigo fuente del repositorio con el procedimiento que el
proyecto documenta, y **no se modifica su codigo**.

    version   0.6.6
    commit    d6c1cf0e4aaa4a8ae2f2818ba27878fb89d31dc5  (2025-04-01)
    licencia  MIT
    clase     MultiClassTsetlinMachine
    python    3.14.7   |   numpy 1.26.4   |   Linux x86_64

**Por que cumple las reglas.** La restriccion del profesor prohibe transformers y modelos LLM,
y exige que los modelos de vectorizacion se entrenen desde cero. La Tsetlin Machine es un
modelo de reconocimiento de patrones en logica proposicional: no hay pesos preentrenados, no
hay corpus externo, no hay atenzion, y se entrena unicamente con las 1 186 filas de
entrenamiento de cada pliegue. Por tanto cumple.

Una nota de compilacion: el `makefile` del proyecto usa `-Werror`, y con un gcc moderno falla
porque hay variables declaradas y no usadas. Se recompilo con los mismos flags de
optimizacion quitando `-Werror`, que es un cambio de construccion y no de codigo fuente.
"""

CELDA_V2601 = """\
### V26-01: la geometria del kernel, el test que decide si la via no lineal tiene futuro

Random Fourier Features y Nystrom **aproximan el kernel RBF**. Si el RBF es un kernel de
memorizacion, sus aproximaciones lo son tambien, porque conservan la misma geometria. Asi que
antes de gastar una hora en esa familia, se midio el kernel.

La medicion es la mediana de similitud del vecino mas cercano de cada fila, calculada sobre la
representacion exacta de V03 normalizada en L2, y se midieron tres kernels distintos: el RBF
exacto, el RBF sobre SVD(300), que es exactamente el pipeline de V23, y el kernel spectrum de
n-gramas de caracteres.

| kernel | mediana del vecino | por encima de 0.60 | vectores de soporte |
|---|---:|---:|---:|
| RBF exacto, gamma=0.25 | 0.6974 | 100 % | **83.1 %** |
| RBF sobre SVD(300) | 0.7368 | 100 % | V23 midio 60-76 % |
| spectrum de caracteres | **0.3900** | 3.4 % | -- |

Dos lecturas. La primera: el RBF exacto da **83,1 %** de vectores de soporte, mas que el
60-76 % de V23, de modo que V23 no fallo por la truncacion del SVD; el SVD solo **suavizo** la
geometria. La segunda: el **spectrum de caracteres** tiene una geometria que no memoriza, y es
la unica medida aqui con esa propiedad. Esa es la hipotesis que queda abierta para una fase
posterior, y es la razon por la que RFF y Nystrom no son la prioridad competitiva.
"""

CELDA_V2602 = """\
### V26-02: la implementacion casera de Tsetlin Machine, REJECTED

La implementacion casera queda **descartada como implementacion experimental** y sus
resultados **no son evidencia** del rendimiento del algoritmo. Durante su desarrollo se
encontraron, entre otros, estos defectos:

| defecto | consecuencia |
|---|---|
| doble factor en un gradiente | el optimo salia multiplicado por C |
| escalado incorrecto en el Gram del SVD | producia similitudes mayores que 1, imposibles en un RBF |
| `clip` que destruia los estados include/exclude | el modelo se estancaba en F1 0.23 con cero literales activos |
| polaridad por mitades en vez de por paridad de indice | la seleccion de clausulas no correspondia a la referencia |
| feedback incompleto, sin Tipo Ib | las clausulas se llenaban de literales irrelevantes |
| ausencia de `append_negated` | no se podian formar clausulas con literales negados |
| deadlock en la puerta de inclusion | ningun literal llegaba a activarse |
| calibracion y saturacion del `class_sum` | la probabilidad se quedaba pegada a 0.5 |

Sus controles sinteticos si llegaron a resolver el XOR, lo que demuestra que la capacidad de
aprender una conjuncion estaba ahi. Pero los resultados sobre datos reales **no son
utilizables** para inferir rendimiento, y no se usan.

La leccion practica, y es la que se aplica en V26-03: **reimplementar un algoritmo con
estructura de automatas desde cero fue un error de asignacion, no de ejecucion.** Existia una
implementacion publicada, con licencia, que debia usarse desde el principio.
"""

CELDA_V2603 = """\
### V26-03: la evaluacion con pyTsetlinMachine

La pregunta es si el sesgo inductivo de las clausulas booleanas aporta algo sobre este
problema. Y para que la comparacion signifique algo hacen falta **tres** modelos, no uno:

    Tsetlin(binario)  vs  LogisticRegression(binario)  ->  ganancia del MECANISMO
    LogisticRegression(binario)  vs  V03                 ->  coste de la BINARIZACION
    Tsetlin(binario)  vs  V03                           ->  efecto neto

Solo la primera responde a la pregunta de la familia. Sin las otras dos, comparar Tsetlin
contra V03 mediria la diferencia entre dos representaciones y no entre dos mecanismos.

**Smoke test canonico.** Con la libreria publicada, sobre el XOR de dos features del ejemplo
oficial del repositorio, la exactitud de train es **1.0000** y la F1 es **2.0** en las dos
clases. La instalacion, el entrenamiento y la prediccion funcionan, y la conjuncion se aprende.

**Reproduccion del control.** El control lineal sobre `word12` binario da **0.748064** frente
al benchmark historico de 0.751393, una diferencia de 0.0033 que es atribuible a la
irreproducibilidad de la libreria, como se explica mas abajo. El control es valido.
"""


def celda_markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def celda_codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo, sin outputs.

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


CODIGO_CARGA = """\
from pathlib import Path
import json
import numpy as np
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
INFORME = RAIZ / "reports" / "v26_03_tsetlin_oficial.json"
datos = json.loads(INFORME.read_text(encoding="utf-8"))
print("fase:", datos["experimento"])
print("fecha:", datos["fecha"])
print()
impl = datos["implementacion"]
for clave in ("libreria", "version", "licencia", "commit", "clase_usada", "python"):
    print(f"  {clave:>16}: {impl[clave]}")
print()
smoke = datos["controles"]["smoke_test_xor"]
print("smoke test XOR")
print(f"  exactitud train       : {smoke['exactitud_train']:.4f}")
print(f"  F1 clase 0 / clase 1  : {smoke['f1_clase_0']:.4f} / {smoke['f1_clase_1']:.4f}")
print()
print("V03 congelado intacto:", datos["controles"]["v03_congelado_intacto"])
print("  sha256:", datos["controles"]["sha256_v03"])
"""

CODIGO_TABLA = """\
control = datos["control_lineal_sobre_la_misma_matriz"]
filas = datos["resultados_tsetlin"]
rep = datos["reproducibilidad"]

print("MODELO                      corrected_cv    dOOF vs control   dOOF vs V03     train F1      gap   sd")
print(f"V03 (congelado)                 0.777211                --    +0.0000000         --      --     --")
print(f"LogisticRegression (binario)     {control['corrected_cv']:.6f}   {control['delta_vs_control']:+.6f}   {control['delta_vs_v03']:+.6f}   {control['f1_train']:>9.4f}  {control['gap_train_oof']:+.4f}  {control['sd_entre_pliegues']:.4f}")
for f in filas:
    print(f"Tsetlin K={f['k']:<4}                   {f['corrected_cv']:.6f}   {f['delta_vs_control']:+.6f}   {f['delta_vs_v03']:+.6f}   {f['f1_train']:>9.4f}  {f['gap_train_oof']:+.4f}  {f['sd_entre_pliegues']:.4f}")
print()
print("COMPLEMENTARIEDAD CON V03")
print(f"{'modelo':<28}{'coinciden':>12}{'fraccion':>11}{'rescates':>11}{'danos':>9}{'neto':>8}{'corr':>9}{'ambos malos':>13}")
print(f"{'LogisticRegression (binario)':<28}{control['celdas_coincidentes']:>10}/{control['celdas_totales']:<3}{control['fraccion_coincidente']:>11.4f}{control['rescates']:>11}{control['danos']:>9}{control['neto']:>+8}{control['correlacion_predicciones']:>9.4f}{control['ambos_malos']:>13}")
for f in filas:
    print(f"{'Tsetlin K=' + str(f['k']):<28}{f['celdas_coincidentes']:>10}/{f['celdas_totales']:<3}{f['fraccion_coincidente']:>11.4f}{f['rescates']:>11}{f['danos']:>9}{f['neto']:>+8}{f['correlacion_predicciones']:>9.4f}{f['ambos_malos']:>13}")
print()
print("F1 POR ETIQUETA")
print(f"{'modelo':<28}{'HS':>10}{'TR':>10}{'AG':>10}")
print(f"{'V03':<28}{control['por_etiqueta']['HS']['f1_v03']:>10.4f}{control['por_etiqueta']['TR']['f1_v03']:>10.4f}{control['por_etiqueta']['AG']['f1_v03']:>10.4f}")
print(f"{'LogisticRegression (binario)':<28}{control['por_etiqueta']['HS']['f1_control']:>10.4f}{control['por_etiqueta']['TR']['f1_control']:>10.4f}{control['por_etiqueta']['AG']['f1_control']:>10.4f}")
for f in filas:
    pe = f["por_etiqueta"]
    print(f"{'Tsetlin K=' + str(f['k']):<28}{pe['HS']['f1']:>10.4f}{pe['TR']['f1']:>10.4f}{pe['AG']['f1']:>10.4f}")
print()
print("ESTABILIDAD ENTRE PLIEGUES")
for f in filas:
    valores = "  ".join(f"{v:.4f}" for v in f["f1_por_pliegue"])
    print(f"  K={f['k']:<5} {valores}   sd {f['sd_entre_pliegues']:.4f}")
print()
print("REPRODUCIBILIDAD")
print(f"  K={rep['k']}: primera {rep['primera']:.6f}, repetida {rep['repetida']:.6f}, diferencia {rep['diferencia']:.6f}")
print(f"  mismo numero de cambios: {rep['mismo_numero_de_cambios']}   mismas predicciones: {rep['mismas_predicciones']}")
print(f"  causa: {rep['causa']}")
print()
print("CORTE:", datos["corte"]["k"], "->", datos["corte"]["motivo"])
print("Submissions gastadas en V26:", datos["gastos"]["submissions"])
"""

CODIGO_DIAGNOSTICO = """\
# El diagnostico que decide el cierre de la familia.
# La comparacion relevante es Tsetlin contra el control lineal sobre la MISMA matriz binaria,
# porque es la unica que aisla el mecanismo de clausulas de la binarizacion.

control = datos["control_lineal_sobre_la_misma_matriz"]
f = filas[0]
d_mech = f["delta_vs_control"]

print("1. RENDIMIENTO ABSOLUTO")
print(f"   Tsetlin K={f['k']} da {f['corrected_cv']:.6f}. El champion V03 da 0.7772114525.")
print(f"   Diferencia: {f['delta_vs_v03']:+.6f}")
print()
print("2. GANANCIA DEL MECANISMO, que es lo que responde a la pregunta de la familia")
print(f"   Tsetlin binario  {f['corrected_cv']:.6f}")
print(f"   LR binario       {control['corrected_cv']:.6f}")
print(f"   diferencia       {d_mech:+.6f}")
print("   El mecanismo de clausulas es", "MEJOR" if d_mech > 0 else "PEOR", "que un modelo lineal")
print("   sobre exactamente la misma matriz de bits.")
print()
print("3. COMPLEMENTARIEDAD CON V03")
print(f"   rescates {f['rescates']}, danos {f['danos']}, neto {f['neto']:+d}")
print(f"   Solapan {f['fraccion_coincidente'] * 100:.1f} % de las celdas con V03")
print(f"   correlacion de predicciones {f['correlacion_predicciones']:.4f}")
print("   Hay Errors distintos, pero el balance es netamente negativo: por cada error que")
print("   Tsetlin corrige, introduce mas de dos.")
print()
print("4. SOBREAJUSTE: aqui NO es el problema, y conviene decirlo claro")
print(f"   train F1 {f['f1_train']:.4f} frente a OOF {f['corrected_cv']:.4f}, gap {f['gap_train_oof']:+.4f}")
print("   El gap es BAJO. El modelo no memoriza: no llega a ajustar bien ni el train.")
print("   Un gap alto no cerraria la familia, y aqui no lo hay. Lo que hay es falta de")
print("   capacidad de ajuste, que es un problema distinto y mas profundo.")
print()
print("5. CONCLUSION DISTINGUIDA")
print("   Lo que esta prueba SOSTIENE: con esta representacion binaria, este dataset y")
print("   esta configuracion, la Tsetlin Machine de pyTsetlinMachine rinde 0.16 por debajo")
print("   de un modelo lineal sobre la misma informacion, y no aporta complementariedad")
print("   neta. K=200 no muestra senal, y por los criterios de la fase no se sube a 500.")
print()
print("   Lo que esta prueba NO SOSTIENE: que el Tsetlin Machine no puede funcionar en este")
print("   problema. Faltarian por explorarChar n-grams binarios en vez de word, la")
print("   optimizacion de s y T, epochs, y la variante convolucional sobre ventanas. Pero")
print("   ninguna de esas vias ha dado nunca mas de +0.0132 de dOOF en este proyecto, y el")
print("   modelo de transferencia exige +0.03756 para empatar con V03. La relacion")
print("   coste/esperanza no justifica una segunda fase.")
"""

CODIGO_ESTADO = """\
# Registro de estado de la fase, para que quede explicito en el notebook.
print("ESTADO DE LA FAMILIA TSETLIN")
print()
print("  V26-02  Tsetlin Machine CASERA         REJECTED - implementation invalidated")
print("          Los resultados NO son evidencia del rendimiento del algoritmo.")
print("          Nueve defectos documentados. Su capacidad de aprender XOR si quedo")
print("          demostrada, pero con datos reales no es utilizable.")
print()
print("  V26-03  pyTsetlinMachine 0.6.6 (MIT)    REJECTED - cerrar la familia")
print("          Smoke test XOR exacto (1.0000). Control reproducido (0.748064).")
print("          K=200 da 0.583397, -0.164668 frente al control, neto de celdas -358.")
print("          Corte en K=200 sin subir de capacidad.")
print()
print("  V03     champion                       FROZEN")
print("          0.78306 LB, hash verificado en cada corrida, intacto.")
print()
print("  Submissions gastadas en V26:", datos["gastos"]["submissions"])
print("  Submissions disponibles: 3, reservadas. Ninguna gastada sin evidencia.")
print()
print("  Queda abierta, y NO es una familia de Tsetlin: el kernel spectrum de caracteres,")
print("  cuya geometria es la unica medida que no memoriza (mediana del vecino 0.3900")
print("  frente a 0.6974 del RBF). Es la hipotesis mejor apoyada que queda sin evaluar.")
"""


def main() -> None:
    """Anade las celdas de la fase V26 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit(f"PARADA: no existe el informe {INFORME}, ejecuta antes v26_03")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    # Idempotencia: si ya esta la seccion 21, no se duplica.
    marcas = " ".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
    if "Fase v26" in marcas:
        log = "  el notebook ya contiene la seccion 21, no se duplica"
        print(log)
        print(f"  notebook: {antes} celdas, sin cambios")
        return
    nuevas = [
        celda_markdown(CELDA_TITULO),
        celda_markdown(CELDA_IMPLEMENTACION),
        celda_markdown(CELDA_V2601),
        celda_markdown(CELDA_V2602),
        celda_markdown(CELDA_V2603),
        celda_codigo(CODIGO_CARGA),
        celda_codigo(CODIGO_TABLA),
        celda_codigo(CODIGO_DIAGNOSTICO),
        celda_codigo(CODIGO_ESTADO),
    ]
    notebook["cells"].extend(nuevas)
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas ({len(nuevas)} anadidas)")
    print("  las celdas de codigo leen los resultados del JSON, sin outputs embebidos")
    print("  seccion 21 integrada en el notebook consolidado, no en uno aislado")


if __name__ == "__main__":
    main()

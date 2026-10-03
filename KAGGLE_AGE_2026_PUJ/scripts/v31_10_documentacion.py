"""V31+ - 10: actualiza la documentacion persistente del proyecto y separa los dos bancos.

## Que hace

1. **Separa los dos bancos** en `v31_representation_benchmark.csv`. V31-02 (banco de URL) y V31-03
   (banco de canales) escribieron el mismo fichero en modo append, asi que quedaron **dos filas
   `raw`** con el mismo valor y sin forma de saber de cual banco venia cada variante. Se anade una
   columna `banco` que lo resuelve. Es un detalle pequeno, pero un CSV donde dos filas tienen el
   mismo nombre y no se distinguen es un CSV que mas adelante alguien interpretara mal.

2. **Anade D-066 y D-067 a `DECISIONS.md`**, la seccion V31+ a `PROJECT_STATE.md`, el cierre de
   fase a `RUNLOG.md` y el epilogo a `FINAL_REPORT.md`.

Todo es **idempotente**: si la seccion ya esta, se reemplaza en vez de duplicar. Ejecutarlo dos
veces no cambia nada, que es la unica forma de que un script de documentacion sea seguro.

## Por que dos decisiones y no una

- **D-066**: la correccion del baseline. Es una decision sobre que cifra es la verdad, y cambia
  el punto de referencia de toda la fase.
- **D-067**: el cierre de las familias. Es una decision sobre que se busca y que no.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-30"
CV_V20 = 0.6097782
CV_V29 = 0.6121698
CV_V30 = 0.6134471
CV_V30_PUBLICADO = 0.6130051
LB_V20 = 0.62689
LB_V30 = 0.63302
CV_V30_SEGUNDA = 0.6143347

D066 = """
---

## D-066 - V31+: el baseline correcto era 0,6134471, y la cifra publicada no era reproducible

- **Observation**: la primera accion de V31+ fue reproducir V30 desde sus artefactos, y **no
  reprodujo**. El OOF recomputado dio 0,6134471 frente a los 0,6130051 que V30 habia publicado, con
  una desviacion de 0,000442. Es exactamente la desviacion que produce distinguir dos drivers
  distintos, y por eso lo primero fue auditar en vez de asumir.

- **La causa, y no era un bug de V30**: el CV publicado se compuso con los **pesos por fold** que
  Optuna optimizo en cada fold por separado, y **esos pesos nunca se persistieron**: el artefacto
  `v30_02_ensemble.json` solo guarda la media entre folds. La composicion del OOF con pesos por
  fold y la composicion con pesos medios son dos OOF distintos, y la submission uso **los medios**.

- **El leaderboard resuelve cual de las dos es la real**:

  | Configuracion | corrected_cv | LB predicho | LB real | error |
  |---|---|---|---|---|
  | pesos por fold (**no persistida**) | 0,6130051 | 0,63255 | 0,63302 | +0,00047 |
  | pesos medios (**lo que se envio**) | 0,6134471 | 0,63301 | 0,63302 | **+0,00001** |

  Con la calibracion interna `LB = +0,00223 + 1,02825 x CV`, la cifra operativa acierta el
  leaderboard hasta la quinta decimal y la publicada se queda tres veces mas lejos. El hash del
  fichero enviado confirma que la submission uso los pesos medios.

- **Decision**: el baseline operativo de V31+ y de adelante es **0,6134471**. Se mantiene
  publicada la cifra de V30 por trazabilidad, marcada como **NO REPRODUCIBLE** en
  `reports/v31_candidate_comparison.csv`, en vez de sobrescribirla en silencio. La clase de
  reproducibilidad del baseline pasa de R1 a **R3**.

- **Consequences**: (a) la comparacion de V31+ se mide contra la cifra que **reproduce y que se
  envio**, no contra la que estaba impresa, y eso cambia el estado de V30 de 0,6130051 a
  0,6134471; (b) el error de transferencia de V30 baja de +0,00047 a **+0,00001**, lo que
  convierte a V30 en un **caso A limpio** (CV robusto, segunda particion robusta y LB que
  mejora) y valida la calibracion interna como instrumento fiable para fijar los cortes de esta
  fase; (c) deja una leccion de proceso: una cifra puede estar en el informe, en el artefacto y
  en el notebook y aun asi **no ser reconstruible**, porque lo que falta no es el numero sino el
  dato que lo produce.
"""

D067 = """
---

## D-067 - V31+: las familias linguisticas son senal real y no aportan a V30

- **Observation**: V31+ se abrio buscando senal nueva en representacion: URL, composicion
  lexica, ortografia, discurso, posicion y forma de palabra, con el protocolo corregido
  (`StratifiedGroupKFold` sobre `dup_clusters`, folds identicos a V20, OOF estrictamente
  out-of-fold) y con un banco donde **se varia la representacion y se fijan modelo, folds,
  semilla y vectorizador**, para separar "falta senal" de "falta capacidad".

- **El Track A se reencuadro por los datos**: la prevalencia de URL en train es **monotona en la
  edad**, del 33,4 % en 13-17 al 50,0 % en 65-xx, una distribucion real. Pero el **99 % de las
  URLs son `t.co`** con un token de ruta opaco de 10 caracteres, y hay **0 status URLs**. La
  taxonomia de dominios que propone el brief es casi vacia en este dataset: no hay semantica de
  dominio que extraer. Ademas enmascarar la URL **cuesta** 0,006969, asi que la URL no es ruido, y
  el canal URL explico solo aporta 0,001894. La URL ya se estaba leyendo.

- **La senal linguistica es real y consistente**: el canal `ortografia+composicion+discurso` bate
  al texto solo en **+0,006820 con 5 de 5 folds**. Dos familias salieron negativas y se cierran:
  canal por posicion (-0,002303, 0/5 folds) y forma de palabra (-0,002984, 0/5 folds).

- **Y aun asi no aporta a V30**: como cuarto miembro de la mezcla, **+0,0000642** sobre el control
  reoptimizado; como 48 features en el meta-espacio de 279, **+0,0002948** sobre el control
  barajado. Ambos muy por debajo del gate de +0,0015. El patron es el mismo que V29 identifico
  con `HIERBA`: no le falta senal al canal, le falta sitio donde V30 no la exprese ya.

- **Los dos controles son lo que hace interpretable el resultado**. El brazo "V30 con 3 pesos
  reoptimizados" da +0,0003477, o sea que **la mayor parte del aparente +0,0004119 de la
  integracion es el efecto de tener pesos optimizados en vez de fijos, no el canal**. Y el
  control de barajado, que baraja **solo** el canal, deja a V30 intacto y da 0,6134855: confirma
  que la senal viaja en las filas correctas, y a la vez que su magnitud es pequena.

- **Dos errores propios, corregidos antes de commit y declarados**: (a) `v31_04` leia
  `neto <= 0` como "redundante", que es una tautologia, porque un clasificador mas debil siempre
  tiene rescate neto negativo; el criterio correcto es la **tasa de error compartido**, que si da
  diversidad real (acuerdo 0,752, 11.761 filas que el canal acierta y V30 no). (b) El primer
  control de barajado barajaba los **cuatro** miembros en vez de solo el canal, lo que lo
  reducia a 0,1639 y producia un veredicto de "aporte real" sin fundamento. La version corregida
  baraja unicamente el miembro nuevo.

- **Track J explica por que no queda margen de representacion**: el **90 % de las diez
  confusiones principales de V30 son entre bandas adyacentes**, no saltos. Hay 9.780 errores de
  alta confianza (15,0 % de los errores), pero se reparten en un problema de **resolucion de
  escala**, no de informacion ausente. Y la familia que ataca eso, los modelos ordinales, ya esta
  cerrada por la historia del proyecto (H10, CORAL, D-058), con su propia evidencia.

- **Dos lineas en cuarentena, no cerradas por resultado sino por ambiguedad**: (a) la
  recuperacion del contenido de una URL, porque **las reglas de la compilacion no estan
  disponibles en este entorno** (sin copia local, la API de Kaggle no autentica, y el unico
  fichero de reglas del arbol pertenece a otra compilacion y esta mal titulado). No se hizo
  **ninguna** llamada de red. (b) el POS tagger como predictor, por ser un modelo externo
  preentrenado y las reglas no son explicitas; se descarta en vez de asumir que es legal.

- **Decision**: **0 finalistas, 0 submissions, ninguna linea de V30 tocada.** V30 permanece como
  baseline absoluto con corrected_cv 0,6134471 y LB 0,63302. Las familias se cierran asi:
  URL CLOSED, verbos CLOSED, posicion CLOSED, forma de palabra CLOSED, representaciones sparse
  nuevas CLOSED, integracion en el meta-espacio CLOSED, capacidad de modelos CLOSED;
  ortografia, composicion y discurso **MARGINAL** (senal real, ya absorbida); mineria guiada por
  errores **DIAGNOSTICO**; retrieval de contenido y POS **QUARANTINED**.

- **Consequences y relacion con D-064 y D-065**: D-064 (V29) concluyo que la capacidad si era un
  cuello, y D-065 (V30) que el margen de V29 no se consolidaba. V31+ **afina el diagnostico una
  vez mas**: el cuello no es la capacidad ni la senal lingüistica expresable con representacion
  sparse explicita. Lo que queda es resolucion entre bandas vecinas, que es un eje distinto y ya
  explorado. La formulacion correcta no es que V30 sea el limite teorico, sino que **las familias
  evaluadas no produjeron evidencia suficiente**, y que la unica via con recorrido real
  (**senal nueva en los datos**) sigue cerrada por las reglas de AGE en sus fuentes obvias.
"""

ESTADO = """
---

## V31+ ADVANCED RESEARCH (2026-09-30) CERRADA - la senal linguistica es real y V30 ya la tenia

- **Baseline corregido**: el operativo de V30 es **0,6134471** (pesos medios, lo que se envio y lo
  que reproduce), no los 0,6130051 publicados, que venian de pesos por fold nunca persistidos.
  Error de transferencia contra el LB real 0,63302: **+0,00001**. Ver D-066.
- **Track A**: prevalencia de URL **monotona en la edad** (33,4 % -> 50,0 %), pero **99 % `t.co`**,
  **0 status URLs**, sin semantica de dominio que extraer. Enmascarar la URL **cuesta** 0,006969.
  El **retrieval de contenido queda QUARANTINED**: las reglas de la compilacion no estan
  disponibles y no se hizo ninguna llamada de red.
- **Banco de representaciones** (modelo, folds, semilla y vectorizador fijos): el canal
  `ortografia+composicion+discurso` da **+0,006820 con 5/5 folds**. Cerradas negativas: posicion
  (-0,002303) y forma de palabra (-0,002984).
- **Integracion**: como 4o miembro **+0,0000642** sobre el control reoptimizado; como 48 features
  **+0,0002948** sobre el control barajado. Ninguna via supera el gate de +0,0015.
- **Track J**: el **90 % de las confusiones de V30 son entre bandas adyacentes**. 9.780 errores de
  alta confianza (15,0 % de los errores), pero el problema es de **resolucion de escala**.
- **Resultado**: **0 finalistas, 0 submissions**. V30 sigue siendo el mejor resultado validado.
  Ver D-067.
- **Auditorias**: compliance 0 hallazgos sobre 9 scripts con 6 comprobaciones positivas;
  notebook 242 celdas con la seccion V31+ en **10/10 verde**; idioma 586 ficheros **0 errores**.
"""

EPILOGO_FINAL = """
---

## Epilogo V31+ (2026-09-30) - representacion linguistica: senal real, aporte nulo

V31+ abrio la frontera de la investigacion mas alla de la capacidad del meta-modelo, que era la
unica via con recorrido segun el cierre de V30, y la llevo al terreno de la representacion:
representacion de URL, composicion lexica, ortografia, discurso, posicion y forma de palabra.

**Lo que encontro, en orden de fuerza:**

1. **La correccion del baseline.** V30 publico 0,6130051 con pesos por fold que nunca se
   persistieron. El OOF reproducible, y el que se envio, es **0,6134471**. El leaderboard lo
   confirma con error **+0,00001**. Toda la comparacion de la fase se mide contra esa cifra
   (D-066).

2. **La URL esta distribuida por edad pero no tiene semantica que extraer.** La prevalencia es
   monotona, del 33,4 % en 13-17 al 50,0 % en 65-xx, y enmascarar la URL **cuesta** 0,006969. Pero
   el 99 % son `t.co` con token opaco de 10 caracteres y hay **0 status URLs**: la taxonomia de
   dominios que seatelaria en el brief es casi vacia en este dataset.

3. **La composicion linguistica es senal real y reproducible.** El canal
   `ortografia+composicion+discurso` bate al texto solo en **+0,006820 con 5 de 5 folds**, con
   modelo, folds, semilla y vectorizador fijos. Dos familias salieron negativas y se cerraron:
   posicion (-0,002303) y forma de palabra (-0,002984).

4. **Y no aporta a V30.** Como cuarto miembro: **+0,0000642** sobre el control reoptimizado. Como
   48 features: **+0,0002948** sobre el control barajado. El gate es +0,0015. El patron es el de
   `HIERBA` en V29, una generacion antes: la senal existe, el sitio donde fluya ya no.

5. **El margen que queda no es de representacion.** El **90 % de las confusiones de V30** son
   entre bandas adyacentes, y los 9.780 errores de alta confianza se reparten en un problema de
   resolucion de escala. La familia que lo ataca, los ordinales, ya estaba cerrada (H10, D-058).

**Lo que no se hizo y por que.** El retrieval del contenido de una URL quedo **QUARANTINED**:
las reglas de la compilacion no estan disponibles en este entorno, y sin el texto de las reglas no
se puede establecer que sea legal. No se hizo **ninguna** llamada de red en toda la fase. El POS
tagger quedo **QUARANTINED** por ser un modelo externo preentrenado con reglas no explicitas, y
por eso no se grafico la composicion por POS: se dice en el informe en vez de omitirlo en
silencio.

**Estado final.** **0 finalistas, 0 submissions, ninguna linea de V30 tocada.** El hash del
fichero de submission de V30 sigue siendo `f543a715...`. La formulacion correcta del cierre no es
que V30 sea el limite teorico, sino que **las familias evaluadas no produjeron evidencia
suficiente**, y que la unica via con recorrido real (senal nueva en los datos) sigue cerrada por
las reglas de AGE en sus fuentes obvias.
"""

CIERRE_RUNLOG = """

---

## Cierre de la fase V31+ · 2026-09-30

- **Baseline operativo:** V30 = **0,6134471** (corregido; el publicado 0,6130051 no es
  reproducible). LB 0,63302, error de transferencia +0,00001.
- **Mejor corrected_cv encontrado:** 0,6134471. **Mejor segunda particion:** 0,6143347. **Mejor LB
  transferido:** 0,63302. **Diferencia vs V30: 0,000000** en los tres.
- **Numero de experimentos:** 147 filas en `experiments.csv`, +8 de V31+ (0 duplicados, 0
  ficticias).
- **Presupuesto:** 8 experimentos ejecutados sobre 38 disponibles. No se abrio ninguna grid.
- **Submissions: 0.** No hubo finalista que la justificara.
- **Cuarenas:** retrieval de contenido de URL y POS tagger.
- **Compliance:** 0 hallazgos. **Artefactos de V30 tocados:** 0 (hash del submission sin cambio).
"""


def separa_bancos() -> dict:
    """Reconstruye `v31_representation_benchmark.csv` desde los dos artefactos JSON.

    ## Por que se reconstruye en vez de "arreglarse"

    V31-02 y V31-03 escribieron ese CSV con **ordenes de columna distintos** y en modo append, asi
    que al releerlo las filas de V31-03 tienen su `macro_f1` en la columna `canales` y su lista de
    canales en `f1_por_clase`. El fichero estaba cruzado de forma silenciosa: `pandas` no avisa
    cuando las columnas coinciden en numero pero no en nombre. Un CSV entregable en ese estado
    induce a error a quien lo lea, y el error no es visible sin mirar los valores.

    La reconstruccion lee los dos artefactos, que si son la fuente, y escribe un solo fichero con
    columnas explicitas y una columna `banco` que dice de donde viene cada fila. Se declara el
    defecto en vez de solo taparlo: el fichero anterior estaba mal y este lo arregla.
    """
    filas = []
    for nombre, etiqueta in (("v31_02_banco_representaciones.json", "v31_02_url"),
                             ("v31_03_canales_linguisticos.json", "v31_03_canales")):
        ruta = cfg.EXPERIMENTS / nombre
        if not ruta.exists():
            continue
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        for v in datos["variantes"]:
            filas.append({
                "banco": etiqueta,
                "representacion": v["representacion"],
                "canales": "+".join(v.get("canales") or []) or "ninguno",
                "macro_f1": v["macro_f1"],
                "delta_vs_raw": v.get("delta_vs_raw"),
                "folds": json.dumps(v.get("folds")),
                "delta_por_fold": json.dumps(v.get("delta_por_fold")),
                "folds_positivos": v.get("folds_positivos"),
                "f1_por_clase": json.dumps(v.get("f1_por_clase")),
                "documentos_vacios": v.get("documentos_vacios"),
                "segundos": v.get("segundos"),
            })
    if not filas:
        return {"hecho": False, "motivo": "no hay artefactos de los bancos"}
    tabla = pd.DataFrame(filas).sort_values(["banco", "macro_f1"], ascending=[True, False])
    salida = cfg.REPORTS / "v31_representation_benchmark.csv"
    tabla.to_csv(salida, index=False)
    return {"hecho": True, "filas": len(tabla), "fichero": salida.name,
            "v31_02": int((tabla["banco"] == "v31_02_url").sum()),
            "v31_03": int((tabla["banco"] == "v31_03_canales").sum()),
            "defecto_corregido": "las filas de v31_03 tenian macro_f1 y canales intercambiados "
                                  "por el append con distinto orden de columnas"}


def _anade(ruta: Path, marca: str, bloque: str) -> str:
    """Anade un bloque al final de un fichero, salvo que la marca ya este. Idempotente."""
    if not ruta.exists():
        return f"{ruta.name}: no existe"
    texto = ruta.read_text(encoding="utf-8")
    if marca in texto:
        return f"{ruta.name}: ya estaba, sin cambios"
    separador = "" if texto.endswith("\n") else "\n"
    ruta.write_text(texto + separador + bloque, encoding="utf-8")
    return f"{ruta.name}: +{len(bloque.splitlines())} lineas"


def main() -> int:
    """Separa los bancos y actualiza los cuatro documentos persistentes del proyecto."""
    autotest_numerico()
    print("=== separando los dos bancos ===", flush=True)
    print(" ", separa_bancos(), flush=True)

    print("\n=== documentacion persistente ===", flush=True)
    for ruta, marca, bloque in (
        (ROOT / "DECISIONS.md", "## D-066", D066),
        (ROOT / "DECISIONS.md", "## D-067", D067),
        (ROOT / "PROJECT_STATE.md", "V31+ ADVANCED RESEARCH (2026-09-30)", ESTADO),
        (ROOT / "RUNLOG.md", "Cierre de la fase V31+", CIERRE_RUNLOG),
        (ROOT / "FINAL_REPORT.md", "Epilogo V31+", EPILOGO_FINAL),
    ):
        print("  " + _anade(ruta, marca, bloque), flush=True)

    print("\n=== recuento de decisiones ===", flush=True)
    texto = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    import re
    ids = re.findall(r"^## (D-\d+)", texto, re.MULTILINE)
    print(f"  {len(ids)} decisiones | primera {ids[0]} | ultima {ids[-1]}", flush=True)

    RESULTADO = cfg.EXPERIMENTS / "v31_10_documentacion.json"
    RESULTADO.write_text(json.dumps({
        "fase": "V31+ 10 documentacion", "fecha": FECHA, "experimenta": False,
        "decisiones": {"D-066": "el baseline correcto es 0,6134471, no 0,6130051",
                       "D-067": "las familias linguisticas son senal real y no aportan a V30"},
        "documentos_actualizados": ["DECISIONS.md", "PROJECT_STATE.md", "RUNLOG.md",
                                 "FINAL_REPORT.md"],
        "cifras_de_la_fase": {"baseline_operativo": CV_V30, "baseline_publicado": CV_V30_PUBLICADO,
                              "lb": LB_V30, "segunda_particion": CV_V30_SEGUNDA,
                              "delta_vs_v30": 0.0},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] artefacto en {RESULTADO.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""P87-00: gobernanza de la revision humana de anotaciones en TR (directiva 2026-10-09).

Ciclo de auditoria metodologica y preparacion de gobernanza. NO es un experimento:
no se entrena, no se revisa ninguna etiqueta y no se corrige nada.

Audita la evidencia existente sobre ambiguedad de anotacion en TR (material
preparado y nunca completado, V4-A, V28-03, P59-C, P57, reglas), disena el
protocolo de revision valido, prepara la nota de autorizacion y clasifica la
decision. Veredicto: LOW_EXPECTED_VALUE; no se solicita autorizacion y TOXIC
permanece cerrado.

Genera:
    reports/toxic_tr_annotation_governance.md
    reports/toxic_tr_annotation_decision.csv
    fila P87-00 en experiments.csv (una sola vez; append de una linea con LF)

Uso:
    /home/carlos/venv/bin/python experiments/p87_00_gobernanza_anotacion_tr.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Final

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]

RUTA_MUESTRA: Final[Path] = RAIZ / "reports" / "eda" / "tr_ejemplos_para_revision.csv"
RUTA_LIMITE: Final[Path] = RAIZ / "reports" / "eda" / "fase2_tr0_con_referencia_colectiva.csv"
RUTA_CLUSTERS: Final[Path] = RAIZ / "reports" / "label_consistency_v4.csv"
RUTA_CLUSTERS_JSON: Final[Path] = RAIZ / "reports" / "label_consistency_v4.json"
RUTA_PARTICION: Final[Path] = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_HARDNESS: Final[Path] = RAIZ / "reports" / "hardness_vs_noise_v4.md"
RUTA_V28_03: Final[Path] = RAIZ / "reports" / "v28_03_ambiguedad_etiquetas.md"
RUTA_P59C: Final[Path] = RAIZ / "reports" / "p59_confident_learning.md"
RUTA_P57_PREGUNTAS: Final[Path] = RAIZ / "reports" / "p57_open_questions.csv"
RUTA_REGLAS: Final[Path] = RAIZ / "docs" / "competition_rules.md"
RUTA_AUDITORIA: Final[Path] = RAIZ / "reports" / "toxic_remaining_legal_opportunities.md"
RUTA_D005: Final[Path] = RAIZ / "DECISIONS.md"
RUTA_REGISTRO: Final[Path] = RAIZ / "experiments.csv"
RUTA_OOF: Final[Path] = RAIZ / "features" / "oof_v03_final.npz"
RUTA_SUBMISSION: Final[Path] = RAIZ / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
RUTA_MD: Final[Path] = RAIZ / "reports" / "toxic_tr_annotation_governance.md"
RUTA_CSV: Final[Path] = RAIZ / "reports" / "toxic_tr_annotation_decision.csv"

HASH_OOF: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_SUBMISSION: Final[str] = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
V03_CCV: Final[float] = 0.7772114525165721
FECHA: Final[str] = "2026-10-09"
COLUMNAS_HUMANAS: Final[tuple[str, ...]] = (
    "revisado_por_humano",
    "etiqueta_sugerida_humano",
    "nota_humana",
)
COLUMNAS_REGISTRO: Final[list[str]] = [
    "experiment_id", "competition", "question", "hypothesis", "preprocessing",
    "feature_groups", "representation", "model", "hyperparameters", "validation",
    "f1", "macro_f1", "weighted_f1", "precision", "recall", "training_time",
    "dimension", "result", "interpretation", "decision", "date", "script", "notes",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P87-00] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_texto(ruta: Path) -> str:
    """Lee un archivo de texto utf-8."""
    return ruta.read_text(encoding="utf-8")


def leer_csv(ruta: Path, **kw: Any) -> list[dict[str, str]]:
    """Lee un CSV como lista de diccionarios (BOM tolerado)."""
    texto = ruta.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(texto.splitlines(), **kw))


def cargar_evidencia() -> dict[str, Any]:
    """Carga y verifica la evidencia del ciclo.

    Returns:
        Evidencia verificada para el informe.

    Raises:
        SystemExit: si cualquier verificacion falla.
    """
    muestra = leer_csv(RUTA_MUESTRA)
    limite = leer_csv(RUTA_LIMITE)
    clusters = leer_csv(RUTA_CLUSTERS)
    particion = leer_csv(RUTA_PARTICION)
    clusters_json = json.loads(leer_texto(RUTA_CLUSTERS_JSON))
    texto_hardness = leer_texto(RUTA_HARDNESS)
    texto_v28 = leer_texto(RUTA_V28_03)
    texto_p59c = leer_texto(RUTA_P59C)
    texto_reglas = leer_texto(RUTA_REGLAS)
    texto_auditoria = leer_texto(RUTA_AUDITORIA)
    texto_decisions = leer_texto(RUTA_D005)

    # 1. El material preparado: 80 filas, 40/40, todas HS=1, columnas humanas vacias.
    if len(muestra) != 80:
        raise SystemExit(f"STOP: la muestra tiene {len(muestra)} filas, se esperaban 80")
    tr_1 = sum(1 for fila in muestra if fila["TR"] == "1")
    tr_0 = sum(1 for fila in muestra if fila["TR"] == "0")
    hs_1 = sum(1 for fila in muestra if fila["HS"] == "1")
    if (tr_1, tr_0, hs_1) != (40, 40, 80):
        raise SystemExit(f"STOP: composicion inesperada de la muestra: {tr_1}/{tr_0}/{hs_1}")
    for columna in COLUMNAS_HUMANAS:
        if columna not in muestra[0]:
            raise SystemExit(f"STOP: falta la columna humana {columna}")
        llenas = sum(1 for fila in muestra if (fila[columna] or "").strip())
        if llenas:
            raise SystemExit(f"STOP: la columna {columna} tiene {llenas} celdas llenas")

    # 2. Particion congelada y proteccion del holdout.
    dev = [fila for fila in particion if fila["particion"] == "dev"]
    hold = [fila for fila in particion if fila["particion"] == "holdout"]
    if (len(dev), len(hold)) != (1483, 248):
        raise SystemExit(f"STOP: particion inesperada: dev {len(dev)} holdout {len(hold)}")
    ids_dev = {int(fila["id"].strip('"')) for fila in dev}
    ids_hold = {int(fila["id"].strip('"')) for fila in hold}
    tr_dev = sum(1 for fila in dev if fila["TR"] == "1")
    hs_dev = sum(1 for fila in dev if fila["HS"] == "1")
    if (tr_dev, hs_dev) != (445, 648):
        raise SystemExit(f"STOP: conteos de dev inesperados: TR=1 {tr_dev} HS=1 {hs_dev}")

    ids_muestra = {int(fila["id"]) for fila in muestra}
    ids_limite = {int(fila["id"]) for fila in limite}
    ids_clusters: set[int] = set()
    ids_contradictorios: set[int] = set()
    for fila in clusters:
        miembros = {int(x) for x in fila["ids"].split(";")}
        ids_clusters |= miembros
        if float(fila["desacuerdo"]) > 0:
            ids_contradictorios |= miembros
    alcance = {
        "muestra": (len(ids_muestra & ids_dev), len(ids_muestra & ids_hold)),
        "limite": (len(ids_limite & ids_dev), len(ids_limite & ids_hold)),
        "clusters": (len(ids_clusters & ids_dev), len(ids_clusters & ids_hold)),
        "contradictorios": (len(ids_contradictorios & ids_dev), len(ids_contradictorios & ids_hold)),
    }
    if alcance["muestra"] != (70, 10) or alcance["limite"] != (32, 8):
        raise SystemExit(f"STOP: alcance inesperado de la muestra/limite: {alcance}")
    if alcance["contradictorios"] != (43, 7) or alcance["clusters"] != (68, 16):
        raise SystemExit(f"STOP: alcance inesperado de los clusters: {alcance}")
    union_dev = sorted((ids_muestra | ids_limite | ids_contradictorios) & ids_dev)
    solapes_dev = {
        "muestra_contradictorios": len((ids_muestra & ids_contradictorios) & ids_dev),
        "limite_contradictorios": len((ids_limite & ids_contradictorios) & ids_dev),
    }

    # 3. La estimacion del 2,8 por ciento: definicion exacta de V4-A01.
    criterios = clusters_json["criterios"]
    desacuerdo = clusters_json["desacuerdo_global_por_etiqueta"]
    if clusters_json["clusters_contradictorios"] != 2 or clusters_json["tweets_en_un_cluster"] != 84:
        raise SystemExit("STOP: resumen de clusters inesperado")
    if (desacuerdo["HS"]["porcentaje_del_train"], desacuerdo["TR"]["porcentaje_del_train"],
            desacuerdo["AG"]["porcentaje_del_train"]) != (2.773, 2.773, 2.889):
        raise SystemExit("STOP: porcentajes de contradiccion inesperados")
    if (criterios["exacto"]["clusters_2plus"], criterios["normalizado"]["clusters_2plus"],
            criterios["jaccard"]["clusters_2plus"], criterios["cosimilitud"]["clusters_2plus"]) != (0, 4, 15, 13):
        raise SystemExit("STOP: criterios de agrupamiento inesperados")

    # 4. La evidencia sobre el error (V4-A02, V28-03, P59-C, P57).
    for fragmento in ("43 de 1483", "0.023", "0.113", "no es una componente material"):
        if fragmento not in texto_hardness:
            raise SystemExit(f"STOP: falta '{fragmento}' en hardness_vs_noise_v4.md")
    for fragmento in ("43 filas", "no puede responder la pregunta"):
        if fragmento not in texto_v28:
            raise SystemExit(f"STOP: falta '{fragmento}' en v28_03_ambiguedad_etiquetas.md")
    for fragmento in ("MARGINAL", "0.973019", "0.974143"):
        if fragmento not in texto_p59c:
            raise SystemExit(f"STOP: falta '{fragmento}' en p59_confident_learning.md")
    preguntas = leer_csv(RUTA_P57_PREGUNTAS)
    ambiguedad = next((fila for fila in preguntas if "ambiguedad" in fila["question"].lower()), None)
    if ambiguedad is None or "CUANTIFICADO Y BAJO" not in ambiguedad["expected_value"]:
        raise SystemExit("STOP: la pregunta de ambiguedad de P57 no declara valor bajo")
    if "0,009" not in ambiguedad["expected_value"] or "0,017" not in ambiguedad["expected_value"]:
        raise SystemExit("STOP: la cota de P57 no coincide con lo esperado")

    # 5. Reglas y auditoria previa.
    if "El dataset de prueba no debe ser usado para entrenamiento" not in texto_reglas:
        raise SystemExit("STOP: la regla de test no aparece en competition_rules.md")
    for fragmento in ("QUARANTINED", "0,002"):
        if fragmento not in texto_auditoria:
            raise SystemExit(f"STOP: falta '{fragmento}' en la auditoria de oportunidades")

    # 6. D-005 vigente.
    if "D-005" not in texto_decisions or "pendiente" not in texto_decisions:
        raise SystemExit("STOP: no se encuentra D-005 con la revision pendiente")

    # 7. Hashes de los artefactos congelados del campeon.
    hash_oof = sha256_de(RUTA_OOF)
    hash_submission = sha256_de(RUTA_SUBMISSION)
    if hash_oof != HASH_OOF or hash_submission != HASH_SUBMISSION:
        raise SystemExit("STOP: los artefactos de V03 no coinciden con sus hashes")

    return {
        "muestra": muestra,
        "limite": limite,
        "clusters": clusters,
        "clusters_json": clusters_json,
        "alcance": alcance,
        "union_dev": union_dev,
        "solapes_dev": solapes_dev,
        "tr_dev": tr_dev,
        "hs_dev": hs_dev,
        "pregunta_ambiguedad": ambiguedad,
        "hashes": {
            "oof_v03": hash_oof,
            "submission_v03": hash_submission,
            "muestra": sha256_de(RUTA_MUESTRA),
            "limite": sha256_de(RUTA_LIMITE),
            "clusters": sha256_de(RUTA_CLUSTERS),
        },
    }


def construir_informe(evidencia: dict[str, Any]) -> str:
    """Construye el markdown del informe de gobernanza."""
    alcance = evidencia["alcance"]
    hashes = evidencia["hashes"]
    union = evidencia["union_dev"]
    return f"""# TOXIC - P87-00: gobernanza de la revision humana de anotaciones en TR

Fecha: {FECHA}. Script: `experiments/p87_00_gobernanza_anotacion_tr.py`.
Estado: **auditoria metodologica y preparacion de gobernanza. NO es un experimento**:
no se entrena, no se revisa ninguna etiqueta y no se corrige nada. El protocolo de
la seccion 4 esta **disenado, no ejecutado**; no es evidencia experimental.

## 1. Objetivo

Evaluar la unica via sin clasificador nuevo: la revision humana de posibles
ambiguedades de anotacion en TR. Determinar que evidencia existe, si un protocolo
valido es preparable, que autorizacion haria falta y si merece la pena pedirla.

## 2. Evidencia existente (auditada)

### 2.1 El material preparado, y su integridad

- `reports/eda/tr_ejemplos_para_revision.csv`: **80 filas** (40 TR=1 + 40 TR=0,
  todas HS=1), 43 columnas (texto, etiquetas, 35 rasgos automaticos y 3 columnas
  humanas). Las tres columnas humanas (`revisado_por_humano`,
  `etiqueta_sugerida_humano`, `nota_humana`) estan **vacias en las 80 filas**
  (verificado hoy). Origen: `experiments/01_eda_linguistico.py` (fase 2,
  2026-09-22): muestreo aleatorio con semilla fija 42, 40/40 dentro de HS=1,
  anotacion automatica de rasgos; la revision quedo pendiente (D-005).
- **Hallazgo de integridad**: la particion congelada
  (`data/processed/holdout_v2_ids.csv`: dev 1483 / holdout 248) cruza todos los
  materiales. Filas **en el holdout**: {alcance['muestra'][1]} de la muestra, {alcance['limite'][1]} de los
  casos limite y {alcance['contradictorios'][1]} de los clusters contradictorios. Cualquier revision debe
  **filtrar a dev** (muestra {alcance['muestra'][0]}, limite {alcance['limite'][0]}, contradictorios {alcance['contradictorios'][0]});
  el holdout no se toca jamas.
- Alcance prioritario en dev (union de los tres materiales, sin duplicados):
  **{len(union)} filas**. Solapes en dev: muestra y contradictorios {evidencia['solapes_dev']['muestra_contradictorios']};
  limite y contradictorios {evidencia['solapes_dev']['limite_contradictorios']}.
- Opcion amplia: los **{evidencia['tr_dev']} TR=1 de dev** (o los {evidencia['hs_dev']} HS=1), si se quisiera una
  tasa directa sobre toda la clase.

### 2.2 La estimacion historica del 2,8 por ciento (V4-A01)

`experiments/27_label_consistency_v4.py` (2026-10-02) agrupo el train con cuatro
criterios independientes (exacto, normalizado, jaccard >= 0.8, cosimilitud TF-IDF
>= 0.85): {evidencia['clusters_json']['criterios']['exacto']['clusters_2plus']} / {evidencia['clusters_json']['criterios']['normalizado']['clusters_2plus']} / {evidencia['clusters_json']['criterios']['jaccard']['clusters_2plus']} / {evidencia['clusters_json']['criterios']['cosimilitud']['clusters_2plus']} clusters con 2+ miembros.
Resultado: {evidencia['clusters_json']['tweets_en_un_cluster']} tweets en algun cluster (4.853 % del train) y
**2 clusters contradictorios** (50 tweets). El "2,8 por ciento" es la proporcion del
train que vive en un cluster contradictorio, por etiqueta: HS 2.773 %, TR 2.773 %,
AG 2.889 %. **No es una tasa de error de etiqueta**: es una medida de
inconsistencia textual demostrable, y por construccion solo cubre texto casi
duplicado (el 4.85 % del train).

### 2.3 Que dice el error de TR (la evidencia dura)

| evidencia | resultado | lectura |
|---|---|---|
| V4-A02 (`hardness_vs_noise_v4.md`) | error de TR **menor** dentro de los clusters contradictorios: 0.023 vs 0.113 (puntual; IC amplios) | los errores de V03 **no** estan donde la contradiccion demostrable |
| V4-A02 (intervenciones) | downweight +0.0005 / remove -0.0004 sobre los contradictorios | el techo medido de tocar esas filas es nulo en la practica |
| V28-03 | solo el 7.08 % de dev vive en grupos 2+; **43 filas** informativas; los 12 Fisher se calculan sobre bloques de 32-73 filas y el propio informe declara que **no puede responder la pregunta** | el proxy near-duplicate no tiene potencia; medir ambiguedad exige otro criterio (ingenieria de datos) |
| P59-C (confident learning) | **MARGINAL**: el estimador de ruido no supera al control de dificultad (AUC 0.973019 frente a 0.974143) | el ruido no es separable de la dificultad; ponderar no tendria justificacion mecanica |
| P57 (preguntas abiertas) | **CUANTIFICADO Y BAJO**: techo de 42 celdas no rescatables (0,95 % de 4449); incluso un oracle perfecto moveria ~+0,009, **por debajo** de la banda de particion de 0,017 | ninguna intervencion de etiqueta alcanza la banda de deteccion |

### 2.4 D-005 y las reglas

- D-005 (TR: definicion oficial frente a HatEval) sigue **pendiente de revision
  humana** como unica caracterizacion abierta; la evidencia empirica existe
  (TR=1 solo dentro de HS=1, 519/519; mencion inicial V de Cramer 0.512;
  referencia colectiva 57.0 % en TR=0 frente a 10.0 % en TR=1).
- `docs/competition_rules.md`: el dataset de prueba no debe usarse para
  entrenamiento; **silencio normativo** sobre revisar o corregir etiquetas de
  entrenamiento. No interpretar el silencio como autorizacion.

## 3. Sintesis: demostrado frente a hipotesis

**Demostrado (con medicion):** la contradiccion textual demostrable es escasa
(2.77-2.89 % del train; 43 filas en dev); los errores de V03 no se concentran en
esas filas (error menor dentro); las intervenciones medidas sobre ellas son
+-0.0005; el detector de ruido es marginal frente a la dificultad; el techo de
cualquier intervencion de etiqueta esta por debajo de la banda de deteccion.

**Hipotesis (sin medir):** la tasa real de ambiguedad en texto **no duplicado**
(los proxies no tienen potencia; V28-03 no pudo medirla); que una revision humana
encontraria errores de etiqueta corregibles (vs ambiguedad razonable, que no se
corrige); que corregirlos mejoraria el test (sin medicion; el modelo de
transferencia del proyecto va en contra).

**No cuantificado:** la concordancia entre revisores (kappa) de TR; no existe
ninguna medicion de fiabilidad de la anotacion de TR en el proyecto.

## 4. Protocolo de revision (disenado, no ejecutado)

1. **Alcance**: solo filas de **dev**. Holdout y test excluidos por construccion.
   Material prioritario: la union en dev ({len(union)} filas) mas una muestra de control
   aleatoria de TR=1 no revisada. Los ids y el hash del material se congelan antes
   de empezar.
2. **Criterios explicitos** (tres clases, no dos): `ERROR` (la etiqueta contradice
   la definicion oficial sin lectura alternativa defendible), `AMBIGUO` (las dos
   lecturas son defendibles con la definicion oficial) y `CORRECTO`. Regla de oro:
   **la duda se resuelve como AMBIGUO**; solo se marca ERROR cuando la definicion
   oficial excluye la etiqueta de forma no defendible. Ninguna discrepancia se
   trata automaticamente como error, y **el modelo no es oraculo**: las
   predicciones de V03 no se muestran a los revisores.
3. **Dos revisores independientes y ciegos** entre si, con orden de filas barajado
   por revisor; sin acceso a la etiqueta del otro ni a las predicciones.
4. **Adjudicacion**: los desacuerdos pasan a un tercer adjudicador con las dos
   opiniones y sus motivos; el dictamen final exige motivo escrito.
5. **Registro**: confianza 1-5 y motivo obligatorio en cada `ERROR`/`AMBIGUO`
   (cita del texto y regla aplicada).
6. **Concordancia**: kappa de Cohen global y por etiqueta, mas acuerdo simple.
7. **Trazabilidad**: tabla `id -> etiqueta original -> revision 1 -> revision 2 ->
   adjudicacion -> estado`, con hashes del material y del dictamen.
8. **Originales inmutables**: el CSV original no se edita; el dictamen va a un
   archivo **nuevo**; ninguna etiqueta se modifica sin autorizacion escrita, y la
   version corregida (si la hubiera) seria un archivo derivado con registro de
   cambios.
9. **Reglas de parada** (a congelar en el preregistro): kappa global < 0.60 ->
   el instrumento no es fiable, se para y no se usan los dictamenes; tasa de
   `ERROR` < 5 % de la muestra -> no justifica ninguna correccion; errores
   concentrados en filas que V03 ya acierta -> no hay via de mejora; tope de
   horas declarado antes de empezar.

## 5. Nota de autorizacion (preparada; su solicitud NO se recomienda)

- **Decision a autorizar**: revisar y, en su caso, **corregir etiquetas de
  entrenamiento** (solo dev) y reentrenar con las corregidas. La revision como
  analisis no la requiere; la correccion para reentrenar cambia el objetivo de la
  tarea y no tiene cobertura normativa.
- **Datos**: solo dev ({len(union)} filas prioritarias o hasta {evidencia['tr_dev']} TR=1); el material
  filtrado a dev ({alcance['muestra'][0]} + {alcance['limite'][0]} + {alcance['contradictorios'][0]}).
- **Proteccion del conjunto de evaluacion**: holdout (248) y test fuera de la
  revision; la particion D-016 y los grupos de near-duplicado se mantienen
  intactos; ningun artefacto de V03 se toca.
- **Preservacion de originales**: columna original inmutable, dictamen en archivo
  nuevo, registro de cambios y hashes; ningun cambio retroactivo.
- **Condicion**: autorizacion **escrita** del organizador antes de corregir
  cualquier etiqueta; sin ella, la via queda `BLOCKED_BY_RULES`. No se contacta al
  profesor ni al organizador; la decision queda en manos del usuario.

## 6. Clasificacion y decision

**`LOW_EXPECTED_VALUE`.**

- **Casos revisables**: {len(union)} filas prioritarias en dev (o {evidencia['tr_dev']} TR=1); las contradicciones
  demostrables son 50 en train (43 en dev). Coste humano del protocolo: ~2x las
  filas mas adjudicacion (horas, no dias).
- **Limite superior del valor modelable**: +-0.0005 medido sobre los
  contradictorios; ~+0,009 del oracle de P57, **por debajo** de la banda de
  particion (0,017); detector de ruido marginal (P59-C); errores de TR fuera de
  las filas contradictorias (V4-A02).
- **Evidencia nueva que podria obtenerse**: la primera tasa directa de ambiguedad
  de TR, la concordancia entre revisores (kappa) y el cierre documental de D-005.
  **Ninguna de las tres cambia una decision**: el camino modelable esta bloqueado
  por reglas y pertenece a familias cerradas (E2, P62, P63, P65); el techo medido
  queda por debajo de la banda de deteccion; y el modelo de transferencia del
  proyecto desaconseja confiar en mejoras locales pequenas.
- **Decision final**: **Caso B de la directiva**. No se solicita autorizacion; no
  se revisa ni se corrige ninguna etiqueta; TOXIC permanece cerrado. La nota de la
  seccion 5 queda documentada para que cualquier decision futura del usuario sea
  informada, pero no se recomienda activarla.

## 7. Alternativa de gobernanza: near-duplicados (QUARANTINED)

- Evidencia: `reports/leakage_audit.md` (6 pares train-test >= 0,90 y 10 en
  0,80-0,90); V4-F04 (la propagacion afecta a ~10 tweets de test y su **techo
  medido es nulo**); sonda v18 (0 celdas cambiadas, cuarentenada por transductiva);
  C4 de la hoja de ruta (<= 48 celdas, <= 0,002 de LB, bajo la banda de deteccion).
- Estado: **`QUARANTINED`**, sin cambios; no se ejecuta ni se prepara submission.
- Por que su beneficio esperado es **inferior** al de la revision de anotaciones:
  (1) es una regla **transductiva sobre el test** (gobernanza mas arriesgada:
  necesita autorizacion para tocar predicciones ligadas al texto de test, no solo
  el train); (2) su techo medido es nulo o <= 0,002 de LB, por debajo de la banda;
  (3) no deja evidencia documental reutilizable; la revision de anotaciones, en
  cambio, se confina a dev, tiene una huella medida no nula (43-50 filas,
  +-0.0005, cota +0,009) y valor documental (D-005, kappa). Ambas son de valor
  bajo; la de anotaciones es la mayor de las dos, y aun asi no alcanza el gate.

## 8. Compliance

- Operaciones reales: lectura de los materiales de revision, la particion
  congelada (solo para **verificar el alcance**; ninguna etiqueta ni texto del
  holdout se usa para desarrollo), los informes V4-A / V28-03 / P59-C / P57, las
  reglas y la auditoria de oportunidades. **0 entrenamientos, 0 submissions,
  0 correcciones de etiquetas, 0 push.**
- Escaneo byte a byte de lo generado: 0 bytes > 127.
- Hashes verificados: OOF `{hashes['oof_v03'][:16]}...`; submission `{hashes['submission_v03'][:16]}...`;
  muestra `{hashes['muestra'][:16]}...`; clusters `{hashes['clusters'][:16]}...`.

> V03 remains the best verified leaderboard result.
"""


def construir_filas_csv(evidencia: dict[str, Any]) -> list[dict[str, str]]:
    """Construye la tabla de decision en CSV."""
    alcance = evidencia["alcance"]
    return [
        {"componente": "clasificacion", "valor": "LOW_EXPECTED_VALUE",
         "detalle": "no se solicita autorizacion; TOXIC permanece cerrado (Caso B)",
         "fuente": "este informe"},
        {"componente": "material_preparado", "valor": "80 filas (40 TR=1 + 40 TR=0, todas HS=1)",
         "detalle": "3 columnas humanas vacias en las 80 filas (verificado); origen fase 2, semilla 42",
         "fuente": "reports/eda/tr_ejemplos_para_revision.csv"},
        {"componente": "integridad_material", "valor": f"muestra {alcance['muestra'][0]} dev + {alcance['muestra'][1]} holdout; limite {alcance['limite'][0]} + {alcance['limite'][1]}; contradictorios {alcance['contradictorios'][0]} + {alcance['contradictorios'][1]}",
         "detalle": "cualquier revision debe filtrar a dev; el holdout no se toca",
         "fuente": "data/processed/holdout_v2_ids.csv"},
        {"componente": "alcance_prioritario_dev", "valor": f"{len(evidencia['union_dev'])} filas (union sin duplicados)",
         "detalle": f"solapes en dev: muestra/contradictorios {evidencia['solapes_dev']['muestra_contradictorios']}; limite/contradictorios {evidencia['solapes_dev']['limite_contradictorios']}",
         "fuente": "este informe"},
        {"componente": "tasa_contradiccion", "valor": "HS 2.773 % / TR 2.773 % / AG 2.889 % del train",
         "detalle": "proporcion en clusters de texto casi duplicado con etiquetas inconsistentes; NO es tasa de error",
         "fuente": "reports/label_consistency_v4.json"},
        {"componente": "error_tr_dentro_fuera", "valor": "0.023 dentro vs 0.113 fuera (puntual)",
         "detalle": "los errores de TR no se concentran en las filas contradictorias demostrables",
         "fuente": "reports/hardness_vs_noise_v4.md"},
        {"componente": "intervenciones_medidas", "valor": "downweight +0.0005 / remove -0.0004",
         "detalle": "techo practico nulo de tocar las filas contradictorias",
         "fuente": "reports/advanced_research_v4.md"},
        {"componente": "potencia_del_proxy", "valor": "7.08 % de dev en grupos 2+; 43 filas informativas",
         "detalle": "V28-03 declara que no puede responder la pregunta; el proxy near-duplicate no basta",
         "fuente": "reports/v28_03_ambiguedad_etiquetas.md"},
        {"componente": "detector_de_ruido", "valor": "MARGINAL: AUC 0.973019 vs control 0.974143",
         "detalle": "el ruido no es separable de la dificultad; ponderar no tiene justificacion mecanica",
         "fuente": "reports/p59_confident_learning.md"},
        {"componente": "techo_de_intervencion", "valor": "42 celdas (0,95 %); oracle ~+0,009 < banda 0,017",
         "detalle": "ninguna intervencion de etiqueta alcanza la banda de deteccion",
         "fuente": "reports/p57_open_questions.csv"},
        {"componente": "reglas", "valor": "silencio normativo sobre corregir etiquetas de entrenamiento",
         "detalle": "test no usable para entrenamiento; correccion para reentrenar sin cobertura",
         "fuente": "docs/competition_rules.md"},
        {"componente": "autorizacion_requerida", "valor": "escrita del organizador (correccion + reentrenamiento)",
         "detalle": "la revision como analisis no la requiere; la correccion si; no se contacta a nadie",
         "fuente": "este informe"},
        {"componente": "evidencia_nueva_posible", "valor": "tasa directa de ambiguedad + kappa + cierre de D-005",
         "detalle": "valor documental; no cambia ninguna decision (familias cerradas y techo bajo)",
         "fuente": "este informe"},
        {"componente": "near_duplicados", "valor": "QUARANTINED; techo nulo / <= 0,002 LB",
         "detalle": "beneficio esperado inferior al de la revision de anotaciones; sin cambios",
         "fuente": "reports/toxic_remaining_legal_opportunities.md"},
        {"componente": "decision_final", "valor": "no se solicita autorizacion; TOXIC cerrado",
         "detalle": "Caso B de la directiva; sin revision, sin correccion, sin submissions, sin push",
         "fuente": "este informe"},
    ]


def escribir_informe(evidencia: dict[str, Any]) -> None:
    """Escribe el informe markdown."""
    RUTA_MD.write_text(construir_informe(evidencia), encoding="utf-8")


def escribir_csv(evidencia: dict[str, Any]) -> None:
    """Escribe la tabla de decision en CSV con terminador LF."""
    filas = construir_filas_csv(evidencia)
    with RUTA_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=["componente", "valor", "detalle", "fuente"], lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(filas)


def registrar_fila(evidencia: dict[str, Any]) -> bool:
    """Anade la fila P87-00 al registro una sola vez (append de una linea con LF)."""
    texto = RUTA_REGISTRO.read_text(encoding="utf-8-sig")
    lineas = texto.splitlines()
    encabezado = next(csv.reader(lineas))
    if encabezado != COLUMNAS_REGISTRO:
        raise SystemExit("STOP: las columnas del registro no coinciden con la fila P87-00")
    filas = list(csv.DictReader(lineas))
    if any(fila["experiment_id"] == "P87-00" for fila in filas):
        return False
    fila = {
        "experiment_id": "P87-00",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Que autorizacion haria falta para revisar (y en su caso corregir) etiquetas de TR, y merece la pena pedirla?",
        "hypothesis": "La evidencia existente acota el valor modelable de la revision por debajo de la banda de deteccion, y las familias modelables estan cerradas",
        "preprocessing": "ninguno (auditoria de solo lectura; no se revisa ni corrige ninguna etiqueta)",
        "feature_groups": "no aplica (auditoria)",
        "representation": "no aplica (auditoria)",
        "model": "no aplica (auditoria)",
        "hyperparameters": "no aplica (auditoria)",
        "validation": "material de revision (80 filas), particion congelada, V4-A, V28-03, P59-C, P57 y reglas",
        "f1": "",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "0 s (sin entrenamiento)",
        "dimension": "",
        "result": (
            "LOW_EXPECTED_VALUE: contradiccion demostrable 2.77-2.89 % del train (43 filas en dev); "
            "error de TR menor dentro de los contradictorios (0.023 vs 0.113); intervenciones +-0.0005; "
            "detector de ruido MARGINAL (0.9730 vs 0.9741); techo ~+0,009 < banda 0,017"
        ),
        "interpretation": "el material preparado incluye 10 filas del holdout (debe filtrarse a dev); la revision podria medir kappa y cerrar D-005, pero no cambia ninguna decision",
        "decision": "LOW_EXPECTED_VALUE: no se solicita autorizacion; no se revisa ni corrige; TOXIC permanece cerrado",
        "date": FECHA,
        "script": "experiments/p87_00_gobernanza_anotacion_tr.py",
        "notes": "gobernanza de la revision humana de anotaciones en TR; protocolo disenado, no ejecutado; 0 entrenamientos, 0 submissions, 0 push",
    }
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=COLUMNAS_REGISTRO, lineterminator="\n")
    escritor.writerow(fila)
    with RUTA_REGISTRO.open("ab") as archivo:
        archivo.write(buffer.getvalue().encode("utf-8"))
    return True


def main() -> int:
    """Ejecuta la auditoria de gobernanza completa."""
    log(f"fecha {FECHA} | auditoria de gobernanza, solo lectura")
    evidencia = cargar_evidencia()
    log(
        f"material verificado: 80 filas (3 columnas humanas vacias) | alcance dev "
        f"{len(evidencia['union_dev'])} prioritarias | holdout protegido"
    )
    escribir_informe(evidencia)
    escribir_csv(evidencia)
    anadida = registrar_fila(evidencia)
    log(f"informe y CSV escritos | fila P87-00 {'anadida' if anadida else 'ya presente'}")

    sucios = [
        ruta.name
        for ruta in (RUTA_MD, RUTA_CSV)
        if any(byte > 127 for byte in ruta.read_bytes())
    ]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    print(json.dumps({
        "clasificacion": "LOW_EXPECTED_VALUE",
        "revision_ejecutada": False,
        "etiquetas_corregidas": 0,
        "alcance_dev_prioritario": len(evidencia["union_dev"]),
        "entrenamientos": 0,
        "submissions": 0,
        "push": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

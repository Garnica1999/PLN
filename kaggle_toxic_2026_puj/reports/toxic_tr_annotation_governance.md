# TOXIC - P87-00: gobernanza de la revision humana de anotaciones en TR

Fecha: 2026-10-09. Script: `experiments/p87_00_gobernanza_anotacion_tr.py`.
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
  materiales. Filas **en el holdout**: 10 de la muestra, 8 de los
  casos limite y 7 de los clusters contradictorios. Cualquier revision debe
  **filtrar a dev** (muestra 70, limite 32, contradictorios 43);
  el holdout no se toca jamas.
- Alcance prioritario en dev (union de los tres materiales, sin duplicados):
  **140 filas**. Solapes en dev: muestra y contradictorios 2;
  limite y contradictorios 0.
- Opcion amplia: los **445 TR=1 de dev** (o los 648 HS=1), si se quisiera una
  tasa directa sobre toda la clase.

### 2.2 La estimacion historica del 2,8 por ciento (V4-A01)

`experiments/27_label_consistency_v4.py` (2026-10-02) agrupo el train con cuatro
criterios independientes (exacto, normalizado, jaccard >= 0.8, cosimilitud TF-IDF
>= 0.85): 0 / 4 / 15 / 13 clusters con 2+ miembros.
Resultado: 84 tweets en algun cluster (4.853 % del train) y
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
   Material prioritario: la union en dev (140 filas) mas una muestra de control
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
- **Datos**: solo dev (140 filas prioritarias o hasta 445 TR=1); el material
  filtrado a dev (70 + 32 + 43).
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

- **Casos revisables**: 140 filas prioritarias en dev (o 445 TR=1); las contradicciones
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
- Hashes verificados: OOF `559265816ce4f631...`; submission `7d7478f271e6edfd...`;
  muestra `48490e953073f6db...`; clusters `9ef10beac8fc9cdd...`.

> V03 remains the best verified leaderboard result.

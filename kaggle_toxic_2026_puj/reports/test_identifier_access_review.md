# P1 - Auditoria del acceso a identificadores de test (incidente P101)

Fecha: 2026-10-14. Alcance: solo lectura sobre artefactos y codigo existentes.
No se volvio a abrir `toxic_test`; no se leyeron sus textos, labels ni scores;
no se modifico ningun log ni artefacto existente.

## 1. Hechos verificados

- P101 informo: "toxic_test fue consultado unicamente para obtener 1.154
  identificadores, sin leer texto, labels ni scores".
- El codigo que ejecuto esa consulta es
  `experiments/p101_00_auditoria_procedencia.py`, funcion `ids_columna_cero()`
  (lineas 56-68): abre el CSV crudo en modo lectura y extrae SOLO el campo 0
  de cada linea (`linea.split(";", 1)[0]`), saltando la cabecera "id"; el
  resto de la linea nunca se parsea, retiene ni persiste.
- Uso (linea 98): `test_ids = ids_columna_cero(data/raw/toxic_test.csv)`; la
  misma funcion se uso sobre `toxic_train.csv` (linea 99).
- Finalidad registrada: construir la tabla de procedencia P0 de P101
  (1327 filas; `reports/p101_provenance_table.csv`, verificado: 1327 datos),
  con conteos por particion (ids de test: 1154) y verificacion de
  intersecciones de la muestra principal v2.1 con particiones reservadas:
  0 (esperado 0).
- `reports/p101_compliance.md` (linea 40) registra el acceso: "holdout2_ids.csv
  y toxic_test.csv (columna id; 1154 ids)".
- `experiments/p101_verificacion.log`: "chequeos: 60/60 | fallos: ninguno;
  verificador oficial: OK"; limites declarados: "toxic_test: consultado SOLO
  por columna de id (1154 ids); sin texto, labels ni scores".
- Ningun artefacto de P101 contiene texto, labels ni predicciones de test.
- Los ids no se usaron para excluir filas ni construir muestras de
  entrenamiento/validacion: la consulta es de verificacion de procedencia,
  posterior a la fase experimental (P101 es auditoria documental: 0 modelos,
  0 submissions).
- Impacto sobre decisiones experimentales: ninguno demostrable (ver seccion 4).

## 2. Archivos y funciones involucrados

| elemento | rol |
|---|---|
| experiments/p101_00_auditoria_procedencia.py (ids_columna_cero; linea 98) | unico acceso a ids de test en P101 |
| data/raw/toxic_test.csv | abierto en modo lectura; solo campo 0 procesado |
| reports/p101_provenance_table.csv / .md | destino de conteos e intersecciones |
| reports/p101_compliance.md, p101_provenance_audit.md, p101_99_verificacion.log | registro del alcance |
| src/data.py `read_test()` | cargador estandar (id+text) — NO fue el usado por P101 |

## 3. Proposito y uso de los identificadores

- Conteo de filas (1154) para la tabla de procedencia.
- Verificacion de interseccion 0 entre la muestra principal y las particiones
  reservadas (holdout / holdout2 / test).
- No se usaron para: excluir filas del entrenamiento, construir muestras,
  calibrar, seleccionar features, elegir modelos ni generar submissions.

## 4. Impacto experimental demostrable

- 0 intersecciones: nada que excluir; la verificacion es informativa.
- El id de coleccion no porta senal de etiqueta: P63-00 (STRUCTURED_IDENTIFIER)
  y P63-02 (F1 0.419 solo con id; control aleatorio 0.424; "CLOSED. El indice
  de coleccion no aporta senal especifica").
- La consulta ocurrio DESPUES del cierre experimental de P101 y no alimenta
  ningun modelo ni decision.
- Artefactos protegidos verificados intactos: V03 5/5 hashes congelados
  (`sha256sum -c` sin fallos); `toxic_train.csv` 205417 bytes con sha256
  3d40e734... identico al registrado (labels intactas).

## 5. Nota de alcance: otros accesos historicos a test (documentados)

Para no describir el test como "completamente intocado":

- Generacion de submissions: cada submission historica predice sobre el TEXTO
  de test (sin labels); es el uso previsto por la competencia. Varios scripts
  de submission usan `read_test()` (id+text) por necesidad de la prediccion.
- Grupos near-duplicados: `experiments/00_auditoria_datos.py` construyo
  `data/processed/grupos_near_duplicados.csv` con
  `clusters_near_duplicados(train, test, umbral 0.85)` — metodo historico de
  Fase 0, auditado en V29-01 ("El metodo anterior NO se cambia") y validado en
  V9-01: construir los grupos SOLO con train deja la referencia V03 identica
  (corrected_cv 0.777211, delta 0.000000) -> impacto nulo demostrado sobre la
  particion canonica.
- V4-A01: deteccion de 15 tuits de test con texto normalizado identico a
  train (diagnostico de duplicados, sin modificacion).
- V9-03: sonda de propagacion de etiquetas train->test con copias >= 0.85:
  generada y NO adoptada; 0 celdas cambiadas.
- La "prevalencia del test" del proyecto se estimo con SONDAS DE LEADERBOARD
  (submissions publicas, decision del propietario; 23 filas en
  `submissions/leaderboard_log.csv`), no leyendo el archivo de test.
- V47-00: "Holdout y test abiertos SOLO para contar filas y calcular hashes".

## 6. Incertidumbres

- La letra de la instruccion previa ("no utilizar el test durante el
  desarrollo y mantenerlo intacto") admite una lectura estricta: abrir el
  archivo para cualquier verificacion ya es una desviacion.
- La lectura es en streaming: los bytes de la linea pasan por memoria, aunque
  nada despues del primer campo se parsea, retiene ni persiste.
- No existe un registro de intencion mas alla de los documentos citados.

## 7. Interpretacion de la regla aplicable

- Regla escrita operativa (src/data.py, "Regla de oro"): prohibido USAR el
  test para entrenar, ajustar vocabularios, vectorizadores, umbrales o
  seleccion de features. El acceso observado no cae en ninguna categoria.
- Regla de la competencia segun la auditoria V11-02: prohibido el
  entrenamiento; la transduccion es ambigua; nada especifica sobre leer ids.
- Instruccion interna mas estricta: "mantener el test intacto". El archivo NO
  fue modificado (solo lectura; integridad verificable por hash), pero SI fue
  abierto, lo que no satisface la lectura mas estricta.
- Por tanto: no corresponde afirmar "cumplimiento absoluto"; corresponde
  registrar la desviacion de la letra con su impacto nulo demostrable.

## 8. Clasificacion del incidente

**POTENTIAL_SCOPE_DEVIATION**

- No es NO_BREACH_DEMONSTRATED: el archivo se consulto durante el desarrollo
  (aunque solo el campo id), contrario a la letra de la instruccion estricta.
- No es CONFIRMED_SCOPE_DEVIATION: no hay acceso demostrable a texto (mas alla
  del paso por memoria), labels, scores ni influencia en decisiones
  experimentales; los accesos interactuantes (near-dup) tienen auditoria y
  validacion de impacto nulo.
- La resolucion final corresponde a la aclaracion academica (ver
  `reports/methodology_permission_clarification.md`).

## 9. Medida preventiva futura

- Toda verificacion que necesite ids de test se hara, si se autoriza, con un
  lector de columna 0 o con la plantilla oficial de submission, jamas con el
  cargador completo, y se registrara en el documento de compliance.
- Mientras la pausa este vigente: el test no se abre para ningun proposito,
  ni siquiera conteos.

Sin test re-consultado; sin textos, labels ni scores leidos; sin modificacion
de logs ni artefactos.

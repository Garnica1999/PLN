# P92-S2 - Recepcion de la calibracion humana (4 anotadores) y analisis provisional

Fecha: 2026-10-12. **Version 2: incorpora la devolucion del anotador D; la version 1 de
este informe (3 anotadores) queda superada por esta.** Sin entrenamiento, sin modificacion
de etiquetas, sin test/holdout, sin submissions. **La muestra principal NO se ha
distribuido.**

## 1. Recepcion y almacenamiento

- **A y B**: completaron sus copias EN EL SITIO en los paquetes de calibracion (2026-10-12
  21:49). Copias byte a byte en `calibration_responses/calibration_response_A.csv` y `_B.csv`.
- **C**: recibido por mensaje de Carlos (2026-10-12); anotaciones y notas verbatim; texto
  materializado desde la fuente.
- **D**: recibido por mensaje de Carlos (2026-10-12); anotaciones y notas verbatim; los
  textos coinciden EXACTAMENTE con la fuente elegible (0 diferencias).
- Hashes (sha256[:16]): A `a20521989aa3fea9` | B `41e50c1f1d48760e` | C `8de76e21046d5091`
  | D `763229f9a9caa176`. Detalle en `calibration_responses/PROVENANCE.md`.

## 2. Validacion (protocolo de recepcion)

| anotador | schema | 15 filas | review_ids | textos | version | respuestas permitidas | desviaciones |
|---|---|---|---|---|---|---|---|
| A | OK | OK | OK | OK | OK | NO: 0/1 | BOM de Excel; notas 0/1; sin UNABLE_TO_DETERMINE |
| B | OK | OK | OK | OK | OK | NO: 0/1 | BOM de Excel; notas 0/1; sin UNABLE_TO_DETERMINE |
| C | OK | OK | OK | OK | OK | OK | conforme; UTD en HS 7/15 |
| D | OK | OK | OK | OK | OK | OK | conforme; textos pegados identicos a la fuente (0 diferencias); UTD en HS 8/15 |

Los textos de las 60 filas (15 x 4 anotadores) coinciden con la fuente elegible. La muestra
principal sigue intacta.

## 3. Desviaciones y estado

- A/B usaron 0/1 en lugar de los tokens (analisis bajo el supuesto **0=NEGATIVE,
  1=POSITIVE**, PENDIENTE DE CONFIRMACION) y no usaron `UNABLE_TO_DETERMINE`; su columna
  `uncertainty_notes` contiene 0/1 (A: filas 02, 04; B: filas 01, 05, 13).
- C y D usaron el formato completo, con `UNABLE_TO_DETERMINE` y notas ricas.

## 4. Acuerdo entre anotadores (provisional)

| etiqueta | A-B | A-C | A-D | B-C | B-D | C-D | cuadruple | alpha (UTD=categoria) | alpha (UTD=faltante) | UTD |
|---|---|---|---|---|---|---|---|---|---|---|
| HS | 10/15 | 5/15 | 6/15 | 6/15 | 6/15 | 12/15 | 4/15 | 0.243 | 0.466 | C=7 D=8 |
| TR | 11/15 | 9/15 | 9/15 | 11/15 | 11/15 | 14/15 | 8/15 | 0.104 | 0.093 | C=1 D=0 |
| AG | 12/15 | 11/15 | 8/15 | 10/15 | 7/15 | 11/15 | 6/15 | 0.354 | 0.491 | C=1 D=3 |

- Combinacion exacta HS/TR/AG entre los cuatro: **1/15**.
- **Hallazgo central: C y D convergen entre si** (HS 12/15, TR 14/15, AG 11/15) y ambos
  marcan `UNABLE_TO_DETERMINE` en HS en las MISMAS 6 filas: P92C-05, P92C-09, P92C-11, P92C-13, P92C-14, P92C-15.
- A y B (binarios, sin UTD) concuerdan entre si mas que con C/D; la mayor parte de la
  brecha la explica el uso sistematico de UTD por C/D.
- La ambiguedad es de APLICACION DE LA GUIA (odio vs ofensa en insultos individuales),
  no un desacuerdo aleatorio: es exactamente lo que P91 anticipo (definiciones oficiales
  sin criterios operativos).

## 5. Contexto: coincidencia con la etiqueta original (diagnostico, no prevalencia)

A 28/45 | B 28/45 | C 25/45 | D (por calcular en la tabla de acuerdo; las celdas UTD no
coinciden con la etiqueta binaria). La calibracion NO estima tasas poblacionales.

## 6. Ambiguedades registradas (material para la discusion humana)

- **Odio vs ofensa** (insulto sexista individual): C la cita en 6 filas (04, 05, 09, 11,
  14, 15); D la cita en 6 filas (03, 04, 05, 09, 11, 13, 14, 15 con variantes); 6 filas
  reciben UTD de ambos en HS (05, 09, 11, 13, 14, 15).
- **Agresivo vs vulgar**: fila 13 (C y D) y contexto coloquial (08).
- **Contexto insuficiente / referencias no identificables**: filas 01, 07 (C y D).
- Filas 10 y 12 (insulto por identidad trans; insulto racista) sin duda: POSITIVE en las
  cuatro anotaciones relevantes.

## 7. Decisiones pendientes (para Carlos)

1. Confirmar la codificacion de A/B: **0=NEGATIVE, 1=POSITIVE** (y ausencia de UTD).
2. Aclarar el significado del 0/1 en `uncertainty_notes` de A/B.
3. **Discusion humana de calibracion** (paso entre anotadores): con C y D el material es
   inequivoco. Decision sobre la guia: mantener v1 o preparar **v2 con changelog** antes de
   la ronda principal; recomendacion preliminar: v2 que (a) exija los tres tokens,
   (b) explicite que UNABLE_TO_DETERMINE es esperado y valido ante la duda normativa
   odio vs ofensa, (c) documente el caso A/B (0/1) como ejemplo de formato no valido.
4. Formato exigido en la ronda principal (tokens + UTD; sin 0/1).

La muestra principal (250 tweets) permanece **sin distribuir** hasta cerrar 3 y 4.

## 8. Trazabilidad

- Respuestas y hashes: seccion 1; acuerdo por celda: `reports/p92s2_calibration_agreement.csv`
  (incluye C y D); metricas: `reports/p92s2_calibration_results.json` (v2).
- Verificador oficial y Git: ver cierre de la respuesta.

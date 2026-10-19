# P94 - Reproducibilidad y cumplimiento

Fecha: 2026-10-13. Fase de investigacion y diseno: sin entrenamiento, sin
predicciones, sin submissions, sin acceso a test, holdout, holdout2 ni leaderboard.

## 1. Alcance y metodo

- Auditoria de solo lectura sobre el arbol real del proyecto (verificado antes de
  asumir rutas: no se presupuso ningun nombre de fase ni de archivo).
- Herramientas: lectura de documentos, parseo programatico de `experiments.csv`
  (459 filas reales), busquedas en fuentes primarias (ACL Anthology, arXiv, PMC,
  PMLR, IEEE, ScienceDirect, CEUR) y verificacion por scripts propios.
- No se re-ejecuto ningun experimento historico para completar el inventario.
- Generacion de artefactos: `experiments/p94_00_inventario.py` (inventario con
  verificacion de 95 anclas contra el registro) y
  `experiments/p94_01_registro_hipotesis.py` (registro P2/P3 con validacion de
  vocabulario y de campos); `experiments/p94_02_verificacion.py` (checks de cierre,
  hashes y estado de Git). Todos usan el interprete del proyecto
  (`/home/carlos/venv/bin/python`).

## 2. Fuentes internas leidas (y su profundidad)

| Fuente | Profundidad real de lectura |
|---|---|
| PROJECT_STATE.md | completa (987 lineas; fases P77-P82 y estado reciente) |
| experiments.csv | completa por parseo programatico (459 filas, 23 columnas; digest integro + filas puntuales) |
| DECISIONS.md | estructura completa (370 encabezados D-001..D-295) + lectura de los bloques relevantes al inventario (P81-P90, D-281..D-295, D-191..D-217, D-100..D-116, D-169..D-190, P59-P68, v12..v15) + barridos dirigidos por palabra clave (negacion, F1 suave, kernel, bagging, semi-supervisado, bayes, etc.) |
| RUNLOG.md | entradas clave (P84 a P90) y barridos dirigidos |
| FINAL_REPORT.md | estructura completa + lectura de las secciones 14-27 (modelos clasicos, DL, validacion, experimentos, ablacion, error, modelo final, submission, limitaciones, conclusiones, trabajo futuro) |
| reports/ (P84-P93) | los informes citados por las decisiones y por el inventario (auditorias P85-P90, programa de anotacion P91-P93) |
| docs/ | p83_AVISO_NO_EJECUTABLE.md (estado resuelto del bayesiano) |

Nota de honestidad de metodo: DECISIONS.md (10.392 lineas), RUNLOG.md (5.156) y
FINAL_REPORT.md (2.775) no se leyeron linea a linea en su totalidad; se leyo su
estructura completa y todos los bloques que gobiernan las 24 familias del inventario,
y cada familia quedo anclada y verificada contra las 459 filas del registro (95
anclas). Las afirmaciones del inventario son trazables a filas o actas concretas.

## 3. Recursos reservados: cero accesos

- test, holdout historico y holdout2: 0 lecturas en esta fase (ni siquiera de
  identificadores; no se requerian).
- leaderboard: 0 consultas nuevas; la referencia 0.78306 procede del registro
  interno de sondas del proyecto, no de una consulta nueva.
- Sin entrenamiento, sin predicciones nuevas, sin pseudo-labeling, sin recursos
  externos (ni corpus, ni etiquetas, ni embeddings, ni clasificadores), sin
  descargas de modelos; las busquedas web se limitaron a metadatos de literatura
  (arXiv/PMC/ACL/CEUR/ScienceDirect/PMLR).
- Sin exposicion de secretos: ningun artefacto generado contiene credenciales ni
  rutas sensibles; no se generaron envios ni se toco `submissions/`.

## 4. Checklist de auditoria final (resultados reales)

| Comprobacion | Resultado |
|---|---|
| V03 unchanged (modelo + submission) | OK: 263ed5bd0fc0de7c / 7d7478f271e6edfd |
| canonical OOF unchanged | OK: 559265816ce4f631 |
| original labels unchanged (train) | OK: 3d40e7345b588860 |
| particion canonica unchanged | OK: e5b62e5f22c4c63a |
| holdout2 unchanged | OK: f6c6ed958abf728b |
| respuestas C1/C2 (rondas 1 y 2) preservadas | OK: f89f74cdabd9649d / e0449a811fa8b64b / 91911ab70f6d70cc / 5da8fc1b33135489 |
| decisiones finales P93-R4 preservadas | OK: 98e4264b0fb98b7a |
| artefactos P93-R5 preservados | OK: cc7ab96b90155980 / a884cf91467aa3ea / 107355c774b15d8e |
| test not accessed | OK (0 lecturas) |
| holdout / holdout2 not accessed | OK (0 lecturas) |
| leaderboard unused | OK (0 consultas) |
| no model training | OK (fase documental; 0 entrenamientos) |
| no predictions generated | OK |
| no external labeled data / pretrained resources | OK |
| no submissions | OK |
| no secrets exposed | OK |
| ASCII-only generated artifacts | OK: verificador oficial en 0; escaneo byte a byte en 0 |
| all new artifacts hashed | OK (seccion 5) |
| Git changes preserved | OK: sin commit, sin push, sin reset (solo lectura) |

Verificaciones de cierre (comando real):
`tools/verificar_caracteres.py` -> salida 0, 0 caracteres no permitidos.
`git status --short` -> 1 archivo modificado (solo el notebook consolidado, de
fases previas) y el resto sin rastrear; `git diff --stat` -> solo el notebook.
No se hizo commit ni push.

## 5. Hashes completos (sha256)

Entradas canonicas y de contexto (verificadas intactas):
- data/raw/toxic_train.csv: 3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f
- features/oof_v03_final.npz: 559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab
- experiments/baseline_v03_frozen/submission_v03.csv: 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
- experiments/baseline_v03_frozen/model_v03.joblib: 263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca
- data/processed/holdout_v2_ids.csv: e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd

Salidas P94 (hashes de la corrida de verificacion; el hash de este propio
documento queda registrado en la seccion 154 del notebook, y el del notebook en el
mensaje de cierre de la sesion):
- reports/p94_experiment_inventory.csv: 11f376e1d4c1545cf37ade285f7d3be79eb14df7166ade3e787431f2b3a24d60
- reports/p94_sota_review.md: 245cf119df408b4b8c90c566ab64403e72fe6f1a0e4194d96d6acd7ec7a75f90
- reports/p94_hypothesis_registry.csv: 17917ce2a1a132d9c64d2eb62920d540fb86b3ea23b1dee4642d4b20adc6a6bb
- reports/p94_recommended_experiment.md: 7a09af6710b863afe2307664e70ab8c3e236e9c67f1f3b0626591a237a809441
- experiments/p94_00_inventario.py: 65ad5b3087390d0e1066631a20dbceb13567bb487a8d165dc7e0b54f34e39274
- experiments/p94_01_registro_hipotesis.py: 26ef44683ffbc3660c21f08cd5eec3aab694a34bce929d09f04824cfaacce8f6
- experiments/p94_02_verificacion.py: f347ea66e39f6252d5c963124db5041839926ca49a1957c8295893b0494a4794

## 6. Hallazgos de la auditoria de entrada (P0)

- El registro tiene 459 filas (2026-09-22 a 2026-10-09). Los valores de la columna
  `f1` pertenecen a protocolos de fases distintas (la propia D-128 lo documenta);
  el inventario los cita solo de forma cualitativa y la cifra vinculante es la
  definicion canonica de corrected_cv de V03.
- P83 (modelo bayesiano disperso de doble objetivo) existe, se ejecuto en completo
  y su aviso "NO EJECUTABLE" quedo superado el mismo 2026-10-02 (docs/p83_AVISO_
  NO_EJECUTABLE.md: "EJECUTADO EN COMPLETO"); la fila fue anadida al registro por la
  correccion de la auditoria integral. Queda inventariado como F21 y cerrado.
- P82-02 (sonda de submission de la RNN) y P89-00/P90-00 forman parte de las fases
  de auditoria posteriores al cierre competitivo; sin efecto sobre V03.
- No se encontraron familias del espacio permitido cuyo estado difiera del
  registrado en las actas; las 95 anclas del inventario verifican contra el registro.

## 7. Limitaciones de esta fase

- Es una auditoria documental: su fuerza depende de la fidelidad del registro y de
  las actas (gobernado por P89-00 y D-294, con correcciones registradas).
- La lectura de DECISIONS/RUNLOG/FINAL_REPORT fue dirigida (ver seccion 2); no se
  excluye que exista detalle fuera de los bloques leidos, mitigado por el anclaje
  de las 24 familias contra el registro y por los barridos por palabra clave.
- La busqueda bibliografica se cerro conforme a la regla de P1 al no cambiar el
  conjunto de hipotesis viables; es un cierre por suficiencia de decision, no una
  revision sistematica exhaustiva de toda la literatura de odio en espanol.

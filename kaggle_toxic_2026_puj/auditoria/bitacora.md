# Bitacora de la auditoria integral

Registro cronologico de fases, agentes y resultados. Este fichero lo actualiza
el lider al cerrar cada pieza y cada subagente deja aqui su resumen (maximo
20 lineas).

## 2026-10-02

- F0 completada: estado sucio de P67 (P82 y P83, 187 archivos) asegurado en
  el commit `1026ff1`. Verificado que el verificador oficial admite acentos y
  enie (solo persigue alfabetos no latinos).
- F1 completada: commit `9663afb`. Importados 67 archivos unicos de la rama
  de auditoria v8 y 85 artefactos del rerun (16 submissions regeneradas
  incluidas); fuente de fastText (commit `1142dc4`) copiada a `tools/fastText`
  sin `build/`. Manifiesto con 154 filas. Verificador de caracteres en
  0 errores. Sin conflictos ni sobrescrituras.
- F2 completada: commit `41f8a3f`. Inventario unificado: 483 experimentos
  (411 de registro mas 72 scripts sin fila), 376 scripts clasificados, 43
  filas malformadas confirmadas (P67 a P77), 277 experimentos sin documentar
  en el notebook y P78 a P83 sin seccion propia.
- F3a completada: escaner estatico AST sobre 376 scripts
  (`auditoria/escaner_estatico.py`), 2.165 hallazgos (54 ALTA, 319 MEDIA,
  342 BAJA, 1.450 INFO). D1 (funciones probadas sin llamador real) da 154
  casos: 54 sin ningun llamador y 100 solo desde humo. El caso historico
  `decidir_alpha` ya aparece conectado, lo que valida el detector contra el
  fallo conocido; `con_extra` de P83 sigue sin llamador. D2 (aserciones
  vacias) da 0 con el detector implementado. Sensibilidad probada con
  archivo plantado en /tmp/opencode.
- F3b completada: revision adversarial del nucleo (`auditoria/revision_nucleo.md`).
  6 de 7 afirmaciones CONFIRMADAS, incluida la recomputacion exacta de
  `0.7772114525165721` y el +0.0036286 de la rejilla de umbrales sobre el
  mismo OOF. REFUTADA parcialmente la afirmacion 5: `v33_01_validacion.py`
  evaluo el MISMO holdout v2 el 2026-09-27 sin registrarlo, de modo que el
  consumo real es 6 de 5; el registro de 5/5 solo cuenta lo declarado.
  Hallazgos nuevos: refits de submission entrenan con las 1731 filas
  (incluye holdout; es el diseno de refit final, se documenta), y el
  `leaderboard_log.csv` no es UTF-8 (cp1252). D2=0 sostenido con AST
  independiente (64 asserts, 0 tautologicos).
- F3c completada: revision P69-P77 (`auditoria/revision_p69_p77.md`). Ocho
  conclusiones verificadas. Hallazgo CRITICO: el delta de P72-03 se publica
  contra el maximo de las nulas, no contra la media; A queda por encima de la
  media (+0.0105) y solo 1 de 10 nulas lo supera. Hallazgos: P69 es
  re-scoring, no reentrenamiento; P73 sostiene el signo pero no la magnitud.
- F3d completada parcialmente: revision P78-P81
  (`auditoria/revision_p78_p81.md`). P78 confirmado (E* real 8-17, no 10-17);
  P79 con el recuento del scheduler no confirmado; P80 "19/19" impreciso
  (18 mas 1 no aplicable) y la frase del pool de CV mezclaba fases.
- F3d2 completada: revision P82 y su sonda (`auditoria/revision_p82.md`).
  Conteo 23/23 exacto pero A7 contaminada: el determinismo "bit a bit" compara
  dos pasadas de la misma corrida y declara igualdad con NaN y
  `equal_nan=True` sobre mascaras. H3: el submission recalcula E* con
  pliegues internos y no replica el pipeline de B1. H4: el MD de la sonda
  niega un envio que el registro si tiene (LB 0.68150).

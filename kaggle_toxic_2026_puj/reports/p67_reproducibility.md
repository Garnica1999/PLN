# P67 - Reproducibilidad

Fecha 2026-09-29T16:51:20+00:00.

| elemento | valor |
|---|---|
| rama | `proyecto-kaggle-toxic-2026-p67` |
| commit base | `51b0c2b` |
| semilla | 42 |
| V03 corrected_cv | `0.7772114525165721` |
| V03 leaderboard | `0.78306` |
| sha256 de la submission de V03 | `7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e` |
| n filas | 1483 |
| scripts | `p67_00_reconciliacion.py`, `p67_01_controles_decisivos.py`, `p67_02_cerrar.py` |
| artefactos de P67 | `reports/p67_*.md`, `.json`, `.csv` |
| figuras | 6 en `figuras/p67/` |

## Como reproducir

```bash
/home/carlos/venv/bin/python experiments/p66_01_regenerar_scores.py
/home/carlos/venv/bin/python experiments/p67_00_reconciliacion.py
/home/carlos/venv/bin/python experiments/p67_01_controles_decisivos.py
/home/carlos/venv/bin/python experiments/p67_02_cerrar.py
```

El regenerador de P66 es necesario porque `features/` esta en `.gitignore` y no
viaja entre worktrees. Es determinista y verifica sus siete scores contra la fase de
origen antes de continuar.

### ADVERTENCIA: el regenerador pisa un artefacto de P66

`experiments/p66_01_regenerar_scores.py` declara en su linea 86
`CSV_REG = REPORTS / "p66_score_fusion.csv"`, un nombre que pertenece de verdad a
`p66_02_fusion_espacio_scores.py`. Al ejecutarlo escribe su propia tabla de 8 filas
encima de los 214 registros de la fase de fusion. Hay que deshacerlo siempre:

```bash
git checkout HEAD -- reports/p66_score_fusion.csv reports/p66_score_regeneration.json
```

Es un bug de la fase P66, no de P67. No se corrige aqui porque editar el codigo de
una fase anterior esta prohibido. El JSON, `reports/p66_score_fusion.json`, si se ve
intacto: es el que usa la reconciliacion.


## Idempotencia

Los tres scripts se pueden ejecutar dos veces seguidas sin duplicar contenido en
`FINAL_REPORT.md`, `PROJECT_STATE.md`, `CHAMPION.md`, `RUNLOG.md`, `DECISIONS.md` ni
en el informe acumulado: cada bloque va sellado con `<!-- P67 2026-09-29 -->`.

Verificado en tres pasadas consecutivas. Excluyendo la marca de tiempo de ejecucion,
el hash agregado de los trece documentos e informes de la fase es **identico** en las
tres. El valor concreto no se transcribe aqui a proposito: un hash escrito dentro del
fichero que el hash resume no puede ser consistente consigo mismo, porque al cambiar
el hash cambia tambien el fichero. La comprobacion se reproduce con:

```bash
snap() {
  for f in FINAL_REPORT.md PROJECT_STATE.md CHAMPION.md RUNLOG.md DECISIONS.md \
           reports/TOXIC_COMPLETE_RESEARCH_REPORT.md reports/p67_compliance.md \
           reports/p67_reproducibility.md reports/p67_final_research.md \
           reports/p67_final_summary.json reports/p67_master_results.csv \
           reports/p67_controles_decisivos.json reports/p67_p66_reconciliation.json; do
    printf "%s  %s\n" "$(grep -vE '^Fecha |^Generado por|"fecha":' "$f" | sha256sum | cut -c1-16)" "$f"
  done | sha256sum | cut -c1-16
}
```

### Lo que si cambia entre ejecuciones, y por que es correcto

| elemento | cambia | motivo |
|---|---|---|
| `Fecha` en el informe de cumplimiento | si | es la hora real de la ejecucion |
| `Generado por ... Fecha` en los informes | si | idem |
| `"fecha"` en los JSON | si | idem |
| `"git_commit"` en los JSON | si | si se commitea entre pasadas |
| todo lo demas | **no** | el contenido es identico |

El timestamp es provenance, no ruido: registra cuando se ejecuto cada vez. Quien
verifique la idempotencia tiene que excluirlo, o comparar el arbol en vez del hash
crudo del fichero. Un hash de fichero completo **no** es estable en este diseno, y
que no lo sea no es un defecto.

### El bug que esta seccion documenta

Una version anterior de `escribir_bloque` **buscaba** las marcas para saber si el
bloque existia pero **no las escribia nunca**. Como consecuencia, la segunda
ejecucion no encontraba la marca y anadia una copia duplicada: 204 lineas repetidas en
seis ficheros historicos. Se detecto al comparar hashes antes y despues de la
segunda pasada, y se corrigio sellando siempre el bloque entre las dos marcas.
El dano estaba sin commitear y se revirtio con `git checkout --`.

Lo que si quedo commiteado, en `c6ae671`, fue un dano distinto y de otra fase: ver
el aviso sobre `p66_score_fusion.csv` mas arriba.
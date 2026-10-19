# P101 - Reproducibilidad

Fecha: 2026-10-14. Sin entrenamiento; sin anotacion; solo lectura sobre
artefactos historicos.

## 1. Entorno

- Python: /home/carlos/venv/bin/python (numpy; sin GPU).
- Regla de acceso: particiones reservadas SOLO por columna de id.

## 2. Scripts

- experiments/p101_00_auditoria_procedencia.py: regenera los entregables de
  P0 (guarda de no sobrescritura; para regenerar, borrar antes los 3
  archivos de reports/p101_provenance_* y p101_materials_inventory.csv).
- experiments/p101_90_registrar.py: fila P101-00 en experiments.csv
  (idempotente).
- experiments/p101_95_integrar_notebook.py: seccion 164 del notebook
  consolidado (idempotente; celda de verificacion en aislamiento).
- experiments/p101_99_verificacion.py: verificacion de cierre.

## 3. Entradas (hashes en reports/p101_materials_inventory.csv)

- reports/p92w_internal_sampling_key.csv, p92_internal_sampling_key.csv,
  v03_error_review_sample.csv, p91_label_review_candidates.csv,
  p93r3_round2_id_assignment.csv; artifacts/human_annotation_*; docs/*.

## 4. Comandos

    cd /home/carlos/Git/PLN/kaggle_toxic_2026_puj
    /home/carlos/venv/bin/python experiments/p101_00_auditoria_procedencia.py
    /home/carlos/venv/bin/python experiments/p101_90_registrar.py
    /home/carlos/venv/bin/python experiments/p101_95_integrar_notebook.py
    /home/carlos/venv/bin/python experiments/p101_99_verificacion.py
    /home/carlos/venv/bin/python tools/verificar_caracteres.py

## 5. Nota

- La regeneracion requiere los mismos artefactos historicos (hashes
  verificados); no se accede a contenido de particiones reservadas.
- No se ejecuto ningun entrenamiento ni anotacion en P101.

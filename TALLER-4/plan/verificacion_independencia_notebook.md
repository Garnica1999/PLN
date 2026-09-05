# Verificación de independencia y robustez del notebook (v3)

Fecha: 2026-09-05. Alcance: `notebooks/Taller_4_IR_CEV.ipynb` (ensamblado
desde `notebook_cells/`) y su ejecución en Colab (resolución de raíz +
regeneración end-to-end opcional). Evidencia: `tests/test_notebook_dependencias.py`.

## Escaneo de independencia
- Imports permitidos: solo paquetes (pandas, sklearn, spacy, wordcloud,
  matplotlib, gdown, pymupdf...). Ninguna celda code importa `scripts/`,
  `experimentos/`, `importlib` (uso real: import_module/spec_from_file) ni
  muta `sys.path` hacia esas carpetas; los docstrings solo las MENCIONAN.
- Referencias de archivo en celdas code: cada literal .jsonl/.txt/.csv/.json/.png
  es un archivo versionado en git (`git ls-files`) o está en la lista blanca
  documentada {entrevistas_all_2023-03-21_14-24_05.json,
  entrevistas_normalizadas.jsonl, entrevistas_tokens.txt, paginas.jsonl}
  (>100 MB o intermedios regenerables) con guard `.exists()` en la misma celda.
- Sin `../imagenes/` en celdas markdown (no renderiza en GitHub/Colab).
- Resolución de raíz: celdas `buscar_raiz()`/`_raiz_standalone()` lanzan
  `FileNotFoundError` claro con el marcador `segmentos_libros.jsonl`; no
  existe la asignación silenciosa del último candidato (nunca cae a `/`).

## Checks (a)-(f) con pytest
`/home/carlos/venv/bin/python -m pytest tests/test_notebook_dependencias.py`
→ **6/6 passed** (0.04 s):
(a) sin imports de scripts/experimentos/importlib; (b) sin `../imagenes/`;
(c) literales versionados o lista blanca con guard; (d) réplicas con error
claro y raise; (e) sin marcadores `[embed:` pendientes, `CODIGO_EMBED`
presente y `REGEN_AUTOMATICO = False`; (f) celda de diagnóstico de
prerrequisitos presente.

## Nota del embebido (fuente única en build)
`scripts/ensamblar_notebook_t4.py` reemplaza en el ensamblado la línea
`CODIGO_EMBED_XX = None  # [embed: scripts/0X_*.py]` por el contenido literal
del script (celdas de la Sec 0b; 3 marcadores resueltos, error si la ruta no
existe o queda un marcador pendiente). El notebook NUNCA importa los
scripts: los ejecuta con `exec()` en un namespace propio (con `__file__`
sintético y `sys.argv` parcheado en 01) solo si el usuario activa los flags
de regeneración; el ensamblado no ejecuta el contenido incrustado.

## Simulaciones de resolución de raíz (en /tmp/opencode, layouts sintéticos)
Ejecutadas las celdas reales de sec00 (4/4 casos OK): repo descomprimido en
`/content/PLN-main/TALLER-4` con cwd=/content → resuelve por walk ≤3 niveles;
`/content/TALLER-4` directo → resuelve; árbol vacío → FileNotFoundError con
marcador y opciones; `TALLER4_ROOT` → resuelve directo (sin datos locales).

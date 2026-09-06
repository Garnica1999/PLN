# Verificación de independencia y robustez del notebook (v4)

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
- La v4 eliminó el embebido como cadena: el ipynb no contiene `CODIGO_EMBED_*`
  ni llamadas `exec(` (verificación del ensamblador y check (e1) del test).

## Checks (a)-(f) con pytest
`/home/carlos/venv/bin/python -m pytest tests/ -q`
→ **49/49 passed** (6.84 s; 43 del resto de la suite + 6 de este archivo).
`/home/carlos/venv/bin/python -m pytest tests/test_notebook_dependencias.py`
→ **6/6 passed**:
(a) sin imports de scripts/experimentos/importlib; (b) sin `../imagenes/`;
(c) literales versionados o lista blanca con guard `.exists()` (por celda
fuera de bloques; a nivel de bloque expandido `[auto]` para los scripts
partidos 01-03); (d) réplicas con error claro y raise; (e) expansión v4
completa: (e1) sin `CODIGO_EMBED_` ni `exec(`, (e2) celdas code con el
código de scripts/01-03 (distintivos `ORDEN_TOMOS`,
`def construir_nlp`/`cargar_stopwords_habla`, `def etapa_estadisticas`),
(e3) aliases `main_extraccion`/`main_preprocesar`/`main_eda`, (e4)
`REGEN_AUTOMATICO = False`, (e5) sin marcadores `[expand:` ni `[embed:`
pendientes; (f) celda de diagnóstico de prerrequisitos presente.

Ejecución nbclient (v4, kernel del venv, timeout 600 s/celda): 0 errores en
40.8 s local (v3: ~29 s) y 12 outputs image/png inline; verificación de no
escritura: snapshot de `config/`, `data/corpus`, `data/raw`, `resultados/`
e `imagenes/` antes/después con 0 cambios; auto-checks de paridad del
notebook en True (TESTIMONIAL 76.77 %, variantes 54,8/84,6/96,6).

## Nota de la expansión (fuente única en build)
`scripts/ensamblar_notebook_t4.py` convierte en el ensamblado la celda
marcador `# %% [expand: scripts/0X_*.py]` (una celda code de la Sec 0b por
script; 3 marcadores resueltos, error si la ruta no existe o queda un
marcador pendiente) en **celdas de código normales**: elimina el bloque
`if __name__ == "__main__":` (en el kernel de Jupyter `__name__` es
"__main__" y el guard ejecutaría el pipeline), antepone en la primera celda
un comentario `[auto]` con `__file__` sintético (apuntando a la raíz
detectada por sec00/sec0b, para que `parents[1]` resuelva bien), parte el
script por sus banners de sección (nunca dentro de docstrings, vía tokenize)
y cierra el bloque con la celda de alias `[auto]` (`main_extraccion`,
`main_preprocesar`, `main_eda`; para 03 además restaura el backend inline de
matplotlib). En Run all las celdas expandidas solo DEFINEN funciones y
constantes; la regeneración real la disparan los wrappers de la sección solo
si faltan archivos y el usuario activa los flags. El notebook NUNCA importa
ni ejecuta dinámicamente los scripts.

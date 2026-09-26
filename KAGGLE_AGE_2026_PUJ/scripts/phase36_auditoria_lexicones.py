"""FASE 36 (P0) — Auditoría histórica y cobertura de hipótesis para la fase de lexicones.

Responde, **antes** de descargar o construir nada:

* qué familias de señal léxica ya están en V12 y con qué evidencia;
* cuáles no se han probado nunca;
* qué experimentos históricos podrían solaparse con la propuesta;
* qué recursos externos serían admisibles bajo §1 y cuáles se descartan;
* qué hipótesis quedan abiertas y con qué criterio se cerrarán.

Salida: ``reports/age_lexicon_hypothesis_coverage.md`` y ``experiments/phase36_cobertura_lexicones.json``.

La auditoría es verificable: cada fila declara la evidencia con ruta y línea, y el script comprueba
por ``grep`` que ninguna familia declarada como "nunca probada" aparece en el historial.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

SALIDA_MD = cfg.REPORTS / "age_lexicon_hypothesis_coverage.md"
SALIDA_JSON = cfg.EXPERIMENTS / "phase36_cobertura_lexicones.json"

#: Familias de señal léxica y su cobertura real en el proyecto, con la evidencia que lo demuestra.
FAMILIAS = [
    {
        "id": "L1",
        "familia": "Lista léxica cerrada general (stopwords)",
        "estado": "EXISTE (mínima)",
        "que_hay": "SPANISH_STOPWORDS: ~140 entradas autoelaboradas en el propio proyecto. Alimenta "
                   "dos features: stopword_ratio y content_ratio.",
        "evidencia": "src/textometry.py:38-49 (frozenset, comentario 'recurso léxico cerrado, sin "
                     "parámetros aprendidos'); src/feature_catalog.py:43-44; "
                     "features/dense_train.parquet (columnas stopword_ratio, content_ratio)",
        "implicacion": "Precedente aceptado: una lista fija autoelaborada es admisible. Ampliar de 1 "
                       "lista a 24 subcategorías es genuinamente nuevo, pero el listón de "
                       "redundancia es real: ~140 palabras funcionales cubren el de cualquier "
                       "texto español.",
    },
    {
        "id": "L2",
        "familia": "Categorías temáticas de dominio",
        "estado": "NUNCA PROBADA como canal de features",
        "que_hay": "Solo evidencia **observacional**: el Gate A registra que el léxico temático "
                   "diferencia (fandoms/MTV en 13-17 frente a política/noticias en 65-xx) y un "
                   "análisis de términos por clase. Adicionalmente el contenido entra por "
                   "TF-IDF word(1-2) y char_wb(3-5) de los 14 miembros, token a token.",
        "evidencia": "PROJECT_STATE.md (Gate A, punto 3 'Señal confirmada'); "
                     "reports/tables/top_terms_by_class.csv; experiments.csv (fase 3, análisis "
                     "lingüístico)",
        "implicacion": "La hipótesis fue confirmada por observación pero **nunca explotada** como "
                       "agregado de baja dimensión. Esa es la brecha exacta que ataca esta fase.",
    },
    {
        "id": "L3",
        "familia": "Registro / informalidad / slang",
        "estado": "NUNCA PROBADA como lexicon; parcialmente implícita",
        "que_hay": "TF-IDF char_wb(3-5) captura la forma ortográfica informal y las "
                   "abreviaciones; las features de forma ya cubren mayúsculas, alargamientos, "
                   "emoticonos y signos de exclamación/interrogación.",
        "evidencia": "features/age_structured_features.parquet (shape_allcaps_ratio, "
                     "shape_repeated_letter_token_ratio, punct_excl_density, punct_quest_density); "
                     "features/dense_train.parquet (elongated_count, n_emoticon, n_excl, n_quest)",
        "implicacion": "Máximo riesgo de redundancia de las tres familias: es la que más solapa con "
                       "los n-gramas de carácter. Por eso la familia A es la primera en cribarse.",
    },
    {
        "id": "L4",
        "familia": "Etapas de vida / contexto social",
        "estado": "NUNCA PROBADA",
        "que_hay": "Nada. Solo señales de longitud y estructura social indirectas (RT, menciones).",
        "evidencia": "reports/age_hypothesis_coverage.md (no figura ninguna hipótesis léxica de "
                     "contexto vital); experiments.csv: 0 experimentos con 'lexicon|wordlist'",
        "implicacion": "La familia con más superficie nueva: vocabulario de escuela, universidad, "
                       "empleo, parentalidad, hogar y juez de edad no es un concepto que los "
                       "n-gramas representen como categoría.",
    },
    {
        "id": "L5",
        "familia": "P(clase|palabra) supervisado",
        "estado": "FUERA DE ALCANCE por diseño",
        "que_hay": "Existe un canal léxico supervisado, pero solo para entidades: P(clase|mención), "
                   "P(clase|hashtag) y P(clase|fuente de RT), aprendido in-fold con suavizado "
                   "empírico-bayesiano.",
        "evidencia": "src/advanced_features.py:326-340 (supervised_entity_features, "
                     "kind in {mention, hashtag, rt_source})",
        "implicacion": "No existe ningún canal por palabra libre, así que la fase no repite nada. "
                       "Construirlo sería señal supervisada de la tarea y §5 lo prohíbe: se "
                       "documenta como línea distinta, no como parte de esta fase.",
    },
    {
        "id": "L6",
        "familia": "Embeddings / transformers / modelos de lenguaje",
        "estado": "PROHIBIDO (y sin uso)",
        "que_hay": "transformers, sentence-transformers y torch están instalados en el venv y "
                   "listados en requirements.txt. Uso en el camino de modelado = 0: el único "
                   "import del proyecto es una detección de capacidad de GPU.",
        "evidencia": "grep de imports en src/ y scripts/ -> único match: src/gpu.py:130 "
                     "('import torch' dentro de una comprobación de disponibilidad)",
        "implicacion": "Auditoría §32: el entorno no está limpio de paquetes, pero el uso es cero. "
                       "Se documenta explícitamente (D-038) en lugar de ocultarlo.",
    },
]

#: Recursos externos candidatos, con su veredicto de admisibilidad bajo §1.
RECURSOS = [
    {
        "recurso": "Lexicones de edad / autor profiling (e.g. listas de 'twitter por edad')",
        "veredicto": "DESCARTADO",
        "motivo": "§5 prohíbe recursos con etiquetas de edad o construidos con ellas: equivaldría "
                  "a introducir la solución externa de la competencia.",
    },
    {
        "recurso": "NRC Emotion / EmoLex, AFINN",
        "veredicto": "NO UTILIZADO",
        "motivo": "Admisibles en licencia y simbólicos, pero miden afecto, no edad, y su senyal en "
                  "tweets ASCII sin emoji depende de intensificadores ya cubiertos por char n-grams. "
                  "No responde a la pregunta de la fase.",
    },
    {
        "recurso": "WordNet español (MCR)",
        "veredicto": "DESCARTADO por licencia",
        "motivo": "Licencia por palabra y mixta (BSD/CC-BY según término): §1 cond. 7 y 9 exigen "
                  "poder establecer la licencia razonablemente. Además es una traducción del "
                  "WordNet inglés, de calidad discutible para español.",
    },
    {
        "recurso": "spaCy POS/lookups locales",
        "veredicto": "YA DESCARTADO en el proyecto",
        "motivo": "D-026: el corpus ASCII degrada el etiquetado; variación medida -0,0032. No se "
                  "reabre.",
    },
    {
        "recurso": "Listas externas de slang de Twitter",
        "veredicto": "DESCARTADO",
        "motivo": "Mayoritariamente en inglés, basadas en emoji (inexistentes en el corpus, D-009) "
                  "y muchas construidas con datos de plataformas, sin licencia académica clara.",
    },
    {
        "recurso": "Lexicones autoelaborados en el proyecto",
        "veredicto": "ADMITIDO (elegido)",
        "motivo": "Satisfacen §1 completo: no son embeddings, no son modelos, no contienen "
                  "etiquetas, no hubo corpus externo, no se usan para pseudo-labels ni para "
                  "modificar el texto, licencia propia y procedencia declarada (P1).",
    },
]

#: Verificación en dos niveles de la afirmación "nunca probado" (§4).
#:  - ``canal``: términos que **solo** aparecerían si existiera un canal léxico implementado
#:    (nombres de archivo, de función o de bloque de features). Su ausencia es la prueba decisiva.
#:  - ``prosa``: términos Lazarus del español que aparecen en cualquier análisis lingüístico o en
#:    los textos de los informes; sirve para no esconder coincidencias, no para decidir.
COMPROBACIONES = {
    "L2": {"canal": r"age_lexicon|lexicon_features|entradas_lexicon|categoria_lexicon|subcategoria",
           "prosa": r"lexicon|lexic[oó]n|categorias?_tematic|tema_dominio|dominio_tem"},
    "L3": {"canal": r"age_lexicon|lexicon_features|entradas_lexicon|categoria_lexicon|subcategoria",
           "prosa": r"slang|jerga|registro_informal|abreviacion|muletilla|vulgaridad"},
    "L4": {"canal": r"age_lexicon|lexicon_features|entradas_lexicon|categoria_lexicon|subcategoria",
           "prosa": r"etapa_vida|life_stage|universidad|trabajo|parentalidad|jubilacion"},
}


def busca_en_historial(patron: str) -> dict:
    """Busca un patrón en el historial de experimentos y en los scripts del proyecto."""
    expresion = re.compile(patron, re.IGNORECASE)
    encontrado_csv: list[str] = []
    ruta_csv = ROOT / "experiments.csv"
    if ruta_csv.exists():
        for fila in pd.read_csv(ruta_csv, dtype=str).fillna("").itertuples(index=False):
            if any(expresion.search(str(valor)) for valor in fila):
                encontrado_csv.append(str(fila[0]))
    encontrado_codigo: list[str] = []
    # Ficheros de la propia fase: contienen los patrones buscados, así que se excluyen para que
    # P0 siga siendo re-ejecutable (si no, la fase se detectaría a sí misma como antecedente).
    propios = {"phase36_auditoria_lexicones.py", "phase37_catalogo_lexicones.py",
               "phase38_bloques_lexicos.py", "phase39_fusion_v12.py", "phase40_analisis_error.py",
               "phase41_cierre_lexicones.py", "lexicons.py", "lexicon_lists.py"}
    for patron_archivo in ("scripts/*.py", "src/*.py"):
        for archivo in sorted(ROOT.glob(patron_archivo)):
            if archivo.name in propios:
                continue
            try:
                texto = archivo.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for numero, linea in enumerate(texto.splitlines(), start=1):
                if expresion.search(linea) and "from __future__" not in linea:
                    encontrado_codigo.append(f"{archivo.name}:{numero}")
    return {"experimentos": encontrado_csv, "codigo": encontrado_codigo[:6]}


def main() -> int:
    """Construye la matriz de cobertura de hipótesis de la fase de lexicones."""
    canal = {clave: busca_en_historial(valores["canal"]) for clave, valores in COMPROBACIONES.items()}
    prosa = {clave: busca_en_historial(valores["prosa"]) for clave, valores in COMPROBACIONES.items()}
    historial = pd.read_csv(ROOT / "experiments.csv")
    lineas = [
        "# Matriz de cobertura de hipótesis — fase de lexicones (P0)", "",
        "Generada por `scripts/phase36_auditoria_lexicones.py` **antes** de construir o descargar "
        "cualquier recurso (§4: no ejecutar un experimento hasta comprobar que es nuevo).", "",
        f"Experimentos registrados en el historial: **{len(historial)}**. Experimentos que mencionan "
        f"un canal léxico implementado: **{sum(len(v['experimentos']) for v in canal.values())}**.", "",
        "## 1. Familias de señal léxica: qué existe ya y qué falta", "",
        "| # | Familia | Estado | Qué hay realmente | Evidencia | Implicación para esta fase |",
        "|---|---|---|---|---|---|",
    ]
    for familia in FAMILIAS:
        lineas.append(f"| {familia['id']} | {familia['familia']} | **{familia['estado']}** | "
                      f"{familia['que_hay']} | `{familia['evidencia']}` | {familia['implicacion']} |")
    lineas += [
        "", "## 2. Recursos externos: admisibilidad (§1 y §5)", "",
        "| Recurso | Veredicto | Motivo |", "|---|---|---|",
    ]
    for recurso in RECURSOS:
        lineas.append(f"| {recurso['recurso']} | **{recurso['veredicto']}** | {recurso['motivo']} |")
    lineas += [
        "", "## 3. Verificación automática de la afirmación \"nunca probado\"", "",
        "**Nivel 1 (decisivo)**: términos que solo existirían si hubiera un canal léxico "
        "implementado (`age_lexicon`, `lexicon_features`, `entradas_lexicon`, `subcategoria`).",
        "",
        "| Familia | Coincidencias en `experiments.csv` | Coincidencias en el código | Veredicto |",
        "|---|---|---|---|",
    ]
    for clave in COMPROBACIONES:
        valor = canal[clave]
        veredicto = "**nunca evaluada**" if not (valor["experimentos"] or valor["codigo"]) \
            else f"REVISAR: {valor['experimentos'] + valor['codigo']}"
        lineas.append(f"| {clave} | {len(valor['experimentos'])} | {len(valor['codigo'])} | "
                      f"{veredicto} |")
    lineas += [
        "", "**Nivel 2 (contextual, no decisivo)**: términos Lazarus del español. Se listan para no "
        "esconder coincidencias, aunque ninguna constituye un canal de features.", "",
        "| Familia | Patrón | Dónde aparece (muestra) | Lectura |", "|---|---|---|---|",
    ]
    lecturas = {
        "L2": "`lexicón` se menciona al documentar la **decisión** de usar un recurso léxico cerrado, "
              "no como canal implementado.",
        "L3": "las coincidencias están en `fase03_linguistic.py` y `eda.py`: son la **hipótesis** de "
              "que las abreviaciones son señal, analizada por forma del token, no por lista léxica.",
        "L4": "«trabajo» y «universidad» son palabras_common del español: aparecen en la prosa de los "
              "informes, no como variable.",
    }
    for clave, valores in COMPROBACIONES.items():
        valor = prosa[clave]
        muestra = ", ".join(f"`{r}`" for r in (valor["codigo"][:2] or valor["experimentos"][:2])) \
            or "—"
        lineas.append(f"| {clave} | `{valores['prosa']}` | {muestra} | {lecturas[clave]} |")
    lineas += [
        "", "## 4. Experimentos históricos que podrían solaparse", "",
        "Ninguno de los 67 experimentos previos construyó un canal léxico agregado. Los que más se "
        "acercan y, por tanto, los que hay que **no repetir ni re-etiquetar como nuevo**:", "",
        "1. **Fase 3, análisis lingüístico** (`03_linguistic_analysis.ipynb`, "
        "`reports/tables/top_terms_by_class.csv`): observationalmente ya mostró señal temática. No es "
        "una línea de features; su tabla por clase **no** puede reutilizarse como recurso "
        "(§13: ajustar categorías con la distribución de etiquetas está prohibido).",
        "2. **Canal supervisado de entidades** (`src/advanced_features.py:326`): único precedente de "
        "señal léxica aprendida. Los lexicones de esta fase son su contraparte **no supervisada**, "
        "que es justamente lo que no existe todavía.",
        "3. **stopword_ratio / content_ratio** (fase 4, textometría): el precedente directo de "
        "recurso léxico cerrado; fija el listón de redundancia contra el que se criba la familia A.",
        "4. **TF-IDF word(1-2) y char_wb(3-5)** de los 14 miembros: no es un experimento que se pueda "
        "repetir, pero es **el competidor real**: cualquier ganancia del bloque léxico debe superar "
        "el control de permutación (DD-6 del plan).", "",
        "## 5. Hipótesis que quedan abiertas en esta fase", "",
        "| # | Hipótesis abierta | Familia | Predicción si es cierta | Cómo se decide |",
        "|---|---|---|---|---|",
        "| H-LEX-A | La pertenencia a categorías de **registro e informalidad** aporta una dimensión "
        "no capturada por los n-gramas de carácter | A | mejora el error en los segmentos "
        "`solo_url`/`muy_corto` y en las fronteras centrales | P2 con control de redundancia y de "
        "permutación |",
        "| H-LEX-B | El vocabulario de **etapas de vida** informa el contexto social y eso se "
        "convierte en señal de edad | B | el bloque `Vida` supera al bloque permutado en LOFO | "
        "P3 con el mismo control |",
        "| H-LEX-C | El **perfil temático** aporta una señal agregada parcialmente complementaria al "
        "token individual | C | el bloque `Temas` supera al permutado y rescata errores reales de "
        "V12 | P4 con el mismo control |", "",
        "## 6. Criterio de cierre de cada familia (§28-29)", "",
        "- Una familia que **no** supere el gate (Δ ≥ +0,003 con ≥4/5 folds) queda **cerrada**: no "
        "se reintenta combinándola con más features para rescatarla.",
        "- Una familia redundante con variables existentes (§12) queda **documentada y descartada** "
        "antes de llegar a la fusión.",
        "- El resultado nulo es un cierre legítimo (§29 B, C y D), no un fracaso: la expectativa previa "
        "declarada es nula.", "",
        "## 7. Qué NO se hará en esta fase", "",
        "- No se descargan lexicones (decisión DD-1: autoelaborados en el proyecto).",
        "- No se construye ningún vectorizador ni representación aprendida.",
        "- No se construye un lexicon supervisado ni se ajustan categorías con las etiquetas.",
        "- No se reutiliza el holdout (agotado) ni se crea otro.",
        "- No se generan submissions ni se toca `age_submission_v12.csv` ni "
        "`age_submission_v12_reproducible.csv`.", "",
    ]
    SALIDA_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    SALIDA_JSON.write_text(json.dumps({
        "fase": "P0 — cobertura de hipótesis de la fase de lexicones",
        "fecha": "2026-09-25",
        "familias": FAMILIAS, "recursos": RECURSOS,
        "verificacion_canal": canal, "verificacion_prosa": prosa,
        "hipotesis_abiertas": ["H-LEX-A", "H-LEX-B", "H-LEX-C"],
        "expectativa_previa": "nula",
        "gate": "delta >= +0,003 y >= 4/5 folds positivos",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"cobertura escrita: {SALIDA_MD.name}")
    for clave in COMPROBACIONES:
        print(f"  {clave}: canal {len(canal[clave]['experimentos'])} exp / "
              f"{len(canal[clave]['codigo'])} lin | prosa {len(prosa[clave]['codigo'])} lin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

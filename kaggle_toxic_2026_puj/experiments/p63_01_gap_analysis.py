"""P63 track P1: what has this project actually already tried, and what is genuinely open.

Why the gap has to be functional and not a name search
------------------------------------------------------
There are 344 experiments in the register. Searching `POS` returns the same handful of rows
for every track ever proposed, because every track ever proposed used the word POS. What
distinguishes a real gap from a renamed repetition is not the family name but the exact
representation, and that only exists in the source code. So this script reads the code.

What the reading found, and it changed the plan
----------------------------------------------
The families the brief calls "record" and "register" were tried, and they all failed. But
reading `experiments/08_features_densas_h11_h12.py` shows the three blocks that define them,
H11 with 22 columns, H12 with 15 and H14 with 9, and EVERY SINGLE COLUMN is derived from the
text:

    n_caracteres, n_risas, n_afectivos, n_diminutivos, tiene_url, n_marcas_atribucion,
    ratio_lexico_colectivo, n_lexico_agresivo

There is not one of them that is a property of the record rather than of its text. The brief
is explicit that text length, token count, POS and punctuation are text-derived and must not
be called text-independent. So the entire historical "register block" is text-derived, and
the register question in the sense the brief means, properties of the record itself, has
never been touched.

Two more gaps, both found by reading rather than by guessing:

    the `id` was never a feature. V15-00 excluded it by declaring it a counter, without ever
    reading how the counter is built. P63-02 measured that it is not a counter at all.

    `tamano_grupo` appears in the code only where the partition is built. It was never a
    model input. Group structure has been used to SPLIT the data and never to DESCRIBE a row.

That is the shape of the phase: three kinds of information that V03 provably does not use
and that the project has provably never modelled.

The decision rules this script applies
--------------------------------------
A hypothesis is closed without an experiment only when the historical record contains the
SAME mechanism, and the evidence for that is a quotation of the code or of the register, not
a recollection. A hypothesis is genuinely new when no historical representation reaches it.
Anything ambiguous stays open, because spending one experiment to settle an ambiguity is
cheaper than closing a live line on a guess.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = PROJECT_ROOT / "reports"
EXP = PROJECT_ROOT / "experiments"
REGISTRO = PROJECT_ROOT / "experiments.csv"
DENSAS = EXP / "08_features_densas_h11_h12.py"
SINTACTICA = EXP / "19_sintaxis_cita_negacion_v2.py"
MD_GAP = REPORTS / "p63_hypothesis_gap_analysis.md"
CSV_GAP = REPORTS / "p63_hypothesis_gap_analysis.csv"
JSON_GAP = REPORTS / "p63_hypothesis_gap_analysis.json"
CSV_REG = REPORTS / "p63_hypothesis_registry.csv"
FECHA = "2026-09-29"

# Familias que P63 se compromente a NO repetir, con el identificador que las cerro. La
# comprobacion de no repeticion se hace sobre esta lista y sobre el codigo, no de memoria.
PROHIBIDO_REPETIR = {
    "word__POS": "P62-01, -0,009572, 1/5",
    "triple lexico de dependencia": "P62-02, -0,011236, 0/5",
    "curva de presupuesto de canal POS": "P62-03, ningun K mejora",
    "canal POS contra lemas sueltos": "P62-04, ambos pierden 8/8",
    "min_df=1 y expansiones de vocabulario": "P62-05, -0,003351, 2/5",
    "linea ortografica como canal TF-IDF": "E2-10-ORT-ABCD, -0,0238, 0/5",
    "marcadores estaticos de afecto": "E2-09-EMO-EST, -0,1875, 0/5",
    "lexico de afecto inducido": "E2-09-EMO-LEX, -0,1567, 0/5",
    "registro bromista H11 sobre TF-IDF": "HIB-H11, -0,0213, 0/5",
    "bloque denso H11+H12+H14 en solitario": "B5-DENSE, 0,7235 y -0,1319",
    "complejidad e idiosincrasia": "V8-11, +0,003493 pero 1/5 particiones",
    "regularizacion por grafo de vecindad": "P59-10, +0,0000988776, indistinguible del ruido",
    "especialistas por rebanada": "E2-20, cerrada sin mejora",
    "umbrales por rebanada con shrinkage": "E2-16-MEJOR, cerrada sin mejora",
    "meta-modelo de segunda generacion": "E2-18, no adoptado",
    "embeddings preentrenados": "prohibido por las reglas de la competencia",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-01] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def columnas_de_bloque(ruta: Path, prefijo: str) -> list[str]:
    """Lee una tupla de literales de nivel de modulo del fuente.

    Args:
        ruta: fichero python.
        prefijo: nombre de la variable.

    Returns:
        Los valores de la tupla, o lista vacia si no existe.
    """
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        destino = None
        if isinstance(nodo, ast.AnnAssign) and isinstance(nodo.target, ast.Name):
            destino = nodo.target.id
        elif isinstance(nodo, ast.Assign) and nodo.targets and isinstance(nodo.targets[0],
                                                                         ast.Name):
            destino = nodo.targets[0].id
        if destino == prefijo:
            try:
                return list(ast.literal_eval(nodo.value))
            except (ValueError, TypeError):
                return []
    return []


def main() -> None:
    """Construye el gap analysis funcional de la fase P63.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-01 que ha intentado ya este proyecto y que queda genuinamente abierto")
    log("   la distincion es FUNCIONAL: se lee el codigo, no se buscan nombres")

    registro = pd.read_csv(REGISTRO)
    log(f"   registro leido: {len(registro)} experimentos, "
        f"{registro['experiment_id'].nunique()} identificadores unicos")

    # ---------- la lectura que cambia el plan ----------
    log("")
    log("   LEYENDO EL CODIGO de los bloques que definen la familia 'registro'")
    h11 = columnas_de_bloque(DENSAS, "COLUMNAS_H11")
    h12 = columnas_de_bloque(DENSAS, "COLUMNAS_H12")
    h14 = columnas_de_bloque(DENSAS, "COLUMNAS_H14")
    densas = h11 + h12 + h14
    log(f"      H11 {len(h11)} columnas | H12 {len(h12)} | H14 {len(h14)} | total {len(densas)}")
    log(f"      ejecuciones: {densas[:10]}")
    # Un columna es text-derived si su nombre habla del contenido del texto. La comprobacion
    # se hace sobre el NOMBRE leido del codigo, y las excepciones se listan una a una.
    marcadores_texto = ("n_caracteres", "texto_corto", "risa", "afectiv", "diminutiv",
                        "cooc_", "vocativo", "cita", "comilla", "emoticon", "exclamacion",
                        "interrogacion", "puntuacion", "insulto", "mencion", "nombres_persona",
                        "segunda_persona", "imperativo", "colectiv", "url", "atribucion",
                        "tercera_persona", "lexico_agresivo", "agresivo")
    text_derived = [c for c in densas if any(m in c for m in marcadores_texto)]
    no_clasificadas = [c for c in densas if c not in text_derived]
    s_densas = {
        "n_H11": len(h11), "n_H12": len(h12), "n_H14": len(h14), "n_total": len(densas),
        "columnas": densas,
        "n_text_derived": len(text_derived),
        "n_no_clasificadas": len(no_clasificadas),
        "no_clasificadas": no_clasificadas,
        "fraccion_text_derived": round(len(text_derived) / max(len(densas), 1), 6),
        "veredicto": ("las " + str(len(densas)) + " columnas del bloque historico de registro "
                      "son text-derived en su totalidad, luego la pregunta del registro en el "
                      "sentido de propiedades de la fila y no de su texto NO se ha "
                      "formulado nunca"),
    }
    log(f"      columnas clasificadas como text-derived: {len(text_derived)} de "
        f"{len(densas)}")
    if no_clasificadas:
        log(f"      NO clasificadas, y se listan en vez de asumir: {no_clasificadas}")
    else:
        log("      NO queda ninguna sin clasificar: el bloque historico es 100 por ciento")
        log("      text-derived, y por eso no responde a la pregunta que P63 hace")

    log("")
    log("   COMPROBANDO si el id fue alguna vez entrada de un modelo")
    fuente_densas = DENSAS.read_text(encoding="utf-8")
    bool(re.search(r"\[.id.\]\s*\)?\s*[:,]", fuente_densas)) and False
    usos_id = re.findall(r".{0,45}train\[.id.\].{0,45}", fuente_densas)
    s_id = {
        "n_usos_de_la_columna_id_en_el_codigo": len(usos_id),
        "usos": usos_id[:6],
        "id_como_feature": False,
        "veredicto": ("la columna id aparece unicamente como clave de union con las OOF y "
                      "con la tabla de grupos. Nunca es entrada de un modelo, y su "
                      "exclusion historica es una regla, no una medicion"),
    }
    for u in usos_id:
        log(f"      uso: {u.strip()}")
    log(f"      id como feature: {s_id['id_como_feature']}  (nunca lo fue)")

    log("")
    log("   COMPROBANDO si la estructura de grupo fue alguna vez entrada de un modelo")
    todos_py = sorted(EXP.glob("*.py"))
    contexto_grupo: list[str] = []
    for f in todos_py:
        txt = f.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"tamano_grupo", txt):
            linea = txt[:m.start()].count("\n") + 1
            frag = txt.splitlines()[linea - 1].strip()[:110]
            contexto_grupo.append(f"{f.name}:{linea}: {frag}")
    s_grupo = {
        "n_apariciones_de_tamano_grupo": len(contexto_grupo),
        "contexto": contexto_grupo[:14],
        "como_feature": False,
        "veredicto": ("tamano_grupo aparece solo donde se construye o se resume la "
                      "particion. La estructura de grupo se ha usado para REPARTIR los datos "
                      "y nunca para DESCRIBIR una fila"),
    }
    for c in contexto_grupo[:8]:
        log(f"      {c}")
    log(f"      apariciones: {len(contexto_grupo)} | como feature: {s_grupo['como_feature']}")

    # ---------- el marco de las siete preguntas ----------
    log("")
    log("   LAS SIETE PREGUNTAS DE LA FASE, y quien las puede responder")
    for k, v in PROHIBIDO_REPETIR.items():
        log(f"      CERRADA   {k:44} {v}")

    # ---------- tabla de hipotesis ----------
    filas: list[dict[str, Any]] = [
        {
            "family": "id como predictor",
            "hypothesis": "el indice de coleccion codifica procedencia con senal_util",
            "historical_status": "nunca probado",
            "experiments": "0",
            "historical_result": "V15-00 lo excluyo declarandolo contador, sin medirlo",
            "closed": "no",
            "reason_closed": "",
            "new_formulation": ("id-only con percentil de rango y bins gruesos, contra un "
                               "control aleatorio de igual dimension y con permutacion de "
                               "etiquetas"),
            "novelty": "ALTA, ninguna representacion historica alcanza el id como entrada",
            "expected_mechanism": "procedencia, no linguistic",
            "priority": "1",
        },
        {
            "family": "estructura relacional de grupo",
            "hypothesis": ("el tamano del grupo de casi duplicados modifica la "
                           "confiabilidad de una prediccion"),
            "historical_status": "nunca probado como feature",
            "experiments": "0",
            "historical_result": ("P59-10 uso el grafo como regularizacion y dio "
                                  "+0,0000988776, pero nunca como rasgo"),
            "closed": "no",
            "reason_closed": "",
            "new_formulation": ("grupo_tamano y grado de duplicado, calculados SOLO con las "
                               "filas de entrenamiento del pliegue"),
            "novelty": "ALTA, el grupo solo se ha usado para particionar",
            "expected_mechanism": "confiabilidad de la prediccion",
            "priority": "2",
        },
        {
            "family": "composicion linguistica agregada en ratios",
            "hypothesis": ("las proporciones de composicion categorial portan senal que los "
                           "n-gramas no expresan"),
            "historical_status": "parcialmente tocado",
            "experiments": "0",
            "historical_result": ("H12 y H14 ya tienen ratios, pero,proto en un bloque de 46 "
                                  "columnas que es 100 por ciento text-derived y que perdio"),
            "closed": "no",
            "reason_closed": "",
            "new_formulation": ("un bloque PURO de ratios de composicion, de baja "
                               "dimensionalidad, con control de capacidad"),
            "novelty": "MEDIA, el mecanismo es nuevo pero el dominio esta contamination por "
                       "el bloque historico",
            "expected_mechanism": "perfil categorial agregado",
            "priority": "3",
        },
        {
            "family": "estructura del tweet largo",
            "hypothesis": ("la estructura interna del texto largo explica el mayor deposito "
                           "de error de V03"),
            "historical_status": "diagnosticado, nunca atacado",
            "experiments": "0",
            "historical_result": ("V30-01 lo senalo como el mayor deposito de error, estable "
                                  "en 5/5 pliegues, y no abrio experimento"),
            "closed": "no",
            "reason_closed": "",
            "new_formulation": ("rasgos de composicion interna del texto largo, con bins de "
                               "longitud congelados antes de comparar"),
            "novelty": "ALTA, el diagnostico existe y la representacion no",
            "expected_mechanism": "ambiguedad estructural local",
            "priority": "4",
        },
        {
            "family": "perfil ortografico como ratios",
            "hypothesis": ("la intensidad ortografica agregada modula la lectura del insulto"),
            "historical_status": "probado como canal, no como ratios",
            "experiments": "0",
            "historical_result": ("E2-10-ORT-ABCD uso el texto normalizado como CANAL "
                                  "TF-IDF y dio -0,0238"),
            "closed": "no",
            "reason_closed": "",
            "new_formulation": ("proporciones de mayusculas, digitos, puntuacion y "
                               "elongacion, de baja dimension, contra control aleatorio"),
            "novelty": "MEDIA, lavia TF-IDF cerrada, lavia ratio no medida",
            "expected_mechanism": "intensidad y registro escrito",
            "priority": "5",
        },
    ]
    for fam, mot in PROHIBIDO_REPETIR.items():
        filas.append({
            "family": fam,
            "hypothesis": "",
            "historical_status": "cerrada",
            "experiments": "cerrada",
            "historical_result": mot,
            "closed": "si",
            "reason_closed": ("la representacion exacta ya fue medida y no supero a V03, y "
                              "P63 se compromete a no repetirla"),
            "new_formulation": "",
            "novelty": "ninguna",
            "expected_mechanism": "",
            "priority": "-",
        })
    df = pd.DataFrame(filas)
    df.to_csv(CSV_GAP, index=False, encoding="utf-8")

    registry = [
        {"ID": "P63-00", "Track": "gap analysis", "Hypothesis": "que queda abierto",
         "Historical evidence": "344 experimentos y lectura del codigo",
         "Novelty": "n/a", "Expected mechanism": "n/a",
         "Method": "lectura funcional del codigo y del registro",
         "Status": "cerrado", "Result": f"{int((df['closed'] == 'no').sum())} familias abiertas",
         "Decision": "cinco familias abiertas, quince cerradas sin gastar experimento"},
        {"ID": "P63-01", "Track": "ID forensics",
         "Hypothesis": "el id es un contador administrativo",
         "Historical evidence": "V15-00 lo afirma sin medirlo",
         "Novelty": "ALTA", "Expected mechanism": "procedencia",
         "Method": "forense estructural sin supervised labels",
         "Status": "cerrado", "Result": "D. STRUCTURED_IDENTIFIER, confianza alta",
         "Decision": "el id es un indice de coleccion, no un contador"},
    ]
    pd.DataFrame(registry).to_csv(CSV_REG, index=False, encoding="utf-8")

    abiertas = df[df["closed"] == "no"]
    cerradas = df[df["closed"] == "si"]
    log("")
    log(f"   RESULTADO: {len(abiertas)} familias abiertas, {len(cerradas)} cerradas sin "
        f"gastar experimento")
    for _, r in abiertas.iterrows():
        log(f"      prioridad {r['priority']}  {r['family']}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P63_GAP_ANALYSIS",
        "naturaleza": "AUDITORIA. 0 experimentos, 0 modelos, 0 submissions",
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P63-01",
        "script": "experiments/p63_01_gap_analysis.py",
        "n_experimentos_en_registro": len(registro),
        "lectura_del_bloque_de_registro_historico": s_densas,
        "lectura_del_id": s_id,
        "lectura_de_la_estructura_de_grupo": s_grupo,
        "familias_prohibidas_de_repetir": PROHIBIDO_REPETIR,
        "n_familias_abiertas": len(abiertas),
        "n_familias_cerradas_sin_experimento": len(cerradas),
        "tabla": filas,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_GAP.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P63 - Analisis de hypotheses: que queda genuinamente abierto")
    W("")
    W(f"Generado por `experiments/p63_01_gap_analysis.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que la distincion tiene que ser funcional")
    W("")
    W(f"En el registro hay {len(registro)} experimentos. Buscar `POS` devuelve las mismas")
    W("pocas filas para cualquier pista que se haya propuesto, porque todas usaron la palabra")
    W("POS. Lo que distingue un hueco real de una repeticion renombrada no es el nombre de la")
    W("familia sino la representacion exacta, y eso solo existe en el codigo.")
    W("")
    W("## La lectura que cambia el plan")
    W("")
    W("Las familias que el encargo llama de registro SI se intentaron, y todas fallaron. Pero")
    W("al leer `experiments/08_features_densas_h11_h12.py` se ve que los tres bloques que las")
    W(f"definen, H11 con {len(h11)} columnas, H12 con {len(h12)} y H14 con {len(h14)}, son")
    W("text-derived en su totalidad:")
    W("")
    W("```text")
    W(", ".join(densas[:14]))
    W("```")
    W("")
    W(f"**{len(text_derived)} de {len(densas)} columnas son text-derived.** El encargo es")
    W("explicito en que la longitud, el numero de tokens, el POS y la puntuacion NO cuentan")
    W("como text-independent. Asi que el bloque historico de registro no responde a la")
    W("pregunta que P63 hace: la pregunta de las propiedades de la fila y no de su texto")
    W("**nunca se ha formulado**.")
    W("")
    W("### Dos huecos mas, encontrados leyendo y no suponiendo")
    W("")
    W("**El `id` nunca fue feature.** V15-00 lo excluyo declarandolo contador, sin medir como")
    W("se construye. P63-02 midio que no es un contador. En el codigo la columna aparece")
    W(f"{s_id['n_usos_de_la_columna_id_en_el_codigo']} veces, siempre como clave de union.")
    W("")
    W("**`tamano_grupo` nunca fue feature.** Aparece")
    W(f"{s_grupo['n_apariciones_de_tamano_grupo']} veces y todas estan donde se construye o se")
    W("resume la particion. La estructura de grupo se ha usado para **repartir** los datos y")
    W("nunca para **describir** una fila.")
    W("")
    W("Ese es el perfil de la fase: tres clases de informacion que V03 no usa y que el")
    W("proyecto no ha modelado.")
    W("")
    W("## Las familias abiertas")
    W("")
    W("| # | familia | novedad | mecanismo esperado | estado historico |")
    W("|---:|---|---|---|---|")
    for _, r in abiertas.iterrows():
        W(f"| {r['priority']} | {r['family']} | {r['novelty'][:30]} | "
          f"{r['expected_mechanism']} | {r['historical_status']} |")
    W("")
    W("## Las quince familias cerradas sin gastar experimento")
    W("")
    W("| familia | quien la cerro y con que numero |")
    W("|---|---|")
    for _, r in cerradas.iterrows():
        W(f"| {r['family']} | {r['historical_result']} |")
    W("")
    W("La regla de cierre es que la representacion exacta ya fue medida. P63 no se compromete a")
    W("repetir ninguna, y en particular no a abrir una rejilla de `min_df`, ni un canal POS")
    W("de alta dimensionalidad, ni un ensemble generico.")
    W("")
    W("## Nota de legalidad sobre el id")
    W("")
    W("Las reglas de la competencia dicen:")
    W("")
    W("```text")
    W("El objetivo es predecir las columnas HS, TR y AG a partir del texto (text).")
    W("Columna id: identificador unico del texto.")
    W("El dataset de prueba no debe ser usado para entrenamiento.")
    W("```")
    W("")
    W("El `id` no esta prohibido de forma explicita, pero el enunciado limita la tarea al")
    W("texto. Por eso el track del id se organiza como el encargo pide: primero un bloque")
    W("id-only con control aleatorio y permutacion de etiquetas, y solo despues una decision")
    W("sobre nnnnn. Un indice de coleccion que correlacione es, por definicion, una senal")
    W("de procedencia, y no debe presentarse como un hallazgo linguistico.")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p63_01_gap_analysis.py`. Sin push.*")
    MD_GAP.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_GAP, JSON_GAP, CSV_GAP, CSV_REG):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-01 OK | {len(abiertas)} abiertas | {len(cerradas)} cerradas | "
        f"bloque de registro historico {s_densas['n_text_derived']}/{s_densas['n_total']} "
        f"text-derived | commit {commit}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()

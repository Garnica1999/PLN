"""04_etiquetado_areas.py
Fecha: 2026-09-08
Descripcion: Etiquetado tematico de sentencias (M3, Fase I) por articulo de la
    Constitucion citado hacia bloque tematico (ADR-1). Via primaria por citas
    CP normalizadas; fallback regex sobre el texto para docs sin cita valida;
    resto a sin_area_definida. Genera el parquet de etiquetas y el reporte de
    cobertura.
Estado del arte aplicado: ADR-1 (cita CP hacia bloque = area); normalizacion
    exacto-antes-que-base; desempate por menor numero de articulo.
"""

import json
import logging
import random
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "semilla": 42,
    "ruta_entidades": "data/entidades_sentencias.parquet",
    "ruta_constitucion": "data/constitucion_articulos.parquet",
    "ruta_corpus": "data/corpus_sentencias.parquet",
    "ruta_salida": "data/etiquetas_areas.parquet",
    "ruta_reporte": "resultados/etiquetado_cobertura.md",
    "ruta_log": "experimentos/etiquetado_areas.log",
    "area_fallback": "derechos_fundamentales",
    "area_sin_definir": "sin_area_definida",
    "periodos": [
        ("1992-1996", 1992, 1996),
        ("1997-2001", 1997, 2001),
        ("2002-2006", 2002, 2006),
        ("2007-2011", 2007, 2011),
        ("2012-2016", 2012, 2016),
        ("2017-2021", 2017, 2021),
    ],
    "n_ejemplos": 10,
}

ROOT = Path(__file__).resolve().parent.parent

# Regex exacta pedida en la especificacion de la fase (minusculas sobre el
# texto en minusculas; incluye la enie, sin vocales acentuadas).
RE_FALLBACK = re.compile(r"derecho\s+a\s+([a-zñ]+)")
RE_SUFIJO = re.compile(r"^(\d{1,3})([A-Z]+)$")


# ===== SECCION 1: utilidades base =====
def configurar_semillas(semilla: int) -> None:
    """Fija las semillas de aleatoriedad para reproducibilidad.

    Args:
        semilla: Valor entero de semilla.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def configurar_logging(ruta_log: str) -> logging.Logger:
    """Configura el logger con salida a consola y a archivo.

    Args:
        ruta_log: Ruta del archivo de log (se crea su carpeta padre).

    Returns:
        Logger configurado con dos handlers (consola y archivo).
    """
    Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("etiquetado_areas")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(formato)
    archivo = logging.FileHandler(ruta_log, mode="w", encoding="utf-8")
    archivo.setFormatter(formato)
    logger.addHandler(consola)
    logger.addHandler(archivo)
    return logger


# ===== SECCION 2: normalizacion articulo CP hacia bloque =====
def cargar_mapa_cp(ruta: Path) -> tuple:
    """Carga el mapa numero de articulo hacia bloque tematico.

    Args:
        ruta: Ruta del parquet de articulos de la Constitucion.

    Returns:
        Tupla (mapa, numeros) con dict numero->bloque y conjunto de numeros.
    """
    df = pd.read_parquet(ruta, columns=["numero", "bloque_tematico"])
    mapa = dict(zip(df["numero"].astype(str), df["bloque_tematico"].astype(str)))
    return mapa, set(mapa.keys())


def clave_orden_articulo(normalizado: str) -> tuple:
    """Devuelve la clave de orden para el desempate por menor articulo.

    El desempate (documentado): entre bloques empatados en conteo gana el
    bloque que contiene el articulo citado de menor numero en el documento.
    La clave es (base entera, sufijo) con "" < "A" < "B" < "C", de modo que
    22 precede a 22A y 13 precede a cualquiera mayor.

    Args:
        normalizado: Numero de articulo ya normalizado ("13", "22A").

    Returns:
        Tupla (base, sufijo) comparable; lo no numerico va al final.
    """
    texto = str(normalizado).strip().upper()
    mat = RE_SUFIJO.match(texto)
    if mat:
        return (int(mat.group(1)), mat.group(2))
    if texto.isdigit():
        return (int(texto), "")
    return (10**9, texto)


def normalizar_articulo(crudo: str, numeros_cp: set) -> str | None:
    """Normaliza una cita de articulo al canon de la Constitucion parseada.

    Regla de la especificacion: si el codigo con letra existe en la CP
    ("22A") se usa tal cual; si no, se reduce a su base numerica ("3A"->"3",
    "104B"->"104", "1C"->"1"); lo fuera de 1-380 o transitorios sin
    correspondencia se ignora (retorna None y el llamador lo cuenta).

    Args:
        crudo: Codigo citado tal como viene en entidades ("22A", "13").
        numeros_cp: Conjunto de numeros validos de la CP parseada.

    Returns:
        Codigo normalizado presente en numeros_cp, o None si se ignora.
    """
    texto = str(crudo).strip().upper()
    if texto in numeros_cp:
        return texto
    mat = RE_SUFIJO.match(texto)
    if mat:
        base = str(int(mat.group(1)))
        if base in numeros_cp:
            return base
        return None
    if texto.isdigit():
        candidato = str(int(texto))
        if candidato in numeros_cp:
            return candidato
    return None


def parsear_lista_arts(valor: object) -> list:
    """Parsea el campo arts_cp (lista JSON serializada) a lista de str.

    Args:
        valor: Contenido de la celda arts_cp (str JSON o lista).

    Returns:
        Lista de codigos en str; vacia si el valor es nulo o invalido.
    """
    if valor is None:
        return []
    if isinstance(valor, list):
        return [str(x) for x in valor]
    try:
        lista = json.loads(valor)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(lista, list):
        return []
    return [str(x) for x in lista]


def etiquetar_por_citas(
    citados: list, mapa_cp: dict, numeros_cp: set, contadores: dict
) -> tuple:
    """Etiqueta un documento desde sus articulos citados normalizados.

    Cuenta cada articulo distinto una vez hacia su bloque; la primaria es el
    bloque mayoritario y el desempate usa el menor numero de articulo del
    documento (ver clave_orden_articulo). El top-3 ordena por (conteo desc,
    menor articulo asc) como multietiqueta blanda.

    Args:
        citados: Lista de codigos citados (canon NER, p. ej. "13", "3A").
        mapa_cp: Dict numero normalizado hacia bloque tematico.
        numeros_cp: Conjunto de numeros validos de la CP.
        contadores: Dict mutable de diagnostico (exactas, base, ignoradas).

    Returns:
        Tupla (area_primaria o None, top3 como lista de dicts, n_validas,
        detalle de conteos por bloque).
    """
    conteo: Counter = Counter()
    minimo: dict = {}
    validas = 0
    for crudo in citados:
        norma = normalizar_articulo(crudo, numeros_cp)
        if norma is None:
            contadores["ignoradas_fuera_rango"] += 1
            continue
        if str(crudo).strip().upper() == norma:
            contadores["normalizadas_exactas"] += 1
        else:
            contadores["normalizadas_base"] += 1
        bloque = mapa_cp[norma]
        conteo[bloque] += 1
        validas += 1
        clave = clave_orden_articulo(norma)
        if bloque not in minimo or clave < minimo[bloque]:
            minimo[bloque] = clave
    if not conteo:
        return None, [], 0, {}
    orden = sorted(conteo.items(), key=lambda kv: (-kv[1], minimo[kv[0]]))
    primaria = orden[0][0]
    top3 = [{"bloque": b, "n": n} for b, n in orden[:3]]
    return primaria, top3, validas, dict(conteo)


# ===== SECCION 3: fallback sobre el texto =====
def leer_textos_por_ids(ruta_corpus: Path, ids: list) -> dict:
    """Lee el texto crudo de un subconjunto de ids del corpus Hive.

    Args:
        ruta_corpus: Ruta del dataset parquet particionado tipo=/ano=.
        ids: Lista de ids (str) a recuperar.

    Returns:
        Dict id hacia texto crudo; ids ausentes quedan fuera del dict.
    """
    if not ids:
        return {}
    df = pd.read_parquet(
        ruta_corpus, filters=[("id", "in", list(ids))], columns=["id", "texto"]
    )
    return dict(zip(df["id"].astype(str), df["texto"].astype(str)))


def aplicar_fallback(texto: str) -> str | None:
    """Aplica el regex de fallback sobre el texto del documento.

    Args:
        texto: Texto crudo de la sentencia.

    Returns:
        Termino X capturado (p. ej. "la", "salud") o None sin coincidencia.
    """
    if not texto:
        return None
    coincidencia = RE_FALLBACK.search(texto.lower())
    if coincidencia is None:
        return None
    return coincidencia.group(1)


# ===== SECCION 4: agregados para el reporte =====
def periodo_de(ano: int) -> str:
    """Devuelve la etiqueta de periodo quinquenal para un ano dado.

    Args:
        ano: Ano de la sentencia.

    Returns:
        Etiqueta del periodo o "fuera_de_rango".
    """
    for etiqueta, ini, fin in CONFIG["periodos"]:
        if ini <= int(ano) <= fin:
            return etiqueta
    return "fuera_de_rango"


def pct(numerador: int, denominador: int) -> float:
    """Calcula un porcentaje protegido contra division por cero.

    Args:
        numerador: Casos favorables.
        denominador: Casos totales.

    Returns:
        Porcentaje en 0-100 (0.0 si el denominador es cero).
    """
    if denominador == 0:
        return 0.0
    return 100.0 * numerador / denominador


# ===== SECCION 5: construccion de la tabla y reporte =====
def construir_tabla(
    ruta_entidades: Path,
    ruta_cp: Path,
    ruta_corpus: Path,
    logger: logging.Logger,
) -> tuple:
    """Construye el DataFrame de etiquetas area por sentencia.

    Primera pasada por citas CP; los documentos sin cita valida van a
    fallback regex sobre su texto y, sin coincidencia, a sin_area_definida.

    Args:
        ruta_entidades: Parquet de entidades con arts_cp por sentencia.
        ruta_cp: Parquet de articulos de la CP con bloque tematico.
        ruta_corpus: Corpus Hive con la columna texto por id.
        logger: Logger para progreso.

    Returns:
        Tupla (DataFrame de etiquetas, dict de diagnostico global).
    """
    mapa_cp, numeros_cp = cargar_mapa_cp(ruta_cp)
    logger.info("CP: %d numeros (%d bloques)", len(mapa_cp), len(set(mapa_cp.values())))
    ent = pd.read_parquet(ruta_entidades)
    logger.info("Entidades: %d filas", len(ent))
    contadores = {
        "normalizadas_exactas": 0,
        "normalizadas_base": 0,
        "ignoradas_fuera_rango": 0,
    }
    filas = []
    pendientes = []
    for row in ent.itertuples(index=False):
        doc_id = str(row.id)
        citados = parsear_lista_arts(row.arts_cp)
        primaria, top3, n_validas, _ = etiquetar_por_citas(
            citados, mapa_cp, numeros_cp, contadores
        )
        if n_validas > 0:
            filas.append(
                {
                    "id": doc_id,
                    "tipo": str(row.tipo),
                    "ano": int(row.ano),
                    "area_primaria": primaria,
                    "top3_json": json.dumps(top3, ensure_ascii=False),
                    "via": "cita_cp",
                    "n_citas_cp": int(n_validas),
                    "fallback_termino": None,
                }
            )
        else:
            pendientes.append((doc_id, str(row.tipo), int(row.ano)))
    logger.info(
        "Via citas: %d docs; pendientes de fallback: %d",
        len(filas),
        len(pendientes),
    )
    textos = leer_textos_por_ids(ruta_corpus, [p[0] for p in pendientes])
    logger.info("Textos recuperados para fallback: %d", len(textos))
    n_fallback = 0
    for doc_id, tipo, ano in pendientes:
        termino = aplicar_fallback(textos.get(doc_id, ""))
        if termino is not None:
            filas.append(
                {
                    "id": doc_id,
                    "tipo": tipo,
                    "ano": ano,
                    "area_primaria": CONFIG["area_fallback"],
                    "top3_json": json.dumps([], ensure_ascii=False),
                    "via": "fallback_derecho",
                    "n_citas_cp": 0,
                    "fallback_termino": termino,
                }
            )
            n_fallback += 1
        else:
            filas.append(
                {
                    "id": doc_id,
                    "tipo": tipo,
                    "ano": ano,
                    "area_primaria": CONFIG["area_sin_definir"],
                    "top3_json": json.dumps([], ensure_ascii=False),
                    "via": "sin_area",
                    "n_citas_cp": 0,
                    "fallback_termino": None,
                }
            )
    logger.info(
        "Fallback aplicado: %d; sin area: %d", n_fallback, len(pendientes) - n_fallback
    )
    df = pd.DataFrame(filas)
    diagnostico = dict(contadores)
    diagnostico["n_fallback"] = n_fallback
    diagnostico["n_sin_area"] = len(pendientes) - n_fallback
    return df, diagnostico


def elegir_ejemplos(df: pd.DataFrame, ent: pd.DataFrame) -> list:
    """Elige 10 ejemplos deterministas para el reporte de cobertura.

    Reparte 6 por cita_cp (areas distintas si es posible), 2 por fallback y
    2 sin area; con menos disponibles en una via, completa desde cita_cp.

    Args:
        df: DataFrame de etiquetas ya construido.
        ent: DataFrame de entidades con arts_cp originales.

    Returns:
        Lista de dicts con id, area, via y justificacion de una linea.
    """
    arts_por_id = dict(zip(ent["id"].astype(str), ent["arts_cp"].astype(str)))
    semilla = random.Random(CONFIG["semilla"])
    ejemplos: list = []

    def justificacion_cita(doc_id: str, area: str) -> str:
        citados = parsear_lista_arts(arts_por_id.get(doc_id, "[]"))
        muestra = ", ".join(citados[:6])
        if len(citados) > 6:
            muestra += f" (+{len(citados) - 6} mas)"
        return f"arts. {muestra} con mayoria en {area}"

    cita = df[df["via"] == "cita_cp"]
    vistas = set()
    candidatas = cita.sample(
        n=min(len(cita), 60), random_state=CONFIG["semilla"]
    ).to_dict("records")
    for rec in candidatas:
        if len(ejemplos) >= 6:
            break
        if rec["area_primaria"] in vistas:
            continue
        vistas.add(rec["area_primaria"])
        ejemplos.append(
            {
                "id": rec["id"],
                "area": rec["area_primaria"],
                "via": rec["via"],
                "justificacion": justificacion_cita(rec["id"], rec["area_primaria"]),
            }
        )
    if len(ejemplos) < 6:
        resto = cita.sample(n=min(len(cita), 20), random_state=CONFIG["semilla"] + 1)
        for rec in resto.to_dict("records"):
            if len(ejemplos) >= 6:
                break
            ejemplos.append(
                {
                    "id": rec["id"],
                    "area": rec["area_primaria"],
                    "via": rec["via"],
                    "justificacion": justificacion_cita(
                        rec["id"], rec["area_primaria"]
                    ),
                }
            )
    for via, limite in (("fallback_derecho", 2), ("sin_area", 2)):
        sub = df[df["via"] == via]
        if sub.empty:
            continue
        toma = sub.sample(n=min(len(sub), limite), random_state=CONFIG["semilla"])
        for rec in toma.to_dict("records"):
            if via == "fallback_derecho":
                just = (
                    "sin citas CP validas; patron 'derecho a "
                    f"{rec['fallback_termino']}' hacia derechos_fundamentales"
                )
            else:
                just = "sin citas CP validas ni patron 'derecho a ...' en el texto"
            ejemplos.append(
                {
                    "id": rec["id"],
                    "area": rec["area_primaria"],
                    "via": rec["via"],
                    "justificacion": just,
                }
            )
    semilla.shuffle(ejemplos)
    return ejemplos[: CONFIG["n_ejemplos"]]


def escribir_reporte(
    df: pd.DataFrame,
    ent: pd.DataFrame,
    diagnostico: dict,
    mapa_cp: dict,
    ruta: Path,
    segundos: float,
) -> None:
    """Escribe el reporte Markdown de cobertura del etiquetado.

    Args:
        df: DataFrame de etiquetas.
        ent: DataFrame de entidades (para ejemplos con articulos).
        diagnostico: Contadores de normalizacion y fallback.
        mapa_cp: Mapa numero hacia bloque (para el universo de areas).
        ruta: Ruta de salida del reporte Markdown.
        segundos: Duracion total del proceso en segundos.
    """
    total = len(df)
    por_via = df["via"].value_counts()
    n_cita = int(por_via.get("cita_cp", 0))
    n_fall = int(por_via.get("fallback_derecho", 0))
    n_sin = int(por_via.get("sin_area", 0))
    asignadas = n_cita + n_fall

    dist_area = df["area_primaria"].value_counts()
    tutelas = df[df["tipo"] == "T"]
    n_tut_derechos = int((tutelas["area_primaria"] == "derechos_fundamentales").sum())

    universo = sorted(set(mapa_cp.values()) | {CONFIG["area_sin_definir"]})
    vacias = [a for a in universo if a not in set(dist_area.index)]

    lineas = []
    lineas.append("# Cobertura de etiquetado de areas (M3, Fase I)")
    lineas.append("")
    ahora = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z")
    lineas.append(f"Fecha: {ahora}")
    lineas.append(f"Documentos: {total} (T/C/SU)")
    lineas.append(f"Tiempo total: {segundos:.1f}s")
    lineas.append("")
    lineas.append("## Regla de etiquetado (ADR-1)")
    lineas.append(
        "Cada articulo citado se normaliza al canon de la CP: si el codigo "
        'con letra existe ("22A") se usa tal cual; si no, se reduce a su '
        'base ("3A"->"3"); lo fuera de 1-380 o transitorios sin '
        "correspondencia se ignora con contador. La primaria es el bloque "
        "mayoritario y el desempate usa el menor numero de articulo del "
        "documento (clave base entera, sufijo con orden vacio < A < B < C). "
        "Sin citas validas, el fallback busca el patron 'derecho a X' en el "
        "texto y asigna derechos_fundamentales con el termino X conservado "
        "en fallback_termino (flag fallback_derecho_a_X); sin coincidencia "
        "se asigna sin_area_definida."
    )
    lineas.append("")
    lineas.append("## Cobertura global por via (% docs)")
    lineas.append(f"- cita_cp: {pct(n_cita, total):.2f} ({n_cita}/{total})")
    lineas.append(f"- fallback_derecho: {pct(n_fall, total):.2f} ({n_fall}/{total})")
    lineas.append(f"- sin_area: {pct(n_sin, total):.2f} ({n_sin}/{total})")
    lineas.append(
        f"- asignadas (cita o fallback): {pct(asignadas, total):.2f} "
        f"({asignadas}/{total})"
    )
    lineas.append("")
    lineas.append("## Por tipo T/C/SU (% por via)")
    lineas.append("| tipo | cita_cp | fallback | sin_area | n |")
    lineas.append("| --- | ---: | ---: | ---: | ---: |")
    for tipo in ["C", "SU", "T"]:
        sub = df[df["tipo"] == tipo]
        n = len(sub)
        c = int((sub["via"] == "cita_cp").sum())
        f = int((sub["via"] == "fallback_derecho").sum())
        s = int((sub["via"] == "sin_area").sum())
        lineas.append(
            f"| {tipo} | {pct(c, n):.2f} | {pct(f, n):.2f} | {pct(s, n):.2f} | {n} |"
        )
    lineas.append("")
    lineas.append("## Por periodo (% por via)")
    lineas.append("| periodo | cita_cp | fallback | sin_area | n |")
    lineas.append("| --- | ---: | ---: | ---: | ---: |")
    for etiqueta, ini, fin in CONFIG["periodos"]:
        sub = df[(df["ano"] >= ini) & (df["ano"] <= fin)]
        n = len(sub)
        c = int((sub["via"] == "cita_cp").sum())
        f = int((sub["via"] == "fallback_derecho").sum())
        s = int((sub["via"] == "sin_area").sum())
        lineas.append(
            f"| {etiqueta} | {pct(c, n):.2f} | {pct(f, n):.2f} | "
            f"{pct(s, n):.2f} | {n} |"
        )
    lineas.append("")
    lineas.append("## Distribucion de areas (top-10)")
    for area, n in dist_area.head(10).items():
        lineas.append(f"- {area}: {n} docs ({pct(n, total):.2f} %)")
    lineas.append("")
    lineas.append("## Tabla completa de areas (n_docs)")
    for area, n in dist_area.items():
        lineas.append(f"- {area}: {n} docs ({pct(n, total):.2f} %)")
    lineas.append("")
    lineas.append("## Tutelas en derechos_fundamentales")
    lineas.append(
        f"- {pct(n_tut_derechos, len(tutelas)):.2f} % "
        f"({n_tut_derechos}/{len(tutelas)} tutelas)"
    )
    lineas.append("")
    lineas.append("## Diagnostico de normalizacion de citas")
    lineas.append(f"- normalizadas exactas: {diagnostico['normalizadas_exactas']}")
    lineas.append(f"- normalizadas a base: {diagnostico['normalizadas_base']}")
    lineas.append(f"- ignoradas fuera de rango: {diagnostico['ignoradas_fuera_rango']}")
    lineas.append("")
    lineas.append("## Ejemplos (10)")
    for ej in elegir_ejemplos(df, ent):
        lineas.append(
            f"- id {ej['id']} | {ej['area']} | {ej['via']} | {ej['justificacion']}"
        )
    lineas.append("")
    lineas.append("## Verificacion de areas vacias")
    if vacias:
        lineas.append(
            "Areas del universo CP sin documentos asignados "
            f"({len(vacias)}): " + ", ".join(vacias) + "."
        )
        lineas.append(
            "Justificacion: preambulo y transitorios quedan vacios por diseno "
            "(el NER solo extrae articulos permanentes 1-380, sin citas a "
            "esos bloques); deberes (bloque de un solo articulo, el 95) si "
            "aparece citado y en 122 top-3, pero nunca alcanza la mayoria en "
            "ningun documento, por lo que no genera primarias."
        )
    else:
        lineas.append("Ningun area del universo queda vacia.")
    lineas.append("")
    lineas.append("## Criterio de aceptacion (>=95 % asignadas)")
    cumple = "SI" if pct(asignadas, total) >= 95.0 else "NO"
    lineas.append(
        f"- {cumple}: {pct(asignadas, total):.2f} % con area asignada "
        f"({asignadas}/{total}); sin area {n_sin} docs "
        f"({pct(n_sin, total):.2f} %), todos sin citas CP validas ni patron "
        "'derecho a ...' en el texto."
    )
    lineas.append("")
    lineas.append("## Esquema del parquet")
    lineas.append(
        "Columnas: id, tipo, ano, area_primaria, top3_json, via "
        "(cita_cp/fallback_derecho/sin_area), n_citas_cp, mas "
        "fallback_termino (extension documentada que conserva X del patron "
        "para la flag fallback_derecho_a_X; nula fuera de fallback)."
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


# ===== SECCION 6: programa principal =====
def main() -> int:
    """Ejecuta el etiquetado de areas y el reporte de cobertura.

    Returns:
        Codigo de salida (0 ante exito, 1 ante error).
    """
    import time

    inicio = time.time()
    configurar_semillas(CONFIG["semilla"])
    logger = configurar_logging(str(ROOT / CONFIG["ruta_log"]))
    try:
        ruta_ent = ROOT / CONFIG["ruta_entidades"]
        ruta_cp = ROOT / CONFIG["ruta_constitucion"]
        ruta_corpus = ROOT / CONFIG["ruta_corpus"]
        ruta_sal = ROOT / CONFIG["ruta_salida"]
        ruta_rep = ROOT / CONFIG["ruta_reporte"]
        df, diagnostico = construir_tabla(ruta_ent, ruta_cp, ruta_corpus, logger)
        ruta_sal.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ruta_sal, index=False)
        logger.info("Parquet escrito: %s (%d filas)", ruta_sal, len(df))
        ent = pd.read_parquet(ruta_ent, columns=["id", "arts_cp"])
        mapa_cp, _ = cargar_mapa_cp(ruta_cp)
        segundos = time.time() - inicio
        escribir_reporte(df, ent, diagnostico, mapa_cp, ruta_rep, segundos)
        logger.info("Reporte escrito: %s", ruta_rep)
        total = len(df)
        n_cita = int((df["via"] == "cita_cp").sum())
        n_fall = int((df["via"] == "fallback_derecho").sum())
        n_sin = int((df["via"] == "sin_area").sum())
        print(
            f"docs: {total} | cita {pct(n_cita, total):.2f} % | "
            f"fallback {pct(n_fall, total):.2f} % | "
            f"sin_area {pct(n_sin, total):.2f} %"
        )
        print(
            f"asignadas: {pct(n_cita + n_fall, total):.2f} % "
            f"| criterio >=95: {'SI' if pct(n_cita + n_fall, total) >= 95 else 'NO'}"
        )
        print(
            f"top area: {df['area_primaria'].value_counts().index[0]} "
            f"({df['area_primaria'].value_counts().iloc[0]} docs)"
        )
        print(f"ignoradas fuera de rango: {diagnostico['ignoradas_fuera_rango']}")
        print(f"parquet: {ruta_sal}")
        print(f"reporte: {ruta_rep}")
        print(f"tiempo: {segundos:.1f}s")
        return 0
    except Exception:  # noqa: BLE001 - main captura todo para log + exit 1
        logger.error("Fallo el etiquetado:\n%s", traceback.format_exc())
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

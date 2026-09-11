"""eda_ext_lemas_stems_pos.py
Fecha: 2026-09-08
Descripcion: EDA extendido E2 Juris-PLN sobre las 23.742 sentencias.
Estado del arte aplicado: lematizacion y POS con spaCy es_core_news_md,
    diversidad lexica MTLD de McCarthy-Jarvis 2010 (factor 0.72),
    stemming Snowball espanol para compresion de vocabulario.
"""

import collections
import csv
import datetime
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd

CONFIG = {
    "semilla": 42,
    "corpus_path": (
        "/home/carlos/Git/PLN/PROYECTO/SEGUNDA ENTREGA/data/corpus_sentencias.parquet"
    ),
    "modelo_spacy": "es_core_news_md",
    # NOTA DE DESVIACION RESPECTO A LA SPEC (2026-09-08):
    # La spec pedia n_process=12 y batch_size=64, pero en esta maquina
    # (31 GB RAM) esa combinacion es inviable: cada worker spaCy retiene
    # el lote en vuelo (docs de ~11k palabras con tensores) y alcanza
    # ~2,6 GB RSS, de modo que 12 workers agotan la RAM y el forkserver
    # muere con "pickle data was truncated" (reproducido dos veces, sin
    # completar ni el primer checkpoint). Alternativa de menor riesgo:
    # n_process=6 y batch_size=16 (misma imagen, mismo modelo, mismos
    # componentes desactivados y mismo algoritmo MTLD). Memoria estimada
    # del job: ~10 GB frente a >28 GB con los valores originales.
    "spacy_batch_size": 16,
    "spacy_n_process": 6,
    "spacy_disable": ["parser", "ner"],
    # FIX 2026-09-08 (robustez anti-OOM, dos fases):
    # medicion con pyarrow sobre el corpus (23.742 docs): >200k chars=1076
    # docs (4,5 %), >500k=114, >1M=26, max=3.007.155 chars (id 16989).
    # UMBRAL=200k: deja el 95,5 % en el pipe multiproceso (lote en vuelo
    # maximo ~16x200k=3,2M chars por worker, seguro) y aisla la cola
    # pesada (4,5 %) a monoproceso secuencial (un doc en RAM cada vez).
    # Con 500k seguirian 114 docs gigantes en el pipe, suficiente para
    # repetir el OOM que mato los workers hacia el doc 7000.
    "spacy_long_umbral_chars": 200_000,
    # Watchdog de la fase pipe: aborta con error explicito si no hay
    # checkpoint en este tiempo (evita colgarse eternamente si los
    # workers mueren en silencio por OOM).
    "watchdog_pipe_seg": 20 * 60,
    # Checkpoints de la fase monoproceso de largos (docs lentos: cada
    # uno puede tardar minutos, asi que se checkpointa mas seguido).
    "checkpoint_largos_cada": 50,
    "mtld_umbral": 0.72,
    "checkpoint_cada": 500,
    "muestra_n": 50,
    "longitud_minima_muestra": 4,
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "resultados", "eda_extendido", "lemas_stems_pos")
EXP_DIR = os.path.join(BASE_DIR, "experimentos")
LOG_PATH = os.path.join(EXP_DIR, "eda_ext_lemas_stems_pos.log")
OMITIDOS_LOG = os.path.join(EXP_DIR, "eda_ext_docs_omitidos.log")
ESTADO_PATH = os.path.join(OUT_DIR, "estado.json")
LEMAS_PARCIAL = os.path.join(OUT_DIR, "contadores_lemas_parcial.parquet")
STEMS_PARCIAL = os.path.join(OUT_DIR, "contadores_stems_parcial.parquet")
TOKENS_PARCIAL = os.path.join(OUT_DIR, "contadores_tokens_parcial.parquet")
POS_GLOBAL_PARCIAL = os.path.join(OUT_DIR, "pos_global_parcial.parquet")
POS_TIPO_ANO_PARCIAL = os.path.join(OUT_DIR, "pos_tipo_ano_parcial.parquet")


def configurar_semillas(semilla):
    """Fija las semillas de aleatoriedad para reproducibilidad.

    Args:
        semilla: Valor entero de semilla.

    Returns:
        None.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def calcular_mtld(tokens, umbral=0.72):
    """Calcula la diversidad lexica MTLD (McCarthy-Jarvis 2010).

    Promedia el recorrido hacia adelante y hacia atras con el
    factor de estabilidad TTR indicado. Implementacion estandar:
    se cuenta un factor cada vez que el TTR cae a `umbral` o menos
    y el segmento final aporta un factor parcial.

    Args:
        tokens: Secuencia de cadenas (formas superficiales).
        umbral: Umbral TTR de corte (0.72 por defecto).

    Returns:
        Valor MTLD (float). 0.0 si no hay tokens.
    """
    if not tokens:
        return 0.0

    def recorrido(secuencia):
        tipos = set()
        n_palabras = 0
        factores = 0.0
        for tok in secuencia:
            n_palabras += 1
            tipos.add(tok)
            ttr = len(tipos) / n_palabras
            if ttr <= umbral:
                factores += 1.0
                tipos = set()
                n_palabras = 0
        if n_palabras > 0:
            ttr = len(tipos) / n_palabras
            factores += (1.0 - ttr) / (1.0 - umbral)
        if factores == 0.0:
            return float(len(secuencia))
        return len(secuencia) / factores

    lista = list(tokens)
    adelante = recorrido(lista)
    atras = recorrido(lista[::-1])
    return (adelante + atras) / 2.0


def asegurar_directorios():
    """Crea los directorios de salida si no existen.

    Returns:
        None.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)


def escribir_log(mensaje):
    """Agrega una linea con marca temporal al log de progreso.

    Args:
        mensaje: Texto a registrar.

    Returns:
        None.
    """
    marca = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{marca}] {mensaje}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(linea)
    print(linea.strip(), flush=True)


def escribir_omitido(doc_id, error):
    """Registra un documento omitido por fallo de procesamiento.

    Args:
        doc_id: Identificador del documento que fallo.
        error: Excepcion capturada.

    Returns:
        None.
    """
    marca = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    detalle = f"{type(error).__name__}: {error}".replace("\n", " ").strip()
    linea = f"[{marca}] id={doc_id} {detalle}\n"
    os.makedirs(EXP_DIR, exist_ok=True)
    with open(OMITIDOS_LOG, "a", encoding="utf-8") as fh:
        fh.write(linea)


def formatear_eta(seg_por_doc, restantes):
    """Estima el tiempo restante en formato legible.

    Args:
        seg_por_doc: Segundos promedio por documento.
        restantes: Numero de documentos pendientes.

    Returns:
        Cadena con formato H:MM:SS o indeterminado.
    """
    if seg_por_doc is None or restantes <= 0:
        return "00:00:00"
    total = int(seg_por_doc * restantes)
    horas = total // 3600
    minutos = (total % 3600) // 60
    segundos = total % 60
    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"


def cargar_estado(total_docs):
    """Carga el estado de reanudacion o crea uno nuevo.

    Args:
        total_docs: Numero total de documentos del corpus.

    Returns:
        Diccionario con claves spacy_hasta y stems_hasta.
    """
    if os.path.exists(ESTADO_PATH):
        try:
            with open(ESTADO_PATH, encoding="utf-8") as fh:
                estado = json.load(fh)
            spacy_hasta = int(estado.get("spacy_hasta", 0))
            stems_hasta = int(estado.get("stems_hasta", 0))
            return {"spacy_hasta": spacy_hasta, "stems_hasta": stems_hasta}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {"spacy_hasta": 0, "stems_hasta": 0}


def guardar_estado(spacy_hasta, stems_hasta, total_docs):
    """Guarda el estado de avance de forma atomica.

    Args:
        spacy_hasta: Documentos procesados en la pasada spaCy.
        stems_hasta: Documentos procesados en la pasada stems.
        total_docs: Total de documentos.

    Returns:
        None.
    """
    estado = {
        "spacy_hasta": int(spacy_hasta),
        "stems_hasta": int(stems_hasta),
        "total_docs": int(total_docs),
        "actualizado": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    tmp = ESTADO_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(estado, fh)
    os.replace(tmp, ESTADO_PATH)


def cargar_corpus(ruta):
    """Carga el corpus parquet y lo ordena por id para determinismo.

    Args:
        ruta: Ruta al parquet Hive particionado.

    Returns:
        DataFrame con columnas id, tipo, numero, ano, texto, texto_limpio.
    """
    df = pd.read_parquet(ruta)
    df = df.copy()
    df["texto"] = df["texto"].fillna("").astype(str)
    df["texto_limpio"] = df["texto_limpio"].fillna("").astype(str)
    df["tipo"] = df["tipo"].astype(str)
    df["ano"] = df["ano"].astype(str)
    df = df.sort_values("id").reset_index(drop=True)
    return df


def contador_a_parquet(contador, ruta, col_nombre):
    """Guarda un Counter como parquet ordenado por frecuencia.

    Args:
        contador: Objeto Counter con las frecuencias.
        ruta: Ruta de salida parquet.
        col_nombre: Nombre de la columna de la forma.

    Returns:
        None.
    """
    if contador:
        items = sorted(contador.items(), key=lambda kv: kv[1], reverse=True)
        df = pd.DataFrame(items, columns=[col_nombre, "freq"])
    else:
        df = pd.DataFrame(columns=[col_nombre, "freq"])
    df.to_parquet(ruta, index=False)


def parquet_a_contador(ruta):
    """Carga un parquet de frecuencias a un Counter.

    Args:
        ruta: Ruta del parquet con columnas (forma, freq).

    Returns:
        Counter reconstruido (vacio si el archivo no existe).
    """
    if not os.path.exists(ruta):
        return collections.Counter()
    df = pd.read_parquet(ruta)
    if df.empty:
        return collections.Counter()
    col = df.columns[0]
    return collections.Counter(dict(zip(df[col].astype(str), df["freq"].astype(int))))


def guardar_parciales_spacy(lemas, pos_global, pos_tipo_ano):
    """Persiste los acumulados parciales de la pasada spaCy.

    Args:
        lemas: Counter de lemas.
        pos_global: Counter global de etiquetas POS.
        pos_tipo_ano: Dict con clave (tipo, ano) y Counter de POS.

    Returns:
        None.
    """
    contador_a_parquet(lemas, LEMAS_PARCIAL, "lema")
    contador_a_parquet(pos_global, POS_GLOBAL_PARCIAL, "pos")
    filas = []
    for (tipo, ano), cnt in pos_tipo_ano.items():
        for pos, freq in cnt.items():
            filas.append((str(tipo), str(ano), str(pos), int(freq)))
    df = pd.DataFrame(filas, columns=["tipo", "ano", "pos", "freq"])
    df.to_parquet(POS_TIPO_ANO_PARCIAL, index=False)


def cargar_parciales_spacy():
    """Recupera los acumulados parciales de la pasada spaCy.

    Returns:
        Tupla (lemas, pos_global, pos_tipo_ano).
    """
    lemas = parquet_a_contador(LEMAS_PARCIAL)
    pos_global = parquet_a_contador(POS_GLOBAL_PARCIAL)
    pos_tipo_ano = {}
    if os.path.exists(POS_TIPO_ANO_PARCIAL):
        df = pd.read_parquet(POS_TIPO_ANO_PARCIAL)
        for row in df.itertuples(index=False):
            clave = (str(row.tipo), str(row.ano))
            pos_tipo_ano.setdefault(clave, collections.Counter())[str(row.pos)] = int(
                row.freq
            )
    return lemas, pos_global, pos_tipo_ano


def guardar_parciales_stems(stems, tokens):
    """Persiste los acumulados parciales de la pasada de stems.

    Args:
        stems: Counter de stems.
        tokens: Counter de tokens limpios.

    Returns:
        None.
    """
    contador_a_parquet(stems, STEMS_PARCIAL, "stem")
    contador_a_parquet(tokens, TOKENS_PARCIAL, "token")


def pasada_spacy(df, estado, total_docs):
    """Ejecuta la pasada A: spaCy sobre texto crudo con checkpoints.

    Diseno en DOS FASES (FIX 2026-09-08, robustez anti-OOM):
    fase A: docs cortos (len <= UMBRAL) por nlp.pipe multiproceso;
    fase B: docs largos en MONOPROCESO secuencial con nlp() directo,
    ordenados de menor a mayor longitud. Ambas fases acumulan en los
    mismos contadores, escriben checkpoints mtld_ckpt_*.parquet en el
    mismo directorio y actualizan el mismo estado.json.
    Un watchdog aborta con error explicito si la fase pipe no produce
    checkpoint en 20 min (workers muertos en silencio, sin colgarse).

    Reanudacion por id (FIX 2026-09-08): la fuente de verdad son los ids
    presentes en los checkpoints, no el cursor posicional spacy_hasta
    (que avanzo sobre 28 omitidos EOFError y 78 docs cuyo bloque en
    memoria se perdio al morir el pipe: checkpoints con 6990 filas para
    spacy_hasta=7096). Los pendientes se calculan por diferencia de ids,
    asi los docs largos y los omitidos falsos se recuperan solos.

    Args:
        df: DataFrame del corpus ordenado por id.
        estado: Dict con el avance previo (spacy_hasta).
        total_docs: Total de documentos.

    Returns:
        Tupla (lemas, pos_global, pos_tipo_ano, segundos_pasada).
    """
    import glob
    import threading

    import spacy

    umbral = CONFIG["spacy_long_umbral_chars"]
    cada = CONFIG["checkpoint_cada"]
    cada_largos = CONFIG["checkpoint_largos_cada"]
    espera_watchdog = CONFIG["watchdog_pipe_seg"]
    umbral_mtld = CONFIG["mtld_umbral"]

    textos = df["texto"].tolist()
    ids = df["id"].astype(str).tolist()
    tipos = df["tipo"].tolist()
    anos = df["ano"].tolist()

    # Reconciliacion por id contra los checkpoints existentes.
    hechas = set()
    for ruta_ckpt in sorted(glob.glob(os.path.join(OUT_DIR, "mtld_ckpt_*.parquet"))):
        try:
            df_ckpt = pd.read_parquet(ruta_ckpt, columns=["id"])
            hechas.update(df_ckpt["id"].astype(str).tolist())
        except (OSError, ValueError):
            continue

    pendientes_cortos = []
    pendientes_largos = []
    for pos in range(total_docs):
        if ids[pos] in hechas:
            continue
        longitud = len(textos[pos])
        if longitud <= umbral:
            pendientes_cortos.append(pos)
        else:
            pendientes_largos.append((longitud, pos))
    pendientes_largos.sort()
    n_largos_total = sum(1 for t in textos if len(t) > umbral)
    escribir_log(
        f"pasada=spacy reconciliada hechas={len(hechas)} "
        f"spacy_hasta_previo={estado.get('spacy_hasta', 0)} "
        f"pend_cortos={len(pendientes_cortos)} "
        f"pend_largos={len(pendientes_largos)}/{n_largos_total} "
        f"umbral={umbral}"
    )

    lemas, pos_global, pos_tipo_ano = cargar_parciales_spacy()

    if not pendientes_cortos and not pendientes_largos:
        escribir_log("pasada=spacy ya completa, se omite")
        return lemas, pos_global, pos_tipo_ano, 0.0

    nlp = spacy.load(CONFIG["modelo_spacy"], disable=CONFIG["spacy_disable"])
    # El corpus contiene docs de hasta ~3M caracteres y el limite por
    # defecto (1M) aborta con E088/E871. Con parser y NER deshabilitados
    # es seguro ampliarlo a 5M.
    nlp.max_length = 5_000_000

    t0 = time.time()
    procesados_run = 0
    bloque_mtld = []
    base_hechas = len(hechas)

    def emitir_checkpoint(nombre, fase):
        """Persiste acumulados MTLD y estado para el avance indicado."""
        pd.DataFrame(bloque_mtld).to_parquet(os.path.join(OUT_DIR, nombre), index=False)
        guardar_parciales_spacy(lemas, pos_global, pos_tipo_ano)
        hechos = base_hechas + procesados_run
        guardar_estado(hechos, int(estado.get("stems_hasta", 0)), total_docs)
        elapsed = time.time() - t0
        vel = procesados_run / elapsed if elapsed > 0 and procesados_run > 0 else 0.0
        seg_doc = elapsed / procesados_run if procesados_run > 0 else 0.0
        eta = formatear_eta(seg_doc, total_docs - hechos)
        escribir_log(
            f"pasada=spacy fase={fase} hechos={hechos}/{total_docs} "
            f"({100.0 * hechos / total_docs:.1f}%) "
            f"vel={vel:.2f} docs/s ETA={eta} ckpt={nombre}"
        )
        ultimo_ckpt[0] = time.time()

    def procesar_doc(doc, idx):
        """Acumula lemas, POS y MTLD de un doc ya analizado por spaCy."""
        formas = []
        for tok in doc:
            if tok.is_alpha:
                formas.append(tok.text.lower())
                lema = tok.lemma_.lower().strip()
                if lema:
                    lemas[lema] += 1
                pos_global[tok.pos_] += 1
                clave = (str(tipos[idx]), str(anos[idx]))
                if clave not in pos_tipo_ano:
                    pos_tipo_ano[clave] = collections.Counter()
                pos_tipo_ano[clave][tok.pos_] += 1
        n_tokens = len(formas)
        mtld = calcular_mtld(formas, umbral=umbral_mtld)
        bloque_mtld.append(
            {
                "id": str(ids[idx]),
                "tipo": str(tipos[idx]),
                "ano": str(anos[idx]),
                "mtld": float(mtld),
                "n_tokens": int(n_tokens),
            }
        )

    ultimo_ckpt = [time.time()]
    detener = threading.Event()

    def vigilar_pipe():
        """Watchdog: aborta si la fase pipe no checkpointa en 20 min."""
        while not detener.wait(60):
            if time.time() - ultimo_ckpt[0] > espera_watchdog:
                escribir_log(
                    "WATCHDOG pipe: sin checkpoint en "
                    f"{espera_watchdog // 60} min, aborto explicito "
                    "(fase pipe colgada, posibles workers OOM)"
                )
                os._exit(3)

    # ===== FASE A: docs cortos por nlp.pipe multiproceso =====
    if pendientes_cortos:
        hilo = threading.Thread(target=vigilar_pipe, daemon=True)
        hilo.start()
        try:
            textos_cortos = [textos[pos] for pos in pendientes_cortos]
            pipe = nlp.pipe(
                textos_cortos,
                batch_size=CONFIG["spacy_batch_size"],
                n_process=CONFIG["spacy_n_process"],
                disable=CONFIG["spacy_disable"],
            )
            n_cortos = len(pendientes_cortos)
            for rel, doc in enumerate(pipe):
                pos = pendientes_cortos[rel]
                try:
                    procesar_doc(doc, pos)
                except Exception as exc_doc:  # noqa: BLE001 - guard por doc
                    escribir_omitido(ids[pos], exc_doc)
                    continue
                procesados_run += 1
                if (procesados_run % cada == 0) or (rel + 1 == n_cortos):
                    hechos = base_hechas + procesados_run
                    emitir_checkpoint(f"mtld_ckpt_a_{hechos:06d}.parquet", "pipe")
                    bloque_mtld.clear()
        except SystemExit:
            raise
        except Exception as exc_pipe:
            if bloque_mtld:
                hechos = base_hechas + procesados_run
                emitir_checkpoint(f"mtld_ckpt_a_{hechos:06d}.parquet", "pipe")
                bloque_mtld.clear()
            else:
                guardar_estado(
                    base_hechas + procesados_run,
                    int(estado.get("stems_hasta", 0)),
                    total_docs,
                )
            detener.set()
            escribir_log(f"fase=pipe aborta por error del pipe: {exc_pipe!r}")
            raise
        detener.set()
        escribir_log(f"fase=pipe terminada cortos_procesados={procesados_run}")

    # ===== FASE B: docs largos en MONOPROCESO secuencial =====
    if pendientes_largos:
        longitudes = [lon for lon, _ in pendientes_largos]
        escribir_log(
            f"fase=largos inicio n={len(pendientes_largos)} "
            f"min={min(longitudes)} max={max(longitudes)} chars "
            f"orden=ascendente ckpt_cada={cada_largos}"
        )
        for k, (lon, pos) in enumerate(pendientes_largos, start=1):
            t_doc = time.time()
            try:
                doc = nlp(textos[pos])
            except Exception as exc_doc:  # noqa: BLE001 - guard por doc largo
                escribir_omitido(ids[pos], exc_doc)
                continue
            try:
                procesar_doc(doc, pos)
            except Exception as exc_doc:  # noqa: BLE001 - guard por doc largo
                escribir_omitido(ids[pos], exc_doc)
                continue
            procesados_run += 1
            seg_doc = time.time() - t_doc
            if k == 1 or (k % cada_largos == 0) or (k == len(pendientes_largos)):
                hechos = base_hechas + procesados_run
                emitir_checkpoint(f"mtld_ckpt_b_{hechos:06d}.parquet", "largos")
                bloque_mtld.clear()
                escribir_log(
                    f"fase=largos k={k}/{len(pendientes_largos)} "
                    f"ultimo_id={ids[pos]} len={lon} seg_doc={seg_doc:.1f}"
                )
        escribir_log("fase=largos terminada")

    segundos = time.time() - t0
    return lemas, pos_global, pos_tipo_ano, segundos


def pasada_stems(df, estado, total_docs, tiempo_spacy):
    """Ejecuta la pasada B: SnowballStemmer sobre texto_limpio.

    Cuenta tokens limpios y sus stems. Registra checkpoints del
    acumulado cada 500 documentos para permitir reanudacion.

    Args:
        df: DataFrame del corpus ordenado por id.
        estado: Dict con el avance previo (stems_hasta).
        total_docs: Total de documentos.
        tiempo_spacy: Segundos usados en la pasada spaCy (para estado).

    Returns:
        Tupla (stems, tokens, segundos_pasada).
    """
    from nltk.stem.snowball import SnowballStemmer

    inicio = int(estado.get("stems_hasta", 0))
    cada = CONFIG["checkpoint_cada"]
    spacy_hasta = int(estado.get("spacy_hasta", inicio))

    stems = parquet_a_contador(STEMS_PARCIAL)
    tokens = parquet_a_contador(TOKENS_PARCIAL)

    if inicio >= total_docs:
        escribir_log("pasada=stems ya completa, se omite")
        return stems, tokens, 0.0

    stemmer = SnowballStemmer("spanish")
    textos_limpios = df["texto_limpio"].tolist()

    t0 = time.time()
    for hechos, idx in enumerate(range(inicio, total_docs), start=1):
        toks = textos_limpios[idx].split()
        for tok in toks:
            tokens[tok] += 1
            stems[stemmer.stem(tok)] += 1
        fin_global = idx + 1
        if (hechos % cada == 0) or (fin_global == total_docs):
            guardar_parciales_stems(stems, tokens)
            guardar_estado(spacy_hasta, fin_global, total_docs)
            elapsed = time.time() - t0
            vel = hechos / elapsed if elapsed > 0 else 0.0
            seg_doc = elapsed / hechos if hechos > 0 else 0.0
            eta = formatear_eta(seg_doc, total_docs - fin_global)
            escribir_log(
                f"pasada=stems docs={fin_global}/{total_docs} "
                f"({100.0 * fin_global / total_docs:.1f}%) "
                f"vel={vel:.2f} docs/s ETA={eta}"
            )
    segundos = time.time() - t0
    return stems, tokens, segundos


def concatenar_mtld(total_docs):
    """Concatena los checkpoints MTLD en un unico DataFrame.

    Args:
        total_docs: Total esperado de documentos.

    Returns:
        DataFrame con columnas id, tipo, ano, mtld, n_tokens.
    """
    import glob

    patron = os.path.join(OUT_DIR, "mtld_ckpt_*.parquet")
    archivos = sorted(glob.glob(patron))
    if not archivos:
        raise FileNotFoundError("No se encontraron checkpoints MTLD")
    partes = [pd.read_parquet(a) for a in archivos]
    df = pd.concat(partes, ignore_index=True)
    if len(df) != total_docs:
        raise ValueError(f"MTLD incompleto: {len(df)} filas para {total_docs} docs")
    return df


def guardar_salidas_finales(
    df_mtld, lemas, pos_global, pos_tipo_ano, stems, tokens, seg_spacy, seg_stems
):
    """Escribe todos los artefactos finales del EDA extendido.

    Args:
        df_mtld: DataFrame por documento con MTLD.
        lemas: Counter de lemas.
        pos_global: Counter global de POS.
        pos_tipo_ano: Dict (tipo, ano) a Counter de POS.
        stems: Counter de stems.
        tokens: Counter de tokens limpios.
        seg_spacy: Segundos de la pasada spaCy.
        seg_stems: Segundos de la pasada de stems.

    Returns:
        Diccionario con el resumen escrito en RESUMEN.json.
    """
    contador_a_parquet(lemas, os.path.join(OUT_DIR, "contadores_lemas.parquet"), "lema")
    contador_a_parquet(stems, os.path.join(OUT_DIR, "contadores_stems.parquet"), "stem")

    df_mtld.to_parquet(os.path.join(OUT_DIR, "mtld_por_doc.parquet"), index=False)

    pos_items = sorted(pos_global.items(), key=lambda kv: kv[1], reverse=True)
    with open(
        os.path.join(OUT_DIR, "pos_global.csv"), "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["pos", "freq"])
        writer.writerows(pos_items)

    with open(
        os.path.join(OUT_DIR, "pos_por_tipo_ano.csv"), "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["tipo", "ano", "pos", "freq"])
        for tipo, ano in sorted(pos_tipo_ano):
            for pos, freq in pos_tipo_ano[(tipo, ano)].most_common():
                writer.writerow([tipo, ano, pos, freq])

    tiempos = {
        "pasada_spacy_seg": round(seg_spacy, 2),
        "pasada_stems_seg": round(seg_stems, 2),
        "total_seg": round(seg_spacy + seg_stems, 2),
    }
    with open(os.path.join(OUT_DIR, "tiempos.json"), "w", encoding="utf-8") as fh:
        json.dump(tiempos, fh, indent=2)

    muestra = construir_muestra_calidad(tokens)
    muestra.to_csv(os.path.join(OUT_DIR, "muestra_calidad.csv"), index=False)

    n_tok = len(tokens)
    n_lem = len(lemas)
    n_stm = len(stems)
    top10 = [[pos, int(freq)] for pos, freq in pos_items[:10]]
    resumen = {
        "n_docs": len(df_mtld),
        "vocab_tokens": int(n_tok),
        "vocab_lemas": int(n_lem),
        "vocab_stems": int(n_stm),
        "ratio_lemas_tokens": round(n_lem / n_tok, 4) if n_tok else 0.0,
        "ratio_stems_tokens": round(n_stm / n_tok, 4) if n_tok else 0.0,
        "ratio_stems_lemas": round(n_stm / n_lem, 4) if n_lem else 0.0,
        "tiempos_seg": tiempos,
        "top10_pos": top10,
        "mtld_media": round(float(df_mtld["mtld"].mean()), 3),
        "mtld_mediana": round(float(df_mtld["mtld"].median()), 3),
        "mtld_min": round(float(df_mtld["mtld"].min()), 3),
        "mtld_max": round(float(df_mtld["mtld"].max()), 3),
    }
    with open(os.path.join(OUT_DIR, "RESUMEN.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    return resumen


def construir_muestra_calidad(tokens):
    """Construye la muestra de 50 terminos para verificacion manual.

    Selecciona los 50 tokens mas frecuentes con longitud minima
    configurada (heuristica que deja fuera stopwords cortas y deja
    vocabulario juridico dominante) y calcula lema con spaCy y stem
    con Snowball para cada uno.

    Args:
        tokens: Counter de tokens limpios.

    Returns:
        DataFrame con columnas token, freq, lema, stem.
    """
    import spacy
    from nltk.stem.snowball import SnowballStemmer

    minimo = CONFIG["longitud_minima_muestra"]
    candidatos = [
        (tok, freq) for tok, freq in tokens.most_common() if len(tok) >= minimo
    ][: CONFIG["muestra_n"]]
    nlp = spacy.load(CONFIG["modelo_spacy"], disable=CONFIG["spacy_disable"])
    stemmer = SnowballStemmer("spanish")
    filas = []
    for tok, freq in candidatos:
        doc = nlp(tok)
        lema = doc[0].lemma_.lower().strip() if len(doc) else tok
        filas.append(
            {
                "token": tok,
                "freq": int(freq),
                "lema": lema,
                "stem": stemmer.stem(tok),
            }
        )
    return pd.DataFrame(filas, columns=["token", "freq", "lema", "stem"])


def main():
    """Orquesta las tres pasadas del EDA extendido con reanudacion.

    Pasada A: spaCy sobre texto crudo (lemas, POS, MTLD).
    Pasada B: SnowballStemmer sobre texto_limpio (stems y tokens).
    Pasada C: registro de tiempos y escritura de salidas finales.

    Returns:
        None.
    """
    configurar_semillas(CONFIG["semilla"])
    asegurar_directorios()
    print(f"Corpus: {CONFIG['corpus_path']}", flush=True)
    print(f"Salida: {OUT_DIR}", flush=True)

    df = cargar_corpus(CONFIG["corpus_path"])
    total_docs = len(df)
    print(f"Documentos cargados: {total_docs}", flush=True)
    print(df.head(2)[["id", "tipo", "ano"]].to_string(), flush=True)

    estado = cargar_estado(total_docs)
    escribir_log(
        f"inicio spacy_hasta={estado['spacy_hasta']} "
        f"stems_hasta={estado['stems_hasta']} total={total_docs}"
    )

    # ===== PASADA A: spaCy sobre texto CRUDO (con tildes) =====
    lemas, pos_global, pos_tipo_ano, seg_spacy = pasada_spacy(df, estado, total_docs)
    estado = cargar_estado(total_docs)

    # ===== PASADA B: SnowballStemmer sobre texto_limpio =====
    stems, tokens, seg_stems = pasada_stems(df, estado, total_docs, seg_spacy)
    estado = cargar_estado(total_docs)

    # ===== PASADA C: tiempos y salidas finales =====
    # Suma tiempos acumulados si hubo reanudacion previa de stems.
    df_mtld = concatenar_mtld(total_docs)
    resumen = guardar_salidas_finales(
        df_mtld,
        lemas,
        pos_global,
        pos_tipo_ano,
        stems,
        tokens,
        seg_spacy,
        seg_stems,
    )
    guardar_estado(total_docs, total_docs, total_docs)
    escribir_log(
        f"fin total_seg={seg_spacy + seg_stems:.1f} "
        f"vocab_tokens={resumen['vocab_tokens']} "
        f"vocab_lemas={resumen['vocab_lemas']} "
        f"vocab_stems={resumen['vocab_stems']}"
    )
    print("RESUMEN:", json.dumps(resumen, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - el protocolo exige traceback y exit 1
        traceback.print_exc()
        sys.exit(1)

"""05_lematizacion.py
Fecha: 2026-09-08
Descripcion: Lematizacion E2 Juris-PLN sobre las 23.742 sentencias para M4-clasico
    (ADR-6: TF-IDF sobre lemas). Por documento, sobre `texto` crudo, guarda la
    secuencia de lemas (lemma_.lower(), solo alfabeticos, unidos por espacio)
    hacia data/lemas_por_doc.parquet (id, lemas_texto, n_lemas) con verificacion
    de consistencia contra contadores_lemas.parquet del EDA.
Estado del arte aplicado: diseno robusto portado de D1
    (eda_ext_lemas_stems_pos.py, PRIMERA ENTREGA): dos fases pipe<=200k chars
    multiproceso mas monoproceso secuencial para largos, watchdog con
    os._exit(3), checkpoints c/500 por id, reanudacion por diferencia de ids,
    max_length 5M, n_process=6, batch_size=16, parser y NER deshabilitados.
    No recalcula MTLD, POS ni stems (ya existen en el EDA).
"""

import collections
import datetime
import glob
import json
import os
import random
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd

CONFIG = {
    "semilla": 42,
    "corpus_rel": os.path.join("data", "corpus_sentencias.parquet"),
    "salida_final_rel": os.path.join("data", "lemas_por_doc.parquet"),
    "out_dir_rel": os.path.join("resultados", "lematizacion"),
    "log_rel": os.path.join("experimentos", "lematizacion.log"),
    "omitidos_rel": os.path.join("experimentos", "lematizacion_omitidos.log"),
    # Referencia de solo lectura (no se escribe nada en PRIMERA ENTREGA).
    "referencia_contadores": (
        "/home/carlos/Git/PLN/PROYECTO/PRIMERA ENTREGA/resultados/"
        "eda_extendido/lemas_stems_pos/contadores_lemas.parquet"
    ),
    "modelo_spacy": "es_core_news_md",
    # NOTA PORTADA DE D1 (2026-09-08):
    # La spec original pedia n_process=12 y batch_size=64, pero en esta
    # maquina (31 GB RAM) esa combinacion es inviable: cada worker spaCy
    # retiene el lote en vuelo (docs de ~11k palabras con tensores) y
    # alcanza ~2,6 GB RSS, de modo que 12 workers agotan la RAM y el
    # forkserver muere con "pickle data was truncated". Alternativa de
    # menor riesgo (misma imagen, mismo modelo, mismos componentes
    # desactivados): n_process=6 y batch_size=16. Memoria estimada ~10 GB.
    # FIX ORQUESTADOR (2026-09-08 14:00): n_process 6->4, batch 16->12 por
    # OOM recurrente al concurrir con llama-server/opencode (2 OOM: workers
    # muertos en silencio -> EOFError en el pipe -> padre colgado; el
    # watchdog no lo saco). Misma imagen/modelo/componentes. ~7 GB.
    "spacy_batch_size": 12,
    "spacy_n_process": 4,
    "spacy_disable": ["parser", "ner"],
    # FIX PORTADO DE D1 (robustez anti-OOM, dos fases):
    # >200k chars=1076 docs (4,5 %), >500k=114, >1M=26, max~3M chars.
    # UMBRAL=200k deja el 95,5 % en el pipe multiproceso y aisla la cola
    # pesada a monoproceso secuencial (un doc en RAM cada vez).
    "spacy_long_umbral_chars": 200_000,
    # Watchdog de la fase pipe: aborta con error explicito si no hay
    # checkpoint en este tiempo (evita colgarse si los workers mueren por
    # OOM en silencio).
    "watchdog_pipe_seg": 20 * 60,
    # Checkpoints de la fase monoproceso de largos (docs lentos: cada uno
    # puede tardar minutos, asi que se checkpointa mas seguido que en pipe;
    # cumple la spec c/500 al ser mas frecuente, no menos).
    "checkpoint_cada": 500,
    "checkpoint_largos_cada": 50,
    "muestra_verificacion_n": 5,
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, CONFIG["corpus_rel"])
SALIDA_FINAL = os.path.join(BASE_DIR, CONFIG["salida_final_rel"])
OUT_DIR = os.path.join(BASE_DIR, CONFIG["out_dir_rel"])
LOG_PATH = os.path.join(BASE_DIR, CONFIG["log_rel"])
OMITIDOS_LOG = os.path.join(BASE_DIR, CONFIG["omitidos_rel"])
ESTADO_PATH = os.path.join(OUT_DIR, "estado.json")
CONTADOR_PARCIAL = os.path.join(OUT_DIR, "contador_lemas_parcial.parquet")
VERIFICACION_JSON = os.path.join(OUT_DIR, "verificacion_consistencia.json")


def configurar_semillas(semilla):
    """Fija las semillas de aleatoriedad para reproducibilidad.

    Args:
        semilla: Valor entero de semilla.

    Returns:
        None.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def asegurar_directorios():
    """Crea los directorios de salida si no existen.

    Returns:
        None.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SALIDA_FINAL), exist_ok=True)


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
    with open(OMITIDOS_LOG, "a", encoding="utf-8") as fh:
        fh.write(linea)


def formatear_eta(seg_por_doc, restantes):
    """Estima el tiempo restante en formato legible.

    Args:
        seg_por_doc: Segundos promedio por documento.
        restantes: Numero de documentos pendientes.

    Returns:
        Cadena con formato H:MM:SS.
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
        Diccionario con clave lemas_hasta.
    """
    if os.path.exists(ESTADO_PATH):
        try:
            with open(ESTADO_PATH, encoding="utf-8") as fh:
                estado = json.load(fh)
            return {"lemas_hasta": int(estado.get("lemas_hasta", 0))}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {"lemas_hasta": 0}


def guardar_estado(lemas_hasta, total_docs):
    """Guarda el estado de avance de forma atomica.

    Args:
        lemas_hasta: Documentos procesados acumulados.
        total_docs: Total de documentos.

    Returns:
        None.
    """
    estado = {
        "lemas_hasta": int(lemas_hasta),
        "total_docs": int(total_docs),
        "actualizado": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    tmp = ESTADO_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(estado, fh)
    os.replace(tmp, ESTADO_PATH)


def cargar_corpus(ruta):
    """Carga el corpus parquet y lo ordena por id para determinismo.

    Replica el orden de D1 (sort por id) para que el vocabulario derivado
    sea comparable con los contadores del EDA.

    Args:
        ruta: Ruta al parquet Hive particionado por tipo.

    Returns:
        DataFrame con columnas id, tipo, numero, ano, texto, texto_limpio.
    """
    df = pd.read_parquet(ruta)
    df = df.copy()
    df["texto"] = df["texto"].fillna("").astype(str)
    df = df.sort_values("id").reset_index(drop=True)
    return df


def contador_a_parquet(contador, ruta):
    """Guarda un Counter como parquet ordenado por frecuencia.

    Args:
        contador: Objeto Counter con las frecuencias.
        ruta: Ruta de salida parquet.

    Returns:
        None.
    """
    if contador:
        items = sorted(contador.items(), key=lambda kv: kv[1], reverse=True)
        df = pd.DataFrame(items, columns=["lema", "freq"])
    else:
        df = pd.DataFrame(columns=["lema", "freq"])
    df.to_parquet(ruta, index=False)


def parquet_a_contador(ruta):
    """Carga un parquet de frecuencias a un Counter.

    Args:
        ruta: Ruta del parquet con columnas (lema, freq).

    Returns:
        Counter reconstruido (vacio si el archivo no existe).
    """
    if not os.path.exists(ruta):
        return collections.Counter()
    df = pd.read_parquet(ruta)
    if df.empty:
        return collections.Counter()
    return collections.Counter(
        dict(zip(df["lema"].astype(str), df["freq"].astype(int)))
    )


def pasada_lemas(df, estado, total_docs):
    """Ejecuta la lematizacion sobre texto crudo con checkpoints por id.

    Diseno portado de D1 (dos fases): fase A con docs cortos
    (len <= UMBRAL) por nlp.pipe multiproceso; fase B con docs largos en
    MONOPROCESO secuencial ordenados de menor a mayor longitud. Ambas
    fases acumulan las mismas secuencias, escriben checkpoints
    lemas_ckpt_*.parquet en el mismo directorio y actualizan el mismo
    estado.json. Un watchdog aborta con error explicito si la fase pipe
    no produce checkpoint en 20 min.

    Reanudacion por id (portado de D1): la fuente de verdad son los ids
    presentes en los checkpoints, no el cursor posicional (que puede
    avanzar sobre bloques perdidos si el pipe muere). Los pendientes se
    calculan por diferencia de ids.

    Regla de lemas (identica a D1 para garantizar consistencia de
    vocabulario): por cada token con tok.is_alpha se toma
    tok.lemma_.lower().strip() y se conserva si no es vacio; la secuencia
    del documento es la union con espacio simple. No se recalculan MTLD,
    POS ni stems.

    Args:
        df: DataFrame del corpus ordenado por id.
        estado: Dict con el avance previo (lemas_hasta).
        total_docs: Total de documentos.

    Returns:
        Tupla (contador de lemas, segundos de la pasada).
    """
    import spacy

    umbral = CONFIG["spacy_long_umbral_chars"]
    cada = CONFIG["checkpoint_cada"]
    cada_largos = CONFIG["checkpoint_largos_cada"]
    espera_watchdog = CONFIG["watchdog_pipe_seg"]

    textos = df["texto"].tolist()
    ids = df["id"].astype(str).tolist()

    # Reconciliacion por id contra los checkpoints existentes.
    hechas = set()
    for ruta_ckpt in sorted(glob.glob(os.path.join(OUT_DIR, "lemas_ckpt_*.parquet"))):
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
        f"pasada=lemas reconciliada hechas={len(hechas)} "
        f"lemas_hasta_previo={estado.get('lemas_hasta', 0)} "
        f"pend_cortos={len(pendientes_cortos)} "
        f"pend_largos={len(pendientes_largos)}/{n_largos_total} "
        f"umbral={umbral}"
    )

    contador = parquet_a_contador(CONTADOR_PARCIAL)

    if not pendientes_cortos and not pendientes_largos:
        escribir_log("pasada=lemas ya completa, se omite")
        return contador, 0.0

    nlp = spacy.load(CONFIG["modelo_spacy"], disable=CONFIG["spacy_disable"])
    # El corpus contiene docs de hasta ~3M caracteres y el limite por
    # defecto (1M) aborta con E088/E871. Con parser y NER deshabilitados
    # es seguro ampliarlo a 5M (igual que D1).
    nlp.max_length = 5_000_000

    t0 = time.time()
    procesados_run = 0
    bloque_filas = []
    base_hechas = len(hechas)
    ultimo_ckpt = [time.time()]

    def emitir_checkpoint(nombre, fase):
        """Persiste secuencias, contador parcial y estado de avance."""
        pd.DataFrame(bloque_filas).to_parquet(
            os.path.join(OUT_DIR, nombre), index=False
        )
        contador_a_parquet(contador, CONTADOR_PARCIAL)
        hechos = base_hechas + procesados_run
        guardar_estado(hechos, total_docs)
        elapsed = time.time() - t0
        vel = procesados_run / elapsed if elapsed > 0 and procesados_run > 0 else 0.0
        seg_doc = elapsed / procesados_run if procesados_run > 0 else 0.0
        eta = formatear_eta(seg_doc, total_docs - hechos)
        escribir_log(
            f"pasada=lemas fase={fase} hechos={hechos}/{total_docs} "
            f"({100.0 * hechos / total_docs:.1f}%) "
            f"vel={vel:.2f} docs/s ETA={eta} "
            f"tipos={len(contador)} ckpt={nombre}"
        )
        ultimo_ckpt[0] = time.time()

    def procesar_doc(doc, idx):
        """Extrae la secuencia de lemas de un doc y actualiza el contador."""
        lemas_doc = []
        for tok in doc:
            if tok.is_alpha:
                lema = tok.lemma_.lower().strip()
                if lema:
                    lemas_doc.append(lema)
                    contador[lema] += 1
        bloque_filas.append(
            {
                "id": str(ids[idx]),
                "lemas_texto": " ".join(lemas_doc),
                "n_lemas": len(lemas_doc),
            }
        )

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
                    emitir_checkpoint(f"lemas_ckpt_a_{hechos:06d}.parquet", "pipe")
                    bloque_filas.clear()
        except SystemExit:
            raise
        except Exception as exc_pipe:
            if bloque_filas:
                hechos = base_hechas + procesados_run
                emitir_checkpoint(f"lemas_ckpt_a_{hechos:06d}.parquet", "pipe")
                bloque_filas.clear()
            else:
                guardar_estado(base_hechas + procesados_run, total_docs)
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
            if (
                (procesados_run % cada == 0)
                or (k % cada_largos == 0)
                or (k == len(pendientes_largos))
            ):
                hechos = base_hechas + procesados_run
                emitir_checkpoint(f"lemas_ckpt_b_{hechos:06d}.parquet", "largos")
                bloque_filas.clear()
                escribir_log(
                    f"fase=largos k={k}/{len(pendientes_largos)} "
                    f"ultimo_id={ids[pos]} len={lon} seg_doc={seg_doc:.1f}"
                )
        escribir_log("fase=largos terminada")

    segundos = time.time() - t0
    return contador, segundos


def concatenar_checkpoints(total_docs):
    """Concatena los checkpoints de secuencias en un unico DataFrame.

    Args:
        total_docs: Total esperado de documentos.

    Returns:
        DataFrame con columnas id, lemas_texto, n_lemas ordenado por id.
    """
    patron = os.path.join(OUT_DIR, "lemas_ckpt_*.parquet")
    archivos = sorted(glob.glob(patron))
    if not archivos:
        raise FileNotFoundError("No se encontraron checkpoints de lemas")
    partes = [pd.read_parquet(a) for a in archivos]
    df = pd.concat(partes, ignore_index=True)
    df["id"] = df["id"].astype(str)
    df = df.drop_duplicates(subset="id", keep="first")
    df = df.sort_values("id").reset_index(drop=True)
    if len(df) != total_docs:
        raise ValueError(f"Lemas incompletos: {len(df)} filas para {total_docs} docs")
    return df


def verificar_consistencia(df_final, contador_memoria):
    """Verifica que el vocabulario derivado coincida con el del EDA.

    Universos: ambos usan el documento completo sobre `texto` crudo con la
    misma regla (tok.is_alpha hacia lemma_.lower().strip() no vacio) y el
    mismo modelo spaCy, por lo que deben coincidir. Tolerancia explicada:
    diferencias cercanas a cero son esperables solo por documentos
    omitidos con error, lemas vacios filtrados o cambio de version del
    modelo; cualquier diferencia mayor indica divergencia de regla o de
    universo y debe investigarse.

    Args:
        df_final: DataFrame final con id, lemas_texto, n_lemas.
        contador_memoria: Counter incremental construido durante la pasada.

    Returns:
        Diccionario con el resumen de la verificacion.
    """
    total_secuencias = int(df_final["n_lemas"].sum())
    total_contador = int(sum(contador_memoria.values()))
    n_tipos_nuevo = len(contador_memoria)

    # Chequeo de integridad interna: las secuencias decodifican al contador.
    # Muestra determinista de documentos para validar n_lemas y formato.
    muestra = df_final.sample(
        n=min(CONFIG["muestra_verificacion_n"], len(df_final)),
        random_state=CONFIG["semilla"],
    )
    decodificacion_ok = True
    for row in muestra.itertuples(index=False):
        partes = str(row.lemas_texto).split() if str(row.lemas_texto) else []
        if len(partes) != int(row.n_lemas):
            decodificacion_ok = False
            break
        if any(p != p.lower() or not p.strip() for p in partes):
            decodificacion_ok = False
            break

    resumen = {
        "n_docs": len(df_final),
        "total_lemas_secuencias": total_secuencias,
        "total_lemas_contador": total_contador,
        "integridad_secuencias_contador": bool(total_secuencias == total_contador),
        "decodificacion_muestra_ok": bool(decodificacion_ok),
        "n_tipos_nuevo": n_tipos_nuevo,
    }

    ruta_ref = CONFIG["referencia_contadores"]
    if not os.path.exists(ruta_ref):
        resumen["referencia"] = "no_encontrada"
        escribir_log("verificacion: referencia contadores_lemas.parquet no encontrada")
        with open(VERIFICACION_JSON, "w", encoding="utf-8") as fh:
            json.dump(resumen, fh, indent=2, ensure_ascii=False)
        return resumen

    df_ref = pd.read_parquet(ruta_ref)
    ref_counter = collections.Counter(
        dict(zip(df_ref["lema"].astype(str), df_ref["freq"].astype(int)))
    )
    n_tipos_ref = len(ref_counter)
    total_ref = int(sum(ref_counter.values()))
    set_nuevo = set(contador_memoria.keys())
    set_ref = set(ref_counter.keys())
    inter = set_nuevo & set_ref
    solo_nuevo = set_nuevo - set_ref
    solo_ref = set_ref - set_nuevo
    # Diferencias de frecuencia sobre la interseccion (top 10 absolutas).
    difs = []
    for lema in inter:
        d = abs(int(contador_memoria[lema]) - int(ref_counter[lema]))
        if d:
            difs.append((lema, d))
    difs.sort(key=lambda kv: kv[1], reverse=True)

    resumen.update(
        {
            "referencia": str(ruta_ref),
            "n_tipos_ref": n_tipos_ref,
            "total_lemas_ref": total_ref,
            "tipos_interseccion": len(inter),
            "tipos_solo_nuevo": len(solo_nuevo),
            "tipos_solo_ref": len(solo_ref),
            "solapamiento_sobre_ref": round(len(inter) / n_tipos_ref, 6)
            if n_tipos_ref
            else 0.0,
            "solapamiento_sobre_nuevo": round(len(inter) / n_tipos_nuevo, 6)
            if n_tipos_nuevo
            else 0.0,
            "ej_solo_nuevo": sorted(solo_nuevo)[:10],
            "ej_solo_ref": sorted(solo_ref)[:10],
            "top_difs_freq": [[lema, int(d)] for lema, d in difs[:10]],
            "n_difs_freq": len(difs),
        }
    )
    escribir_log(
        f"verificacion tipos_nuevo={n_tipos_nuevo} tipos_ref={n_tipos_ref} "
        f"inter={len(inter)} solo_nuevo={len(solo_nuevo)} "
        f"solo_ref={len(solo_ref)} difs_freq={len(difs)} "
        f"tot_seq={total_secuencias} tot_ref={total_ref}"
    )
    with open(VERIFICACION_JSON, "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    return resumen


def main():
    """Orquesta la lematizacion por documento con reanudacion y verificacion.

    Returns:
        None.
    """
    configurar_semillas(CONFIG["semilla"])
    asegurar_directorios()
    print(f"Corpus: {CORPUS_PATH}", flush=True)
    print(f"Salida: {SALIDA_FINAL}", flush=True)

    df = cargar_corpus(CORPUS_PATH)
    total_docs = len(df)
    print(f"Documentos cargados: {total_docs}", flush=True)
    print(df.head(2)[["id"]].to_string(), flush=True)
    print(df["texto"].astype(str).str.len().describe().to_string(), flush=True)

    estado = cargar_estado(total_docs)
    escribir_log(
        f"inicio lemas_hasta={estado['lemas_hasta']} total={total_docs} "
        f"modelo={CONFIG['modelo_spacy']}"
    )

    # ===== SECCION 1: pasada de lemas con dos fases y checkpoints =====
    contador, seg_lemas = pasada_lemas(df, estado, total_docs)

    # ===== SECCION 2: concatenado y salida final =====
    df_final = concatenar_checkpoints(total_docs)
    df_final.to_parquet(SALIDA_FINAL, index=False)
    guardar_estado(total_docs, total_docs)

    # ===== SECCION 3: verificacion de consistencia contra el EDA =====
    resumen = verificar_consistencia(df_final, contador)
    escribir_log(
        f"fin total_seg={seg_lemas:.1f} docs={len(df_final)} "
        f"tipos={resumen.get('n_tipos_nuevo')} "
        f"inter={resumen.get('tipos_interseccion', '?')}"
    )
    print("RESUMEN: " + json.dumps(resumen, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - el protocolo exige traceback y exit 1
        traceback.print_exc()
        sys.exit(1)

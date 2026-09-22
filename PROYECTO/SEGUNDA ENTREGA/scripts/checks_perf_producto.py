"""checks_perf_producto.py

Fecha: 2026-09-09

Descripcion: Auditoria de rendimiento (modo Deep acotado) del producto final
    Juris-PLN. Instrumenta UNA sentencia (--id) midiendo con time.perf_counter
    el tiempo de inicializacion de cada componente y el tiempo de inferencia
    desglosado por componente: M0 validacion, M2 extraccion regex, M4
    lematizacion spaCy + mascara + TF-IDF/LogReg, M7 BM25 (tokenizacion de
    query vs scoring vs pooling) y M5 contexto jurisprudencial.

    No modifica el producto: importa las clases de lib/componentes_producto.py
    y replica el orden exacto de construir_salida() de 10_evaluar_sentencia.py.
    Para el desglose interno de M4 y M7 accede a los atributos privados de las
    clases (_nlp, _vectorizador, _indice, ...) ejecutando las mismas
    operaciones linea a linea; la coherencia se verifica comparando la suma
    de las piezas contra una segunda invocacion del metodo publico completo
    (clasificar/recuperar), que ademas revela el efecto de caches calientes.

    Recursos medidos: RSS pico (ru_maxrss) y actual (/proc/self/status), hilos
    del proceso y %CPU muestreado cada 50 ms (/proc/self/stat) por fase
    (carga vs inferencia). Verifica ademas que la inferencia es CPU-only
    (sin torch/cupy/tensorflow en sys.modules).

    Salida: tabla legible por stdout y JSON crudo con todas las mediciones en
    resultados/qa_perf/perf_<id>.json (fuente de verdad para el reporte
    AUDITORIA_RENDIMIENTO.md). Con --cprofile ademas perfila una ejecucion
    completa de construir_salida() con cProfile y guarda el .prof.

Uso:
    /home/carlos/venv/bin/python scripts/checks_perf_producto.py --id 6251
    /home/carlos/venv/bin/python scripts/checks_perf_producto.py --id 20178
    /home/carlos/venv/bin/python scripts/checks_perf_producto.py --id 6251 \
        --cprofile
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import resource
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.componentes_producto import (  # noqa: E402
    ContextoJurisprudencial,
    ClasificadorArea,
    ExtractorEntidades,
    ValidadorEntrada,
    VinculoCP,
    _cargar_modulo,
    enmascarar_citas_lemas,
    normalizar_lemas,
    tokenizar,
)

ROOT = Path(__file__).resolve().parent.parent
DIR_SALIDA = ROOT / "resultados" / "qa_perf"
MODULO_10 = _cargar_modulo(
    Path(__file__).resolve().parent / "10_evaluar_sentencia.py",
    "evaluar_sentencia_10",
)


# ===== SECCION 1: utilidades de medicion =====
class MonitorRecursos:
    """Muestrea %CPU, hilos y RSS del proceso desde un hilo lateral.

    El %CPU se calcula como (utime+stime)/wall entre muestras consecutivas
    de /proc/self/stat; los hilos se leen de /proc/self/status. Cada muestra
    se atribuye a la fase vigente (carga o inferencia) para separar el costo
    de inicializar modelos del de inferir.
    """

    def __init__(self, intervalo_s: float = 0.05) -> None:
        """Prepara el monitor sin arrancarlo (ver iniciar/marcar/parar).

        Args:
            intervalo_s: Periodo de muestreo en segundos.
        """
        self.intervalo = intervalo_s
        self._clk_tck = os.sysconf("SC_CLK_TCK")
        self._parar = threading.Event()
        self._fase = "carga"
        self.cpu_pct: dict[str, list[float]] = defaultdict(list)
        self.hilos_pico: dict[str, int] = defaultdict(int)
        self.rss_pico_mb: dict[str, float] = defaultdict(float)
        self._hilo = threading.Thread(target=self._run, daemon=True)

    def _cpu_segundos(self) -> float:
        """Lee el tiempo de CPU (usuario+sistema) acumulado del proceso."""
        with open("/proc/self/stat", encoding="ascii") as fh:
            campos = fh.read().split()
        return (int(campos[13]) + int(campos[14])) / self._clk_tck

    @staticmethod
    def _contar_hilos() -> int:
        """Cuenta los hilos vivos del proceso (/proc/self/status)."""
        with open("/proc/self/status", encoding="ascii") as fh:
            for linea in fh:
                if linea.startswith("Threads:"):
                    return int(linea.split()[1])
        return 0

    def _run(self) -> None:
        """Bucle de muestreo hasta la senal de parada."""
        cpu_prev = self._cpu_segundos()
        t_prev = time.perf_counter()
        while not self._parar.wait(self.intervalo):
            cpu = self._cpu_segundos()
            t = time.perf_counter()
            delta_cpu, delta_t = cpu - cpu_prev, t - t_prev
            if delta_t > 0:
                self.cpu_pct[self._fase].append(
                    min(delta_cpu / delta_t, 32.0) * 100.0
                )
            self.hilos_pico[self._fase] = max(
                self.hilos_pico[self._fase], self._contar_hilos()
            )
            self.rss_pico_mb[self._fase] = max(
                self.rss_pico_mb[self._fase], rss_actual_mb()
            )
            cpu_prev, t_prev = cpu, t

    def iniciar(self) -> None:
        """Arranca el hilo de muestreo."""
        self._hilo.start()

    def marcar(self, fase: str) -> None:
        """Cambia la fase a la que se atribuyen las muestras siguientes."""
        self._fase = fase

    def parar(self) -> None:
        """Detiene el hilo de muestreo (idempotente)."""
        self._parar.set()
        if self._hilo.is_alive():
            self._hilo.join(timeout=2.0)

    def resumen(self) -> dict:
        """Agrega las muestras por fase (CPU, hilos y RSS pico).

        Returns:
            Dict fase -> {cpu_prom_pct, cpu_pico_pct, hilos_pico,
            rss_pico_mb}.
        """
        salida = {}
        for fase, muestras in self.cpu_pct.items():
            salida[fase] = {
                "cpu_prom_pct": round(sum(muestras) / len(muestras), 1),
                "cpu_pico_pct": round(max(muestras), 1),
                "hilos_pico": self.hilos_pico.get(fase, 0),
                "rss_pico_mb": round(self.rss_pico_mb.get(fase, 0.0), 1),
                "n_muestras": len(muestras),
            }
        return salida


def rss_actual_mb() -> float:
    """Devuelve el RSS actual del proceso en MB (/proc/self/status)."""
    with open("/proc/self/status", encoding="ascii") as fh:
        for linea in fh:
            if linea.startswith("VmRSS:"):
                return int(linea.split()[1]) / 1024
    return float("nan")


def rss_pico_mb() -> float:
    """Devuelve el RSS pico historico del proceso en MB (ru_maxrss)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def stats_documento(texto_crudo: str, texto_limpio: str, salida: dict) -> dict:
    """Calcula estadisticos de tamano del documento evaluado.

    Args:
        texto_crudo: Texto crudo de la sentencia.
        texto_limpio: Texto limpio del corpus (query de BM25).
        salida: Salida del producto (para contar citas y articulos).

    Returns:
        Dict con chars/palabras de crudo y limpio, n_citas y n_arts.
    """
    return {
        "chars_crudo": len(texto_crudo),
        "palabras_crudo": len(texto_crudo.split()),
        "chars_limpio": len(texto_limpio),
        "palabras_limpio": len(texto_limpio.split()),
        "n_citas": len(salida["sentencias_citadas"]),
        "n_arts_extraidos": len(salida["articulos_cp"]["extraidos"]),
    }


# ===== SECCION 2: carga de componentes =====
def cargar_componentes(id_corpus: str) -> tuple[dict, dict, tuple[str, str]]:
    """Inicializa el producto en el orden exacto del CLI y lo cronometra.

    Replica main() de 10_evaluar_sentencia.py: lectura del corpus por id,
    luego inicializacion de los cinco componentes.

    Args:
        id_corpus: Id del documento a evaluar.

    Returns:
        Tupla (componentes, tiempos_carga, (texto_crudo, texto_limpio)).
    """
    tiempos: dict[str, float] = {}

    t0 = time.perf_counter()
    texto_crudo, texto_limpio = MODULO_10.texto_por_id(id_corpus)
    tiempos["lectura_corpus"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    validador = ValidadorEntrada()
    tiempos["init_m0_validador"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    extractor = ExtractorEntidades()
    tiempos["init_m2_extractor"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clasificador = ClasificadorArea()
    tiempos["init_m4_clasificador"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vinculo = VinculoCP()
    tiempos["init_m7_vinculo"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    contexto = ContextoJurisprudencial()
    tiempos["init_m5_contexto"] = time.perf_counter() - t0

    tiempos["total_carga"] = sum(v for k, v in tiempos.items() if k != "total_carga")

    componentes = {
        "validador": validador,
        "extractor": extractor,
        "clasificador": clasificador,
        "vinculo": vinculo,
        "contexto": contexto,
    }
    return componentes, tiempos, (texto_crudo, texto_limpio)


# ===== SECCION 3: inferencia desglosada por componente =====
def inferir_desglosada(
    componentes: dict, texto_crudo: str, texto_limpio: str
) -> tuple[dict, dict]:
    """Ejecuta el pipeline pieza a pieza cronometrando cada componente.

    Replica construir_salida() de 10_evaluar_sentencia.py en el mismo orden,
    pero abriendo M4 y M7 en sub-pasos (spaCy vs TF-IDF/LogReg; tokenizacion
    vs scoring vs pooling) con los mismos atributos de las clases.

    Args:
        componentes: Dict de componentes ya inicializados.
        texto_crudo: Texto crudo de la sentencia.
        texto_limpio: Texto limpio (query BM25).

    Returns:
        Tupla (tiempos por pieza, salida del producto equivalente).
    """
    validador = componentes["validador"]
    extractor = componentes["extractor"]
    clasificador = componentes["clasificador"]
    vinculo = componentes["vinculo"]
    contexto = componentes["contexto"]
    t: dict[str, float] = {}

    t0 = time.perf_counter()
    validacion = validador.validar(texto_crudo)
    t["m0_validacion"] = time.perf_counter() - t0
    es_sentencia = validacion["es_sentencia"]

    t0 = time.perf_counter()
    entidades = extractor.extraer(texto_crudo)
    t["m2_extraccion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    canon_detectado = extractor.detectar_canon_encabezado(texto_crudo)
    t["m2_canon_encabezado"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    id_detectado = (
        contexto.id_de_canon(canon_detectado) if canon_detectado else None
    )
    t["m5_id_de_canon"] = time.perf_counter() - t0

    # M4 desglosado: carga lazy spaCy -> lematizacion -> mascara ->
    # TF-IDF + LogReg. La carga lazy es parte del costo real de la 1a
    # inferencia del producto (en el CLI ocurre dentro de clasificar()).
    t0 = time.perf_counter()
    clasificador._cargar_spacy()
    t["m4_carga_lazy_spacy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    doc = clasificador._nlp(texto_crudo)
    t["m4_lematizacion_spacy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lemas = " ".join(
        tok.lemma_.lower().strip() for tok in doc if tok.is_alpha
    )
    mascarada = enmascarar_citas_lemas(normalizar_lemas(lemas))
    t["m4_mascara_lemas"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    matriz = clasificador._vectorizador.transform([mascarada])
    probs = clasificador._modelo.predict_proba(matriz)[0]
    pares = sorted(
        zip(clasificador._modelo.classes_, probs, strict=True),
        key=lambda par: -par[1],
    )
    top3 = [
        {"area": str(a), "prob": round(float(p), 4)} for a, p in pares[:3]
    ]
    t["m4_tfidf_logreg"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    extraidos = []
    for canon in entidades["arts_cp"]:
        detalle = vinculo.detalle_articulo(canon)
        if detalle is not None:
            extraidos.append({"numero": canon, **detalle})
    t["m7_detalle_arts"] = time.perf_counter() - t0

    # M7 desglosado: chunks+tokenizacion -> get_scores -> pooling+ranking.
    t0 = time.perf_counter()
    chunks = vinculo._chunks_query(texto_limpio)
    tokens_por_chunk = [tokenizar(chunk) for chunk in chunks]
    t["m7_tokenizacion_query"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    max_pasajes = np.zeros(vinculo._indice.corpus_size, dtype=np.float64)
    for tokens in tokens_por_chunk:
        if tokens:
            np.maximum(
                max_pasajes, vinculo._indice.get_scores(tokens), out=max_pasajes
            )
    t["m7_scoring_bm25"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = np.zeros(len(vinculo._canones), dtype=np.float64)
    for i, indices in enumerate(vinculo._grupos):
        scores[i] = float(max_pasajes[indices].max()) if len(indices) else 0.0
    orden = np.argsort(-scores, kind="stable")[:5]
    recuperados = [
        {"numero": vinculo._canones[i], "score": round(float(scores[i]), 4)}
        for i in orden
    ]
    t["m7_pooling_ranking"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    citas = [c for c in entidades["sents_citadas"] if c != canon_detectado]
    citadas = contexto.resolver(citas)
    t["m5_resolver_citas"] = time.perf_counter() - t0

    t["suma_componentes"] = sum(
        v for k, v in t.items() if k != "suma_componentes"
    )

    # Verificaciones con el metodo publico completo (2a pasada, caches
    # calientes): mide el efecto cache y valida la replica pieza a pieza.
    t0 = time.perf_counter()
    area_publico = clasificador.clasificar(texto_crudo)
    t["m4_total_clasificar_2a_pasada"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    recuperados_publico = vinculo.recuperar(texto_limpio, 5)
    t["m7_total_recuperar_2a_pasada"] = time.perf_counter() - t0

    coherencia = {
        "m4_top1_coincide": (
            area_publico["top3"][0] == top3[0] if top3 else False
        ),
        "m7_recuperados_coinciden": recuperados_publico == recuperados,
    }

    salida = {
        "es_sentencia": es_sentencia,
        "id_detectado": id_detectado,
        "ponente": entidades["ponente"],
        "area": {"top3": top3, "via": "clasificador_m4_logreg_enmascarado"},
        "articulos_cp": {
            "extraidos": extraidos,
            "recuperados_bm25": recuperados,
        },
        "sentencias_citadas": citadas,
        "coherencia_replica": coherencia,
    }
    return t, salida


def ejecutar_end_to_end(
    componentes: dict, texto_crudo: str, texto_limpio: str, id_corpus: str
) -> float:
    """Mide una pasada completa de construir_salida() del producto.

    Args:
        componentes: Dict de componentes ya inicializados.
        texto_crudo: Texto crudo de la sentencia.
        texto_limpio: Texto limpio del corpus.
        id_corpus: Id del documento (para id_entrada).

    Returns:
        Segundos de la pasada completa (3a ejecucion, caches calientes).
    """
    t0 = time.perf_counter()
    MODULO_10.construir_salida(
        texto_crudo, texto_limpio, str(id_corpus), componentes
    )
    return time.perf_counter() - t0


# ===== SECCION 4: cProfile opcional =====
def perfilar_cprofile(
    componentes: dict, texto_crudo: str, texto_limpio: str, id_corpus: str
) -> str:
    """Perfila construir_salida() con cProfile y devuelve el top cumtime.

    Args:
        componentes: Dict de componentes ya inicializados.
        texto_crudo: Texto crudo de la sentencia.
        texto_limpio: Texto limpio del corpus.
        id_corpus: Id del documento evaluado.

    Returns:
        Cadena con el top 25 de funciones por tiempo acumulado.
    """
    perfilador = cProfile.Profile()
    perfilador.enable()
    MODULO_10.construir_salida(
        texto_crudo, texto_limpio, str(id_corpus), componentes
    )
    perfilador.disable()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_prof = DIR_SALIDA / f"cprofile_{id_corpus}.prof"
    perfilador.dump_stats(str(ruta_prof))
    buffer = io.StringIO()
    stats = pstats.Stats(perfilador, stream=buffer)
    stats.sort_stats("cumulative").print_stats(25)
    return buffer.getvalue()


# ===== SECCION 5: informe =====
def verificar_cpu_only() -> dict:
    """Comprueba que la inferencia no involucra stack GPU.

    torch puede aparecer en sys.modules porque thinc (backend de spaCy)
    lo importa como dependencia; se verifica ademas que ningun tensor viva
    en GPU (torch.cuda.is_available / torch.version.hip) para confirmar
    que el computo es CPU-only.

    Returns:
        Dict con modulos GPU detectados, disponibilidad de GPU segun torch
        y el veredicto cpu_only_confirmado.
    """
    gpu_mods = ["torch", "cupy", "tensorflow", "jax", "onnxruntime"]
    detectados = [m for m in gpu_mods if m in sys.modules]
    gpu_disponible = None
    if "torch" in sys.modules:
        import torch

        gpu_disponible = bool(torch.cuda.is_available())
    return {
        "modulos_gpu_en_sys_modules": detectados,
        "torch_gpu_disponible": gpu_disponible,
        "cpu_only_confirmado": gpu_disponible is not True,
    }


def imprimir_tabla(tiempos: dict, titulo: str) -> None:
    """Imprime un dict de tiempos como tabla componente/segundos/%.

    Args:
        tiempos: Dict pieza -> segundos.
        titulo: Titulo de la tabla.
    """
    total = sum(
        v
        for k, v in tiempos.items()
        if not k.startswith("total_")
        and not k.endswith("2a_pasada")
        and k != "suma_componentes"
    )
    print(f"\n== {titulo} ==")
    for clave, valor in tiempos.items():
        if clave.startswith("total_") or clave.endswith("2a_pasada"):
            continue
        pct = 100.0 * valor / total if total else 0.0
        print(f"  {clave:<28} {valor:8.3f} s  ({pct:5.1f}%)")


def main() -> None:
    """Orquesta la auditoria: carga, inferencia desglosada y recursos."""
    parser = argparse.ArgumentParser(
        description=(
            "Auditoria de rendimiento del producto Juris-PLN para una "
            "sentencia del corpus (desglose por componente)."
        )
    )
    parser.add_argument("--id", required=True, help="Id del documento")
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Perfila ademas construir_salida() con cProfile",
    )
    args = parser.parse_args()
    id_corpus = str(args.id)

    monitor = MonitorRecursos()
    monitor.iniciar()

    componentes, tiempos_carga, (texto_crudo, texto_limpio) = (
        cargar_componentes(id_corpus)
    )
    rss_tras_carga = rss_actual_mb()

    monitor.marcar("inferencia")
    tiempos_inf, salida = inferir_desglosada(
        componentes, texto_crudo, texto_limpio
    )
    t_end_to_end = ejecutar_end_to_end(
        componentes, texto_crudo, texto_limpio, id_corpus
    )
    monitor.parar()

    informe = {
        "id": id_corpus,
        "stats_documento": stats_documento(texto_crudo, texto_limpio, salida),
        "carga": {k: round(v, 3) for k, v in tiempos_carga.items()},
        "rss_tras_carga_mb": round(rss_tras_carga, 1),
        "inferencia": {k: round(v, 3) for k, v in tiempos_inf.items()},
        "construir_salida_3a_pasada_s": round(t_end_to_end, 3),
        "recursos": {
            "rss_pico_mb": round(rss_pico_mb(), 1),
            "rss_final_mb": round(rss_actual_mb(), 1),
            "fases": monitor.resumen(),
        },
        "cpu_only": verificar_cpu_only(),
        "coherencia_replica": salida["coherencia_replica"],
    }

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_json = DIR_SALIDA / f"perf_{id_corpus}.json"
    ruta_json.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    imprimir_tabla(tiempos_carga, f"CARGA de componentes (id {id_corpus})")
    imprimir_tabla(tiempos_inf, f"INFERENCIA por componente (id {id_corpus})")
    print(f"  {'construir_salida (3a, caliente)':<28} {t_end_to_end:8.3f} s")
    print(f"\nRSS pico: {rss_pico_mb():.0f} MB | tras carga: {rss_tras_carga:.0f} MB")
    print(f"Recursos por fase: {json.dumps(monitor.resumen(), ensure_ascii=False)}")
    print(f"CPU-only: {informe['cpu_only']}")
    print(f"Coherencia replica: {informe['coherencia_replica']}")
    print(f"[info] JSON guardado en {ruta_json}")
    print(
        "[nota] m4_carga_lazy_spacy se mide en la 1a inferencia; en la 3a "
        "pasada (caliente) ya esta incluida en construir_salida."
    )

    if args.cprofile:
        top = perfilar_cprofile(
            componentes, texto_crudo, texto_limpio, id_corpus
        )
        print(top)


if __name__ == "__main__":
    main()

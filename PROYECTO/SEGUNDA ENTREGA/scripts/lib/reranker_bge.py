"""reranker_bge.py
Fecha: 2026-09-10
Descripcion: Cross-encoder BAAI/bge-reranker-v2-m3 para reranking de dos
    etapas (ciclo v2 - Fase T, plan/PLAN_v2_ciclo_mejora.md). Modulo
    compartido entre el benchmark (scripts/12_evaluar_vinculo_cp_v2.py,
    estrategia bm25_rerank) y el producto (VinculoCP de
    scripts/lib/componentes_producto.py, modo opcional usar_reranker):
    encapsula la eleccion de dispositivo con guard de VRAM, la carga lazy
    del modelo (XLMRoberta de 1 clase de relevancia; el logit raw sirve
    para ordenar, sigmoid innecesaria) y la puntuacion de pares
    (query, pasaje) por lotes con padding dinamico y agrupacion por
    longitud para minimizar el relleno. El truncamiento es del PAR a 512
    tokens (longest_first: se recorta primero la secuencia mas larga, no
    el pasaje solo).
Estado del arte aplicado: reranking con cross-encoder multilingue
    (bge-reranker-v2-m3, 568M parametros, apache-2.0; arquitectura
    ganadora en statute retrieval segun DoSSIER/COLIEE y LegalMALR 2026),
    fp16 en GPU, batch 16 en GPU (mitigado a mitades ante OOM) y batch 4
    en CPU.
"""

import numpy as np
import torch

CONFIG_RERANKER = {
    "modelo": "BAAI/bge-reranker-v2-m3",
    "max_length": 512,
    "batch_gpu": 16,
    "batch_cpu": 4,
    "vram_minima_gb": 6.0,
}


def elegir_dispositivo() -> str:
    """Elige GPU solo si hay VRAM libre suficiente para el reranker.

    ciclo v2 - Fase T: la maquina comparte la GPU con procesos externos;
    si CUDA no esta disponible o la VRAM libre es menor que
    CONFIG_RERANKER['vram_minima_gb'] se degrada a CPU con batch pequeno
    y se documenta con WARNING en el log (no bloquea la ejecucion).

    Returns:
        "cuda" si hay >= 6 GiB libres y torch la ve disponible; si no "cpu".
    """
    if not torch.cuda.is_available():
        print(
            "[aviso] reranker: CUDA no disponible; se puntua en CPU "
            f"(batch {CONFIG_RERANKER['batch_cpu']}, documentado)"
        )
        return "cpu"
    libre, _ = torch.cuda.mem_get_info()
    minimo = int(CONFIG_RERANKER["vram_minima_gb"] * 1024**3)
    if libre < minimo:
        print(
            "[aviso] reranker: VRAM libre insuficiente "
            f"({libre / 1024**3:.1f} GiB < {CONFIG_RERANKER['vram_minima_gb']} "
            f"GiB); se puntua en CPU (batch {CONFIG_RERANKER['batch_cpu']}, "
            "documentado)"
        )
        return "cpu"
    return "cuda"


class RerankerBGE:
    """Cross-encoder bge-reranker-v2-m3 para puntuar pares (query, pasaje).

    ciclo v2 - Fase T: carga el modelo una sola vez por proceso con fp16
    en GPU (float32 en CPU) y expone puntuar() que devuelve el logit raw
    de relevancia por par. El batch se agrupa por longitud del par para
    minimizar el padding y ante OOM se rebana a mitades (documentado).
    """

    def __init__(self, dispositivo: str | None = None) -> None:
        """Carga tokenizer y modelo del reranker en el dispositivo elegido.

        Args:
            dispositivo: "cuda"/"cpu" forzado; si None se elige con el
                guard de VRAM de elegir_dispositivo().
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dispositivo = dispositivo or elegir_dispositivo()
        self._batch = (
            CONFIG_RERANKER["batch_gpu"]
            if self.dispositivo == "cuda"
            else CONFIG_RERANKER["batch_cpu"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(CONFIG_RERANKER["modelo"])
        dtype = torch.float16 if self.dispositivo == "cuda" else torch.float32
        self._modelo = AutoModelForSequenceClassification.from_pretrained(
            CONFIG_RERANKER["modelo"], dtype=dtype
        ).to(self.dispositivo)
        self._modelo.eval()
        print(
            f"[info] reranker {CONFIG_RERANKER['modelo']} cargado en "
            f"{self.dispositivo} (fp16: {self.dispositivo == 'cuda'}, "
            f"batch {self._batch}, max_length {CONFIG_RERANKER['max_length']})"
        )

    def _puntuar_lote(self, queries: list[str], pasajes: list[str]) -> np.ndarray:
        """Puntua un lote de pares; ante OOM lo rebana a mitades.

        Args:
            queries: Textos de query del lote.
            pasajes: Textos de pasaje alineados con queries.

        Returns:
            Array (len(lote),) con el logit raw de relevancia por par.
        """
        try:
            entradas = self._tokenizer(
                queries,
                pasajes,
                truncation=True,
                max_length=CONFIG_RERANKER["max_length"],
                padding=True,
                return_tensors="pt",
            ).to(self.dispositivo)
            with torch.inference_mode():
                logits = self._modelo(**entradas).logits
            return logits.squeeze(-1).float().cpu().numpy()
        except RuntimeError as exc:
            if self.dispositivo != "cuda" or len(queries) <= 1:
                raise
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            self._batch = max(1, len(queries) // 2)
            print(
                f"[aviso] OOM en reranker: lote rebanado a {len(queries) // 2} "
                "pares (documentado)"
            )
            mitad = len(queries) // 2
            return np.concatenate(
                [
                    self._puntuar_lote(queries[:mitad], pasajes[:mitad]),
                    self._puntuar_lote(queries[mitad:], pasajes[mitad:]),
                ]
            )

    def puntuar(self, queries: list[str], pasajes: list[str]) -> np.ndarray:
        """Puntua pares (query, pasaje) agrupandolos por longitud.

        Ordena los pares por longitud total (chars como proxy de tokens)
        para que cada lote sea homogeneo y el padding dinamico agregue el
        minimo relleno posible; despues restaura el orden original.

        Args:
            queries: Textos de query (uno por par).
            pasajes: Textos de pasaje alineados con queries.

        Returns:
            Array (n_pares,) de logits raw alineado con la entrada.
        """
        if len(queries) != len(pasajes):
            raise ValueError(
                f"queries y pasajes deben tener el mismo largo "
                f"({len(queries)} != {len(pasajes)})"
            )
        if not queries:
            return np.zeros(0, dtype=np.float32)
        orden = sorted(
            range(len(queries)), key=lambda i: len(queries[i]) + len(pasajes[i])
        )
        scores = np.zeros(len(queries), dtype=np.float32)
        inicio = 0
        while inicio < len(orden):
            indices = orden[inicio : inicio + self._batch]
            scores[indices] = self._puntuar_lote(
                [queries[i] for i in indices], [pasajes[i] for i in indices]
            )
            inicio += self._batch
        return scores

    def max_tokens_par(self, queries: list[str], pasajes: list[str]) -> int:
        """Mide la longitud maxima en tokens de los pares SIN truncar.

        ciclo v2 - Fase T iter2: verificacion de que los pares (chunk,
        pasaje) de la estrategia bm25_rerank_chunks caben en
        CONFIG_RERANKER['max_length'] = 512 tokens. Se tokeniza con
        truncation=False para observar la longitud real; si el maximo
        supera 512 el par se recortaria en _puntuar_lote y el JSON del
        benchmark lo documenta (rerank_chunks_stats).

        Args:
            queries: Textos de query (uno por par).
            pasajes: Textos de pasaje alineados con queries.

        Returns:
            Maxima longitud en tokens (incluidos special tokens) entre
            todos los pares; 0 si la entrada es vacia.
        """
        if len(queries) != len(pasajes):
            raise ValueError(
                f"queries y pasajes deben tener el mismo largo "
                f"({len(queries)} != {len(pasajes)})"
            )
        if not queries:
            return 0
        codificados = self._tokenizer(
            queries, pasajes, truncation=False, padding=False
        )["input_ids"]
        return max(len(ids) for ids in codificados)

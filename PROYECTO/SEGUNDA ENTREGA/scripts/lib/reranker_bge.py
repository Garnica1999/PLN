"""reranker_bge.py
Fecha: 2026-09-10 (parametrizado 2026-09-11)
Descripcion: Cross-encoder de relevancia para reranking de dos etapas.
    Modulo compartido entre el benchmark (scripts/12_evaluar_vinculo_cp_v2.py,
    estrategias bm25_rerank*/hibrido_rerank*) y el producto (VinculoCP de
    scripts/lib/componentes_producto.py, modo opcional usar_reranker):
    encapsula la eleccion de dispositivo con guard de VRAM, la carga lazy
    generica (AutoModelForSequenceClassification de 1 clase de relevancia;
    el logit raw sirve para ordenar, sigmoid innecesaria) y la puntuacion
    de pares (query, pasaje) por lotes con padding dinamico y agrupacion
    por longitud para minimizar el relleno. El truncamiento es del PAR a
    512 tokens (longest_first: se recorta primero la secuencia mas larga,
    no el pasaje solo).
    ciclo v3 - Sprint A3 (2026-09-11): RerankerBGE acepta modelo_id y
    carga CUALQUIER cross-encoder de relevancia con la misma ruta generica
    (XLMRoberta/bge y ModernBERT/MrBERT verificados). El default sigue
    siendo BAAI/bge-reranker-v2-m3, por lo que el comportamiento actual y
    el del producto quedan IDENTICOS; el benchmark A3 pasa
    SINAI/ALIA-MrBERT-es-legal-administrative-reranker (150M) para
    comparar rerankers legales en espanol.
Estado del arte aplicado: reranking con cross-encoder multilingue
    (bge-reranker-v2-m3, 568M parametros, apache-2.0; arquitectura
    ganadora en statute retrieval segun DoSSIER/COLIEE y LegalMALR 2026)
    y reranker de dominio legal en espanol (ALIA MrBERT cross-encoder,
    Apache-2.0, Sprint A3), fp16 en GPU, batch 16 en GPU (mitigado a
    mitades ante OOM) y batch 4 en CPU.
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

# ciclo v3 - Sprint A3: catalogo de rerankers de la comparativa; el
# modelo_id de RerankerBGE acepta estos alias o un HF id arbitrario.
RERANKERS_DISPONIBLES = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "aliarerank": "SINAI/ALIA-MrBERT-es-legal-administrative-reranker",
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
    """Cross-encoder de relevancia para puntuar pares (query, pasaje).

    ciclo v2 - Fase T: carga el modelo una sola vez por proceso con fp16
    en GPU (float32 en CPU) y expone puntuar() que devuelve el logit raw
    de relevancia por par. El batch se agrupa por longitud del par para
    minimizar el padding y ante OOM se rebana a mitades (documentado).
    ciclo v3 - Sprint A3: modelo_id permite cargar cualquier cross-encoder
    de relevancia (bge-reranker-v2-m3 por defecto; ModernBERT/MrBERT
    verificado con el reranker legal de SINAI) manteniendo identica toda
    la logica de puntuacion y de memoria.
    """

    def __init__(
        self, dispositivo: str | None = None, modelo_id: str | None = None
    ) -> None:
        """Carga tokenizer y modelo del reranker en el dispositivo elegido.

        Args:
            dispositivo: "cuda"/"cpu" forzado; si None se elige con el
                guard de VRAM de elegir_dispositivo().
            modelo_id: HF id o alias de RERANKERS_DISPONIBLES del
                cross-encoder; None conserva el default
                BAAI/bge-reranker-v2-m3 (comportamiento actual intacto).

        Raises:
            ValueError: Si el modelo cargado no es de 1 o 2 clases de
                relevancia (no hay score monotonico interpretable).
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dispositivo = dispositivo or elegir_dispositivo()
        self.modelo_id = modelo_id or CONFIG_RERANKER["modelo"]
        self._batch = (
            CONFIG_RERANKER["batch_gpu"]
            if self.dispositivo == "cuda"
            else CONFIG_RERANKER["batch_cpu"]
        )
        self._max_length = int(CONFIG_RERANKER["max_length"])
        self._tokenizer = AutoTokenizer.from_pretrained(self.modelo_id)
        dtype = torch.float16 if self.dispositivo == "cuda" else torch.float32
        self._modelo = AutoModelForSequenceClassification.from_pretrained(
            self.modelo_id, dtype=dtype
        ).to(self.dispositivo)
        num_labels = int(getattr(self._modelo.config, "num_labels", 1))
        if num_labels not in (1, 2):
            raise ValueError(
                f"reranker {self.modelo_id} con num_labels={num_labels}: "
                "se esperaba 1 (logit de relevancia) o 2 (margen positivo-"
                "negativo)"
            )
        self._modelo.eval()
        print(
            f"[info] reranker {self.modelo_id} cargado en "
            f"{self.dispositivo} (fp16: {self.dispositivo == 'cuda'}, "
            f"batch {self._batch}, max_length {self._max_length}, "
            f"num_labels {num_labels})"
        )

    def _logits_a_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Reduce los logits del cross-encoder a un score por par.

        Con num_labels=1 devuelve el unico logit (bge y SINAI); con
        num_labels=2 devuelve el margen logit_positivo - logit_negativo
        (ranking estandar de un clasificador binario). La orientacion es
        siempre "mayor = mas relevante".

        Args:
            logits: Tensor (lote, num_labels) del modelo.

        Returns:
            Tensor (lote,) de scores de relevancia.
        """
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits[..., -1] - logits[..., 0]

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
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.dispositivo)
            with torch.inference_mode():
                logits = self._modelo(**entradas).logits
            return self._logits_a_scores(logits).float().cpu().numpy()
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

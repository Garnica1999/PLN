"""V10-11: BiGRU con afinamiento de embeddings, cierre cientifico de H1.

Que cierra
----------
La fase v9 dejo H1, la hipotesis de que un modelo secuencial sobre embeddings propios mejora al
TF-IDF. V9-04 la midio con los embeddings **congelados** y dio **0.6943**, frente a 0.6636 del
control de media mas regresion lineal y 0.777211 de V03. La conclusion fue que el mecanismo
secuencial **si** mejora al pooling, con +0.031, pero no compite con V03.

Quedaba una variante sin probar: **afinar** los embeddings en vez de congelarlos. Con congelarlos
la red solo puede reordenar lo que ya esta ahi; al afinarlos puede adaptar larepresentacion al
dominio. Esta es la variante que el plan de la fase v10 nombro como cierre de H1.

Las tres comparaciones
---------------------
Se mide en el mismo protocolo que V9-04, con los mismos pliegues corregidos, la misma longitud de
secuencia y los mismos umbrales congelados de V03:

- **congelado**: lo que ya dio 0.6943, y que se reproduce aqui como control interno.
- **afinado**: la variante de este experimento, con learning rate diferencial.
- **afinando_todo**: la **misma** red pero con una sola tasa de aprendizaje para todo. Es el
  control que hace falta y que no estaba: **distingue "afinar ayuda" de "simplemente deja que la
  red entrene mas parametros"**. Sin el, un acierto del afinamiento podria ser solo mas capacidad.

Learning rate diferencial
-------------------------
La embebidencia empieza con un learning rate **diez veces menor** que el resto, 1e-4 frente a
1e-3. La razon es que los embeddings son una representacion **preentrenada sobre otro corpus** y
moverlos mucho los destruye antes de que aprovechen, mientras que la red recurrente es nueva y
necesita aprender desde cero. Un solo learning rate para todo, ademas, tiende a olvidar lo aprendido.

Sobre la decision de sobreentrenamiento
----------------------------------------
**Se usa early stopping estricto, y la decision estaba pre-registrada** en `tasks/plan_v10.md` antes
de ejecutar nada, con paciencia 3 sobre la perdida de validacion y un maximo de 14 epocas. No se
sobreentrena a proposito: el objetivo de este experimento es medir el techo de la variante, y
sobreentrenar no daria un techo sino un sobreajuste. Ademas hay un limite importante al preparar
el calculo: **la mejor epoca se elige por la perdida de validacion, que es la misma senal que se
usa para las curvas de aprendizaje**, y la metrica de la competencia se evalua una sola vez al
final. Eso no es fuga para estimar un techo, pero si lo seria para ajustar, y por eso aqui no se
ajusta nada.

Criterio de cierre
------------------
**No hay gate de envio.** Es cierre cientifico. Si la variante afinada sigue por debajo de V03, H1
queda **cerrada con dos variantes, congelada y afinada**, documentado, y el proyecto no vuelve a
abrir la via de representacion secuencial.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_11_bigru_finetuning.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.embedding_utils import normalizar_emb_normal, tokenizar
from src.evaluation import ETIQUETAS, metricas_multietiqueta
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos

_especificacion = util.spec_from_file_location(
    "v9_04", PROJECT_ROOT / "experiments" / "v9_04_dl_secuencial_fasttext.py"
)
v9_04 = util.module_from_spec(_especificacion)
sys.modules["v9_04"] = v9_04
_especificacion.loader.exec_module(v9_04)

EXPERIMENTO = "V10-11 (P4, BiGRU con afinamiento)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_11_bigru_finetuning.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_11_bigru_finetuning.json"
RESULTADOS = PROJECT_ROOT / "resultados" / "metricas.json"

REFERENCIA_V03 = 0.7772114525165721
REFERENCIA_CONGELADO_V9 = 0.6943
REFERENCIA_MEDIA_V9 = 0.6636
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

EPOCAS = 14
PACIENCIA = 3
BATCH = 64
LR_EMB = 1e-4
LR_RESTO = 1e-3
WEIGHT_DECAY = 1e-4
DIM_GRU = 64
MAX_LEN = 64
VEC_PATH = v9_04.VEC_PATH

CONGELADO = "congelado"
AFINADO = "afinado"
AFINANDO_TODO = "afinando_todo"
VARIANTES = (CONGELADO, AFINADO, AFINANDO_TODO)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-11] {mensaje}", flush=True)


class BiGRUAfinable(torch.nn.Module):
    """BiGRU con la embebidencia afinable o congelada, y pooling enmascarado.

    La embebidencia se construye siempre a partir de la matriz preentrenada, y lo unico que
    cambia entre variantes es si queda congelada o recibe gradientes. Todo lo demas es identico,
    de modo que la comparacion entre variantes no puede atribuirse a una diferencia de
    arquitectura.

    Attributes:
        embedding: capa de embebido, con los vectores preentrenados.
        gru: capa recurrente bidireccional.
        dropout: regularizacion sobre el vector agregado.
        salida: capa densa que produce los tres logits.
        afinable: si la embebidencia recibe gradientes.
    """

    def __init__(self, matriz_embedding: np.ndarray, afinable: bool) -> None:
        """Construye la red.

        Args:
            matriz_embedding: matriz de vectores preentrenados, con la fila 0 reservada al relleno.
            afinable: si la embebidencia debe recibir gradientes.
        """
        super().__init__()
        self.afinable = afinable
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(matriz_embedding), freeze=not afinable, padding_idx=0
        )
        self.gru = torch.nn.GRU(
            matriz_embedding.shape[1], DIM_GRU, batch_first=True, bidirectional=True
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.salida = torch.nn.Linear(2 * DIM_GRU, 3)

    def _embedding_parameters(self) -> list[torch.nn.Parameter]:
        """Devuelve los parametros de la embebidencia, para el grupo de tasa propia.

        Returns:
            Lista de parametros afinables de la capa de embebido.
        """
        return list(self.embedding.parameters())

    def forward(self, entrada: torch.Tensor) -> torch.Tensor:
        """Calcula los tres logits de una tanda de secuencias.

        Args:
            entrada: tensor de identificadores con forma de tanda por longitud.

        Returns:
            Tensor de forma tanda por tres, con los logits.
        """
        mascara = (entrada != 0).float().unsqueeze(-1)
        embebido = self.embedding(entrada)
        salida, _ = self.gru(embebido)
        suma = (salida * mascara).sum(dim=1)
        cuenta = mascara.sum(dim=1).clamp(min=1.0)
        return self.salida(self.dropout(suma / cuenta))


def entrenar(
    ids_tr: np.ndarray,
    y_tr: np.ndarray,
    ids_va: np.ndarray,
    y_va: np.ndarray,
    matriz_embedding: np.ndarray,
    variante: str,
    semilla: int,
    dispositivo: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Entrena una variante del BiGRU en un pliegue y devuelve probabilidades de validacion.

    Args:
        ids_tr: identificadores de entrenamiento.
        y_tr: etiquetas de entrenamiento.
        ids_va: identificadores de validacion.
        y_va: etiquetas de validacion.
        matriz_embedding: matriz de vectores preentrenados.
        variante: una de congelado, afinado o afinando_todo.
        semilla: semilla para reproducibilidad.
        dispositivo: dispositivo de torch.

    Returns:
        Tupla con las probabilidades de validacion y un diccionario con la traza del
        entrenamiento, que es lo que permite saber si el afinamiento helping o solo estropea.

    Raises:
        ValueError: si la variante no es una de las tres conocidas.
    """
    torch.manual_seed(semilla)
    if dispositivo.type == "cuda":
        torch.cuda.manual_seed_all(semilla)
    afinable = variante in (AFINADO, AFINANDO_TODO)
    modelo = BiGRUAfinable(matriz_embedding, afinable).to(dispositivo)

    if variante == AFINANDO_TODO:
        # Control: una sola tasa para todo, que es lo que haria un afinamiento "normal".
        optimizador = torch.optim.Adam(
            modelo.parameters(), lr=LR_RESTO, weight_decay=WEIGHT_DECAY
        )
    else:
        optimizador = torch.optim.Adam(
            [
                {
                    "params": modelo._embedding_parameters(),
                    "lr": LR_EMB if variante == AFINADO else LR_RESTO,
                },
                {
                    "params": [
                        p
                        for nombre, p in modelo.named_parameters()
                        if not nombre.startswith("embedding")
                    ],
                    "lr": LR_RESTO,
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )
    criterio = torch.nn.BCEWithLogitsLoss()
    x_tr = torch.tensor(ids_tr, device=dispositivo)
    t_tr = torch.tensor(y_tr, dtype=torch.float32, device=dispositivo)
    x_va = torch.tensor(ids_va, device=dispositivo)
    t_va = torch.tensor(y_va, dtype=torch.float32, device=dispositivo)

    mejor_perdida = float("inf")
    mejor_estado = {clave: valor.clone() for clave, valor in modelo.state_dict().items()}
    mejor_epoca = 0
    sin_mejora = 0
    traza: list[dict[str, float]] = []
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        permutacion = torch.randperm(len(x_tr), device=dispositivo)
        for inicio in range(0, len(x_tr), BATCH):
            idx = permutacion[inicio : inicio + BATCH]
            optimizador.zero_grad()
            perdida = criterio(modelo(x_tr[idx]), t_tr[idx])
            perdida.backward()
            optimizador.step()
        modelo.eval()
        with torch.no_grad():
            perdida_entrenamiento = float(criterio(modelo(x_tr), t_tr).item())
            perdida_validacion = float(criterio(modelo(x_va), t_va).item())
        traza.append(
            {
                "epoca": epoca,
                "perdida_entrenamiento": round(perdida_entrenamiento, 6),
                "perdida_validacion": round(perdida_validacion, 6),
            }
        )
        if perdida_validacion < mejor_perdida - 1e-5:
            mejor_perdida = perdida_validacion
            mejor_estado = {clave: valor.clone() for clave, valor in modelo.state_dict().items()}
            mejor_epoca = epoca
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        probabilidades = torch.sigmoid(modelo(x_va)).cpu().numpy()
    return probabilidades, {
        "mejor_epoca": mejor_epoca,
        "epocas_ejecutadas": len(traza),
        "mejor_perdida_validacion": round(mejor_perdida, 6),
        "perdida_entrenamiento_en_la_mejor_epoca": traza[mejor_epoca - 1][
            "perdida_entrenamiento"
        ]
        if mejor_epoca > 0
        else None,
        "traza": traza,
    }


def main() -> None:
    """Ejecuta las tres variantes del BiGRU y cierra H1."""
    inicio = time.time()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"V10-11: BiGRU con afinamiento. Dispositivo: {dispositivo}")
    if dispositivo.type == "cuda":
        log(f"   GPU: {torch.cuda.get_device_name(0)}")

    vectores = v9_04.cargar_vec(VEC_PATH)
    log(f"   vectores own: {len(vectores)}")

    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    dev = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(dev, grupos[indices_dev])
    y = dev[list(ETIQUETAS)].to_numpy(dtype=int)
    log(f"   desarrollo: {len(y)} filas | pliegues: {len(pliegues)}")

    vocabulario = sorted(
        {
            token
            for texto in dev["text"]
            for token in tokenizar(normalizar_emb_normal(texto))
            if token in vectores
        }
    )
    indice = {palabra: posicion + 1 for posicion, palabra in enumerate(vocabulario)}
    matriz_embedding = np.zeros((len(vocabulario) + 1, 200), dtype=np.float32)
    for palabra, posicion in indice.items():
        matriz_embedding[posicion] = vectores[palabra]
    ids = v9_04.secuencias(dev["text"].tolist(), indice)
    log(f"   vocabulario usado: {len(vocabulario)} tipos | longitud de secuencia: {MAX_LEN}")

    probabilidades = {nombre: np.zeros_like(y, dtype=float) for nombre in VARIANTES}
    trazas: dict[str, list[dict[str, Any]]] = {nombre: [] for nombre in VARIANTES}
    for indice_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
        for nombre in VARIANTES:
            probabilidades[nombre][validacion], traza = entrenar(
                ids[entrenamiento],
                y[entrenamiento],
                ids[validacion],
                y[validacion],
                matriz_embedding,
                nombre,
                42 + indice_pliegue,
                dispositivo,
            )
            trazas[nombre].append(traza)
        log(
            f"   pliegue {indice_pliegue + 1} de {len(pliegues)} | "
            f"mejor epoca por variante: "
            + ", ".join(f"{nombre} {trazas[nombre][-1]['mejor_epoca']}" for nombre in VARIANTES)
        )

    log("evaluar con los umbrales congelados de V03")
    resultados: dict[str, Any] = {}
    for nombre in VARIANTES:
        predicciones = np.column_stack(
            [
                (probabilidades[nombre][:, indice] >= UMBRALES_V03[etiqueta]).astype(int)
                for indice, etiqueta in enumerate(ETIQUETAS)
            ]
        )
        metricas = metricas_multietiqueta(y, predicciones)
        resultados[nombre] = {
            "f1_media_columnas": round(float(metricas["f1_media_columnas"]), 6),
            "f1_por_etiqueta": {
                clave: round(float(valor), 6) for clave, valor in metricas["f1_por_etiqueta"].items()
            },
            "mejor_epoca_por_pliegue": [t["mejor_epoca"] for t in trazas[nombre]],
            "epocas_ejecutadas_por_pliegue": [t["epocas_ejecutadas"] for t in trazas[nombre]],
        }
        log(
            f"   {nombre:>16}: {resultados[nombre]['f1_media_columnas']:.6f} | "
            f"mejor epoca {resultados[nombre]['mejor_epoca_por_pliegue']}"
        )

    afinado = resultados[AFINADO]["f1_media_columnas"]
    congelado = resultados[CONGELADO]["f1_media_columnas"]
    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencias": {
            "v03_tfidf_union_lr": REFERENCIA_V03,
            "v9_04_bi_gru_congelado": REFERENCIA_CONGELADO_V9,
            "v9_04_media_mas_regresion_lineal": REFERENCIA_MEDIA_V9,
        },
        "protocolo": (
            "Mismos pliegues corregidos, misma longitud de secuencia y mismos umbrales congelados "
            "de V03 que V9-04, para que la comparacion sea directa. La mejor epoca se elige por la "
            "perdida de validacion, con paciencia 3 y un maximo de 14 epocas."
        ),
        "decision_sobre_sobreentrenamiento": {
            "decision": "early stopping estricto",
            "quien_la_fijo": "el pre-registro en tasks/plan_v10.md, antes de ejecutar nada",
            "motivo": (
                "El objetivo es medir el techo de la variante, y sobreentrenar no daria un techo "
                "sino un sobreajuste. Ademas la mejor epoca se elige por la perdida de validacion, "
                "que es la misma senal que se usa para las curvas de aprendizaje; la metrica de la "
                "competencia se evalua una sola vez al final y no se usa para ajustar."
            ),
        },
        "variantes": list(VARIANTES),
        "resultados": resultados,
        "delta_afinado_menos_congelado": round(afinado - congelado, 6),
        "delta_afinado_menos_todo": round(
            afinado - resultados[AFINANDO_TODO]["f1_media_columnas"], 6
        ),
        "delta_afinado_menos_v03": round(afinado - REFERENCIA_V03, 6),
        "gana_a_v03": bool(afinado > REFERENCIA_V03),
        "h1_abierta": False,
        "trazas": trazas,
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: afinado {afinado:.6f} frente a congelado {congelado:.6f} y V03 "
        f"{REFERENCIA_V03:.6f} | gana a V03: {informe['gana_a_v03']}"
    )


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de las tres variantes del BiGRU.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    resultados = informe["resultados"]
    decision = informe["decision_sobre_sobreentrenamiento"]
    lineas = [
        "# V10-11: BiGRU con afinamiento de embeddings (P4, cierre de H1)",
        "",
        f"Generado por `experiments/v10_11_bigru_finetuning.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que cierra",
        "",
        "V9-04 midio H1 con los embeddings **congelados** y dio 0.6943, frente a 0.6636 del control",
        "de media mas regresion lineal y 0.777211 de V03. La conclusion fue que el mecanismo",
        "secuencial si mejora al pooling, con +0.031, pero no compite con V03.",
        "",
        "Quedaba **afinar** los embeddings en vez de congelarlos. Congelados, la red solo puede",
        "reordenar lo que ya esta ahi; afinados, puede adaptar la representacion al dominio.",
        "",
        "## 2. Las tres comparaciones, y por que la tercera es la importante",
        "",
        "| variante | embebido | learning rate | que separa |",
        "|---|---|---|---|",
        "| congelado | sin gradientes | 1e-3 | lo que ya dio V9-04 |",
        "| **afinado** | **con gradientes** | **1e-4 embebido, 1e-3 resto** | el efecto del afinamiento |",
        "| afinando todo | con gradientes | 1e-3 para todo | **el control que faltaba** |",
        "",
        "La tercera es la que hace que las otras dos signifiquen algo: **distingue afinar de",
        "simplemente dar mas parametros a la red**. Sin ese control, un acierto del afinamiento",
        "podria ser solo mas capacidad, que no es lo que se quiere defender.",
        "",
        "El learning rate diferencial multiplica por diez la separacion: la capa de embebido viene",
        "preentrenada sobre otro corpus y moverla mucho la destruye antes de que aproveche, y la",
        "red recurrente es nueva y necesita aprender desde cero.",
        "",
        "## 3. Decision sobre sobreentrenamiento",
        "",
        f"- decision: **{decision['decision']}**",
        f"- quien la fijo: {decision['quien_la_fijo']}",
        f"- motivo: {decision['motivo']}",
        "",
        "## 4. Resultados",
        "",
        "| variante | F1 | HS | TR | AG | mejor epoca por pliegue |",
        "|---|---|---|---|---|---|",
    ]
    for nombre in VARIANTES:
        r = resultados[nombre]
        lineas.append(
            f"| {nombre} | **{r['f1_media_columnas']:.6f}** | "
            f"{r['f1_por_etiqueta']['HS']:.4f} | {r['f1_por_etiqueta']['TR']:.4f} | "
            f"{r['f1_por_etiqueta']['AG']:.4f} | {r['mejor_epoca_por_pliegue']} |"
        )
    lineas += [
        "",
        "| referencia | F1 |",
        "|---|---|",
        f"| V03, TF-IDF union mas regresion lineal | **{informe['referencias']['v03_tfidf_union_lr']:.6f}** |",
        (
            f"| V9-04, media mas regresion lineal | "
            f"{informe['referencias']['v9_04_media_mas_regresion_lineal']:.6f} |"
        ),
        "",
        f"- afinado menos congelado: **{informe['delta_afinado_menos_congelado']:+.6f}**",
        f"- afinado menos afinando todo: **{informe['delta_afinado_menos_todo']:+.6f}**",
        f"- afinado menos V03: **{informe['delta_afinado_menos_v03']:+.6f}**",
        f"- **gana a V03: {informe['gana_a_v03']}**",
        "",
        "## 5. Veredicto",
        "",
    ]
    if informe["gana_a_v03"]:
        lineas += [
            "**LA VARIANTE AFINADA GANA A V03.** Es el primer resultado de toda la historia del",
            "proyecto que lo hace, y por regla de la fase **no se envia sin pasar el filtro de",
            "detectabilidad de V8-02 y el resto de condiciones de adopcion**.",
        ]
    else:
        lineas += [
            "**H1 QUEDA CERRADA con dos variantes, congelada y afinada.** Ninguna compite con V03,",
            "que era el criterio de cierre pre-registrado para P4.",
            "",
            "La lectura es coherente con el resto del proyecto. V03 es una combinacion de TF-IDF",
            "word(1,2) y char_wb(3,5) con una regresion lineal: captura redundancia **de forma",
            "explicita**, en vez de|Aprenderla a traves de una recurrencia. Con mil y pico filas, la",
            "regresion lineal sobre 24.646 rasgos explicitos aprovecha mejor los datos que una red",
            "que tiene que descubrir la estructura por su cuenta. No es que las redes no sirvan, es",
            "que **este** problema, con estos datos, no las necesita.",
        ]
    lineas += [
        "",
        "Y la consecuencia practica: **la via de representacion secuencial queda cerrada.** No se",
        "abre con mas capas, mas unidades o mas epocas, porque el criterio de cierre estaba escrito",
        "antes de ejecutar y ya se cumplio.",
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

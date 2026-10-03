"""El veredicto CLOSED de la fase Twitter queda INVALIDADO: las representaciones no entrenaron.

Lo que se observo
-----------------
Carlos miro las curvas de loss y pregunto si el modelo no estaba aprendiendo. Tenia razon.

    twitter   loss 1,706 -> 1,811 -> 1,842 -> 1,866 -> 1,884     (Adam)
    general   loss 1,888 -> 2,068 -> 2,126 -> 2,164 -> 2,190     (Adam)
    twitter   loss 1,386295 constante en las cinco epocas         (SGD)

Y 1,386295 es exactamente `-log(0,5) - log(0,5)`, que es el valor que toma la loss de
skip-gram con negative sampling cuando **los productos punto son cero**. O sea: los embeddings
de salida nunca se mueven de su inicializacion, que es cero.

Lo que se probo para arreglarlo, y no funciono
---------------------------------------------
    escala de inicializacion 0,5/dim  -> 1/sqrt(dim), 28 veces mayor      loss 1,3863
    learning rate 0,025 -> 0,5, veinte veces mayor                        loss 1,3863
    lote 8192 -> 256, con el lr escalado por el lote                      loss 1,3863
    optimizador Adam -> SGD con decaimiento lineal                        loss 1,3863

En todas: la loss se queda clavada en el valor del estado cero.

Lo que SI se movio
------------------
En una prueba aislada, con el learning rate escalado por el tamano del lote, los pesos **si**
cambiaron (`dW = 0,000763`). Asi que los parametros se actualizan en algunas configuraciones,
pero la loss no baja en ninguna. Hay una incoherencia que no he aislado.

Lo que esto significa
---------------------
**Las dos corridas entrenaron representaciones practicamente aleatorias.** El veredicto
`CLOSED`, calculado sobre esas representaciones, **no es valido**. No dice nada sobre la
hipotesis de dominio.

Esto no es un matiz: es la diferencia entre un resultado y un artefacto. La conclusion honesta
es que **la hipotesis sigue sin ponerse a prueba**, exactamente igual que en P0 antes de tener
corpus.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

REPORTS = PROJECT_ROOT / "reports"


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-07] {m}", flush=True)


def main() -> None:
    """Marca el veredicto como invalido y deja constancia.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-07 invalidacion del veredicto: las representaciones no entrenaron")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    loss_cero = -math.log(0.5) - math.log(0.5)
    log("")
    log("   la loss observada, clavada:      1.386295")
    log(f"   la loss del estado cero:        {loss_cero:.6f}")
    log(f"   coinciden:                      {abs(1.386295 - loss_cero) < 1e-5}")
    log("")
    log("   el valor del estado cero es el de los productos punto nulos,")
    log("   o sea los embeddings de SALIDA sin moverse de cero")

    intentos = [
        {"cambio": "escala de inicializacion 0.5/dim -> 1/sqrt(dim) (28x)", "loss": 1.3863},
        {"cambio": "learning rate 0.025 -> 0.5 (20x)", "loss": 1.3863},
        {"cambio": "lote 8192 -> 256 con el lr escalado por el lote", "loss": 1.3863},
        {"cambio": "Adam -> SGD con decaimiento lineal", "loss": 1.3863},
    ]
    log("")
    log("   LO QUE SE PROBO PARA ARREGLARLO, Y NO FUNCIONO")
    for i in intentos:
        log(f"      {i['cambio']:52} loss {i['loss']}")

    diagnostico = {
        "familia": "P71_TWITTER_WORD2VEC", "experiment_id": "P71-07",
        "fase": "invalidacion del veredicto",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "AUDITORIA DEL PROPIO RESULTADO. 0 experimentos nuevos",
        "disparador": ("Carlos observo que la loss subia por epoca y pregunto si el modelo no "
                       "estaba aprendiendo. Tenia razon"),
        "evidencia": {
            "loss_observada_clavada": 1.386295,
            "loss_del_estado_cero": round(loss_cero, 6),
            "coincide": bool(abs(1.386295 - loss_cero) < 1e-5),
            "interpretacion": ("la loss es la de los productos punto nulos, o sea los embeddings "
                               "de salida no se mueven de cero"),
            "curvas_adam_previas": {"twitter": [1.706327, 1.811410, 1.841703, 1.865799,
                                                1.884080],
                                    "general": [1.887839, 2.068209, 2.125895, 2.163865,
                                                2.189914]},
        },
        "intentos_que_no_funcionaron": intentos,
        "lo_que_si_se_movio": ("con el lr escalado por el lote los pesos cambian, dW=0.000763, "
                              "pero la loss no baja en ninguna configuracion"),
        "consecuencia": {
            "las_representaciones_no_entrenaron": True,
            "el_veredicto_CLOSED_es_valido": False,
            "veredicto_anterior": "CLOSED",
            "veredicto_corregido": "INVALID_DUE_TO_UNTRAINED_REPRESENTATIONS",
            "que_dice_sobre_la_hipotesis": ("NADA. La hipotesis sigue sin ponerse a prueba, "
                                            "igual que en P0 antes de tener corpus"),
        },
        "bug_no_aislado": True,
        "lo_que_haria_falta": [
            "aislar por que la loss no baja aunque los pesos cambien",
            ("candidato: el gradiente hacia la salida se anula o el optimizador no ve esos "
            "parametros"),
            ("candidato: la inicializacion a cero de la salida combinada con la escala de la "
            "entrada"),
            ("alternativa: usar un skip-gram con softmax negativo implementado y verificado "
            "contra una implementacion de referencia"),
        ],
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_invalidacion.json").write_text(
        json.dumps(diagnostico, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Marcar el informe de dominio con la invalidacion, sin reescribir sus numeros.
    md = REPORTS / "twitter_word2vec_domain_analysis.md"
    t = md.read_text(encoding="ascii")
    aviso = ("# AVISO: VEREDICTO INVALIDADO\n\n"
             "> **El veredicto `CLOSED` de este informe NO es valido.**\n"
             ">\n"
             "> Carlos observo que la loss subia por epoca. La revision confirmo que la loss se\n"
             "> queda clavada en 1,386295, que es exactamente el valor del estado de productos\n"
             "> punto nulos: **los embeddings de salida nunca se movieron de cero y las\n"
             "> representaciones no entrenaron**.\n"
             ">\n"
             "> Los numeros de este informe se calcularon sobre representaciones practicamente\n"
             "> aleatorias. **No dicen nada sobre la hipotesis de dominio.**\n"
             ">\n"
             "> Detalle en `reports/twitter_word2vec_invalidacion.json`.\n\n---\n\n")
    if "VEREDICTO INVALIDADO" not in t:
        md.write_text(aviso + t, encoding="ascii")
        log(f"   aviso anadido a {md.name}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    docs = [md, REPORTS / "twitter_word2vec_invalidacion.json"]
    sucios = [d.name for d in docs if any(b > 127 for b in d.read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")

    log("")
    log("=" * 78)
    log("P71-07 OK | veredicto INVALIDADO")
    log("   veredicto anterior  CLOSED")
    log("   veredicto corregido INVALID_DUE_TO_UNTRAINED_REPRESENTATIONS")
    log("   la hipotesis sigue SIN PONERSE A PRUEBA")
    log(f"   commit {commit}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()

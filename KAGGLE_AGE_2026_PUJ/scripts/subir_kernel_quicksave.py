"""Sube un cuaderno ya ejecutado a un kernel existente de Kaggle sin ejecutarlo.

La CLI oficial (`kaggle kernels push`) siempre lanza la ejecucion y, ademas, borra las
salidas de las celdas antes de enviar. Este script usa el mismo cliente autenticado con
`kernel_execution_type = QUICK_SAVE`, que guarda una version nueva conservando el
contenido y las salidas del cuaderno, sin crear una sesion de ejecucion.

Uso:
    /home/carlos/venv/bin/python scripts/subir_kernel_quicksave.py \
        --slug carlosgarnica/nlp02-notebook-age-consolidado \
        --notebook notebooks/99_entrega_consolidado.ipynb \
        --metadata /ruta/kernel-metadata.json

El metadata se obtiene con:
    kaggle kernels pull <slug> -p <carpeta> -m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick Save de un cuaderno en Kaggle (sin ejecutar)")
    ap.add_argument("--slug", required=True, help="slug del kernel, p. ej. usuario/mi-kernel")
    ap.add_argument("--notebook", required=True, help="ruta del .ipynb local ya ejecutado")
    ap.add_argument("--metadata", required=True, help="kernel-metadata.json descargado con pull -m")
    args = ap.parse_args()

    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    nb = json.loads(Path(args.notebook).read_text(encoding="utf-8"))

    # El servidor espera `source` como una sola cadena; el resto del cuaderno (salidas,
    # execution_count y metadatos) se envia tal cual para conservar la ejecucion.
    for celda in nb.get("cells", []):
        if isinstance(celda.get("source"), list):
            celda["source"] = "".join(celda["source"])

    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        request = ApiSaveKernelRequest()
        request.slug = args.slug
        if meta.get("title"):
            request.new_title = meta["title"]
        request.text = json.dumps(nb, ensure_ascii=False)
        request.language = meta.get("language", "python")
        request.kernel_type = meta.get("kernel_type", "notebook")
        request.is_private = bool(meta.get("is_private", True))
        request.enable_internet = bool(meta.get("enable_internet", True))
        if meta.get("enable_gpu"):
            request.enable_gpu = True
        request.dataset_data_sources = meta.get("dataset_sources", [])
        request.competition_data_sources = meta.get("competition_sources", [])
        request.kernel_data_sources = meta.get("kernel_sources", [])
        request.model_data_sources = meta.get("model_sources", [])
        # No se envian docker_image ni enable_gpu: la API rechaza el primero en este
        # endpoint (HTTP 400) y el segundo cambiaria la maquina del kernel sin necesidad.
        # La imagen y la maquina configuradas en el kernel se conservan tal cual.
        request.kernel_execution_type = KernelExecutionType.QUICK_SAVE

        respuesta = kaggle.kernels.kernels_api_client.save_kernel(request)

    print("respuesta:", respuesta)
    print(f"version nueva guardada (sin ejecutar): https://www.kaggle.com/code/{args.slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

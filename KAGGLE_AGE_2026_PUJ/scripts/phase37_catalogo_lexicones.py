"""FASE 37 (P1) — Catálogo de lexicones y cribado de aplicabilidad al corpus.

Escribe ``lexicons/<familia>/<subcategoria>.txt`` (español con tildes) y genera
``reports/age_lexicon_catalog.csv`` con la trazabilidad de §9 y §31.

Dos cribados objetivos, ambos sin usar etiquetas:

1. **Validez ortográfica** (DD-2, DD-5): solo letras del español, se descartan las formas que
   colisionan al transliterar y los tokens presentes en más de una subcategoría.
2. **Aplicabilidad al corpus** (nuevo en P1): una entrada que no aparece en el vocabulario del
   train no puede aportar señal, por muy correcta que sea. Se exige frecuencia mínima y se reporta
   el número de entradas descartadas por este motivo.

Las listas son conocimiento lingüístico general elaborado en el proyecto (DD-1): sin corpus
externo, sin modelos, sin embeddings, sin etiquetas de edad, sin pseudo-labels (§1, §5, §13).

Reconstruido tras el incidente git del 2026-09-25.

Salida: ``lexicons/**``, ``reports/age_lexicon_catalog.csv``,
``experiments/phase37_catalogo_lexicones.json``, ``experiments/phase37_entradas_descartadas.txt``.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src import lexicon_lists as ll  # noqa: E402
from src import lexicons as lx  # noqa: E402

FECHA = "2026-09-25"  # constante, no date.today(): la fecha entra en la cabecera de los .txt y un
# date() cambiaria los SHA256 en cada reejecucion, rompiendo la trazabilidad.
CATALOGO = cfg.REPORTS / "age_lexicon_catalog.csv"
RESULTADO = cfg.EXPERIMENTS / "phase37_catalogo_lexicones.json"
DESCARTES = cfg.EXPERIMENTS / "phase37_entradas_descartadas.txt"
MIN_FRECUENCIA = 3
MUESTRA_COBERTURA = 8000

CABECERA = ("# familia: {familia}\n"
            "# subcategoria: {subcategoria}\n"
            "# descripcion: {descripcion}\n"
            "# idioma: es\n"
            "# procedencia: elaborado en este proyecto (conocimiento linguistico general)\n"
            "# licencia: del proyecto; sin datos externos y sin etiquetas de edad\n"
            "# riesgo: ninguno\n"
            "# fecha: {fecha}\n"
            "# entradas_declaradas: {declaradas}\n"
            "# cribado_corpus: frecuencia minima {minimo} en el train\n")


def vocabulario_corpus(textos) -> Counter:
    """Frecuencia de tokens en el corpus de train (sin usar etiquetas)."""
    conteo: Counter = Counter()
    for valor in textos.fillna("").astype(str):
        conteo.update(lx.tokeniza(valor))
    return conteo


def main() -> int:
    """Escribe los lexicones cribados y construye el catálogo con su trazabilidad."""
    train = datos.load_train()
    conteo = vocabulario_corpus(train["text"])
    documentos = [lx.tokeniza(valor) for valor in train["text"].iloc[:MUESTRA_COBERTURA]]

    descartes: list[str] = []
    filas = []
    for familia, subcategorias in ll.LISTAS.items():
        directorio = lx.LEXICONES / familia
        directorio.mkdir(parents=True, exist_ok=True)
        for subcategoria, palabras in subcategorias.items():
            declaradas = sorted({p.strip() for p in palabras if p.strip()})
            supervivientes = []
            for palabra in declaradas:
                token = lx.transliterar(palabra)
                if len(token) < 2 or token in lx.FORMAS_AMBIGUAS:
                    descartes.append(f"{familia}/{subcategoria}: {palabra} (ortografia/ambiguedad)")
                    continue
                frecuencia = conteo.get(token, 0)
                if frecuencia < MIN_FRECUENCIA:
                    descartes.append(f"{familia}/{subcategoria}: {palabra} "
                                     f"(no aparece en el corpus, frecuencia={frecuencia})")
                    continue
                supervivientes.append(palabra)
            ruta = directorio / f"{subcategoria}.txt"
            ruta.write_text(
                CABECERA.format(familia=familia, subcategoria=subcategoria,
                                descripcion=ll.DESCRIPCIONES[subcategoria], fecha=FECHA,
                                declaradas=len(declaradas), minimo=MIN_FRECUENCIA)
                + "\n".join(sorted(supervivientes)) + "\n", encoding="utf-8")
            filas.append({
                "nombre": f"{familia}/{subcategoria}",
                "familia": familia,
                "subcategoria": subcategoria,
                "tipo": {"registro": "estilo", "vida": "contexto vital", "temas": "contenido"}[familia],
                "fuente": "Elaborado en este proyecto",
                "autor": "Equipo AGE (KAGGLE_AGE_2026_PUJ)",
                "version": "1.0",
                "fecha": FECHA,
                "licencia": "Del proyecto; conocimiento linguistico general, sin datos externos",
                "url": "no aplica (recurso interno, no descargado)",
                "proposito_original": "Senal de registro / etapa vital / dominio tematico",
                "idioma": "es",
                "entradas_declaradas": len(declaradas),
                "entradas_efectivas": len(supervivientes),
                "descartadas": len(declaradas) - len(supervivientes),
                "cobertura_corpus_pct": round(100.0 * len(supervivientes) / max(1, len(declaradas)), 1),
                "formato": "txt, una palabra por linea, encabezado con metadatos",
                "riesgo_academico": "ninguno: sin embeddings, sin modelos, sin etiquetas, sin corpus externo",
                "usa_etiquetas_de_edad": False,
                "genera_pseudo_labels": False,
                "modifica_el_texto": False,
                "construye_vectorizador": False,
                "sha256": lx.sha256_de(ruta),
                "ruta": str(ruta.relative_to(cfg.ROOT)),
            })
    with CATALOGO.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    DESCARTES.write_text("\n".join(descartes) + "\n", encoding="utf-8")

    lexicon = lx.carga_lexicones()
    resumen = {
        "fase": "P1 — catalogo de lexicones y cribado de aplicabilidad",
        "fecha": FECHA,
        "subcategorias": len(filas),
        "entradas_declaradas": sum(f["entradas_declaradas"] for f in filas),
        "entradas_efectivas": sum(f["entradas_efectivas"] for f in filas),
        "descartadas": len(descartes),
        "min_frecuencia_corpus": MIN_FRECUENCIA,
        "tokens_unicos_tras_ambiguedad": len(lexicon["token_a_subcategoria"]),
        "tokens_ambiguos_descartados": len(lexicon["ambiguos"]),
        "cobertura_por_familia": {f: lx.cobertura(documentos, lexicon, f) for f in lx.FAMILIAS},
        "subcategorias_retiradas_por_minimo": lexicon["descartadas_por_minimo"],
        "nota_cribado": "el cribado de frecuencia minima usa el vocabulario del train completo: es "
                        "seleccion no supervisada y transductiva, sin usar etiquetas en ningun punto",
    }
    RESULTADO.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"subcategorias: {len(filas)} | declaradas: {resumen['entradas_declaradas']} | "
          f"efectivas: {resumen['entradas_efectivas']} | descartadas: {resumen['descartadas']}")
    print(f"tokens unicos: {resumen['tokens_unicos_tras_ambiguedad']} | ambiguos: "
          f"{resumen['tokens_ambiguos_descartados']} | retiradas por minimo: "
          f"{resumen['subcategorias_retiradas_por_minimo']}")
    for familia, medida in resumen["cobertura_por_familia"].items():
        print(f"  cobertura {familia}: {medida['cobertura_pct']:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

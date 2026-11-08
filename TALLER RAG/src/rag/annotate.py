"""App de anotación humana de respuestas. Exporta CSV ciego por anotador.

La lista de respuestas se baraja con semilla fija y cada una recibe un
identificador ciego. El mapeo a modelo vive en eval/annotations/key.csv y la
app no lo muestra, para que el anotador juzgue el texto y no la etiqueta.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rag.annotation import EJES, blind_id, load_items, write_key

OUT_DIR = Path("eval/annotations")

st.set_page_config(page_title="Anotación RAG", layout="wide")
st.title("Anotación de respuestas")

with st.sidebar:
    annotator = st.text_input("Anotador", value="")
    run_file = st.text_input("Archivo de corrida", value="reports/corridas/e4_test.jsonl")
    silver_file = st.text_input("Silver", value="eval/silver/silver_standard.jsonl")

if not annotator.strip():
    st.info("Escribe el nombre del anotador para empezar.")
    st.stop()

run_path = Path(run_file)
silver_path = Path(silver_file)
if not run_path.exists() or not silver_path.exists():
    st.error("No se encontró el archivo de corrida o el silver.")
    st.stop()

silver, records = load_items(run_path, silver_path)
try:
    write_key(records, OUT_DIR)
except ValueError as error:
    st.error(f"{error}. Bórralo solo si cambió el experimento.")
    st.stop()

out_path = OUT_DIR / f"{annotator.strip().replace(' ', '_')}.csv"
done: set[str] = set()
if out_path.exists():
    with out_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add(row["item_id"])
    st.caption(f"Anotadas: {len(done)} de {len(records)}")

pending = [r for r in records if blind_id(r["model"], r["question_id"], int(r["rep"])) not in done]
if not pending:
    st.success("No quedan respuestas pendientes. Entrega el CSV al coordinador.")
    st.stop()

if "index" not in st.session_state:
    st.session_state.index = 0
index = min(st.session_state.index, len(pending) - 1)
record = pending[index]
item_id = blind_id(record["model"], record["question_id"], int(record["rep"]))
question = silver.get(record["question_id"], {})
response = record["response"]

st.progress(index / max(len(pending), 1), text=f"{index + 1} de {len(pending)}")
st.subheader(question.get("question", record["question_id"]))
st.caption(f"tipo={question.get('type')} | idioma={question.get('lang')} | k={record.get('k')}")
st.markdown(f"**Respuesta de referencia:** {question.get('reference_answer', 'sin referencia')}")
if question.get("reference_span"):
    st.markdown(f"**Span de referencia:** {question['reference_span']}")
if question.get("criteria"):
    st.caption(f"Criterio: {question['criteria']}")

st.markdown("**Respuesta del modelo:**")
st.markdown(response.get("answer", ""))
if response.get("invalid_citations"):
    st.warning(f"Citas fuera de rango: {response['invalid_citations']}")
if response.get("abstained"):
    st.info(f"Abstención: {response.get('abstain_reason')}")

with st.expander(f"Evidencias recuperadas ({len(response.get('evidence', []))})"):
    for i, hit in enumerate(response.get("evidence", []), start=1):
        st.markdown(f"**[{i}] {hit.get('title')}** ({hit.get('language')}, {hit.get('created_at')})")
        st.write(hit.get("text", "")[:800])
        st.caption(hit.get("url"))

with st.expander("Rúbrica (0 a 3)"):
    st.markdown(
        "- **Corrección**: 3 responde lo que pide la pregunta; 2 correcta con imprecisión; "
        "1 parcial o con error relevante; 0 incorrecta o no responde.\n"
        "- **Fidelidad**: 3 todo el contenido sale de la evidencia; 2 algún añadido menor; "
        "1 inventa o contradice parte; 0 inventa el núcleo.\n"
        "- **Citas**: 3 todas válidas y apuntan al pasaje que sostiene la respuesta; "
        "2 válidas pero no al pasaje; 1 parciales; 0 sin citas o todas inválidas.\n"
        "- **Abstención**: 3 responde cuando hay evidencia; 2 responde con duda razonable; "
        "1 se abstiene teniendo evidencia clara; 0 se abstiene sin motivo."
    )

col1, col2, col3, col4 = st.columns(4)
correccion = col1.selectbox("Corrección", [0, 1, 2, 3], index=None, placeholder="Elige")
fidelidad = col2.selectbox("Fidelidad", [0, 1, 2, 3], index=None, placeholder="Elige")
citas = col3.selectbox("Citas", [0, 1, 2, 3], index=None, placeholder="Elige")
abstencion = col4.selectbox("Abstención", [0, 1, 2, 3], index=None, placeholder="Elige")
nota = st.text_input("Nota (opcional)")

if st.button("Guardar y siguiente", type="primary"):
    if None in (correccion, fidelidad, citas, abstencion):
        st.warning("Faltan ejes por calificar.")
    else:
        new_file = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["item_id", "question_id", "rep", "annotator", *EJES, "nota"],
            )
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "item_id": item_id,
                    "question_id": record["question_id"],
                    "rep": int(record["rep"]),
                    "annotator": annotator.strip(),
                    "correccion": correccion,
                    "fidelidad": fidelidad,
                    "citas": citas,
                    "abstencion": abstencion,
                    "nota": nota,
                }
            )
        st.session_state.index = index + 1
        st.rerun()

# Frontend Juris-PLN (Streamlit)

Frontend separado del backend. Solo consume la API por HTTP; no importa el núcleo ni los modelos.

## Correr

```bash
cd "PROYECTO/ENTREGA FINAL/app/frontend"
API_URL=http://127.0.0.1:8123 \
  /home/carlos/venv/bin/streamlit run streamlit_app.py --server.port 8502 --server.headless=true
```

El puerto 8501 está ocupado por otro proyecto; usa 8502 (o el que esté libre).

## Requisitos

- Backend corriendo (ver `../backend/README.md`).
- `API_URL` apunta al backend (default `http://127.0.0.1:8123`).
- GPU solo si el chat carga el LLM; el resto corre en CPU.

## Pantallas

- **P1 Sentencia a Constitución**: validación M0, área temática (exploratoria), artículos
  extraídos y recuperados con score y bloque, sentencias citadas, evidencia textual por artículo
  (fragmentos de la propia sentencia con score BM25) y descarga del JSON.
- **P2 Artículo a jurisprudencia**: total de sentencias que citan el artículo, distribución por
  año y top 10 por PageRank con filtro por tipo (T/C/SU). Conteos desde etiquetas débiles de M2
  (no gold); el área silver se rotula "referencia automática".
- **P3 Evaluación**: métricas con procedencia y avisos.
- **Chat**: agente con guardrails; selector de modelo en la barra lateral.

## Estados y límites

- Estados vacío, carga y error en cada pantalla.
- Sin datos inventados: lo que la API no expone se muestra como "no expuesto".
- El área temática se rotula "exploratoria" (ADR-19).
- El selector de modelo usa los nombres exactos de ollama; si `tool_calling=false`, se avisa del
  fallback determinista.

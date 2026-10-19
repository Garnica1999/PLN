# P98-01 (Fase A): prueba de infraestructura con datos sinteticos

- Device: AMD Radeon RX 9070 XT | torch 2.14.0+rocm7.2 | ROCm OK.
- Modelo sintetico: 900,992 parametros (char-aware + LSTM + softmax atado).
- Loss train: 8.5559 -> 5.4826; val 5.4464 (converge: True).
- Throughput: 223,167 tokens/s (18.35 ms/paso); VRAM pico 0.56 GB.
- Checkpoint roundtrip identico: True.
- Determinismo (dos corridas de 30 pasos, misma semilla): delta de loss 0.00e+00.
- Tiempo total de la prueba: 8.2 s.

Sin corpus reales (restriccion de Fase A cumplida). El esqueleto EncoderLM y el
pipeline de lotes se reutilizan en C98-0/C98-1 con configuracion mayor.

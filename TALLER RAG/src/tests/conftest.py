"""Registra el marcador de integración dentro del contenedor."""


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "integration: requiere la API y el stack levantados")

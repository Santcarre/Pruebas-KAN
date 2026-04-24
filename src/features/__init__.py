"""
Módulo de extracción de features para el pipeline ResNet-18 + KAN.

Uso planificado:
  - Extraer y guardar embeddings del backbone ResNet-18 (512-dim) en disco
  - Cargar embeddings precalculados para entrenar el KAN head de forma eficiente
  - Facilitar el análisis interpretativo sin necesidad de reejecutar el backbone

Ejemplo futuro:
    from src.features import extract_embeddings
    embeddings = extract_embeddings(dataloader, backbone, device='cuda')
"""

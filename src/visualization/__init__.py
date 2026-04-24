"""
Módulo de visualización avanzada para el pipeline ResNet-18 + KAN.

Uso planificado:
  - Plots comparativos de splines KAN por clase (maligno vs benigno)
  - Exportación automática de figuras de interpretabilidad a reports/
  - Curvas ROC, histogramas de probabilidad y matrices de confusión
  - Grid plots resumen de las top-K aristas más importantes (en lugar de
    miles de imágenes individuales sp_*.png)

Ejemplo futuro:
    from src.visualization import plot_class_splines
    plot_class_splines(kan_model, benign_acts, malignant_acts, layer=0)
"""

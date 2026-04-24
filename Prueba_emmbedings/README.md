# Prueba_emmbedings

Flujo solicitado:

1. Extraer embeddings con ResNet18 tuneado.
2. Entrenar KAN para clasificar esos embeddings.
3. Aplicar Sproud sobre aristas KAN.
4. Intentar gráficas; si no se puede, guardar fórmula simbólica.

## Prerrequisitos

- Haber generado manifiestos:
  - `src/data/processed/manifest_train.csv`
  - `src/data/processed/manifest_test.csv`
- Tener checkpoint backbone:
  - `reports/models/resnet18_tuned_best.pt`
- Dependencias:
  - `torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`
  - `pykan` (`pip install pykan`)

## Ejecución (desde la raíz del repo)

```bash
python Prueba_emmbedings/01_extract_embeddings.py
python Prueba_emmbedings/02_train_kan_on_embeddings.py
python Prueba_emmbedings/03_sproud_symbolic_report.py
```

## Salidas

Se guardan en `Prueba_emmbedings/artifacts/`:

- `embeddings.pt`: tensores `X_train/X_val/X_test` y `y_*`, más stats `mu/sigma`.
- `kan_head.pt`: pesos del KAN entrenado.
- `kan_config.json`: configuración mínima para reconstruir KAN.
- `training_summary.json`: métricas (val/test) y umbral calibrado por recall.
- `sproud_edges.csv`: top aristas por importancia tipo Sproud.
- `symbolic_edges.csv`: ajuste simbólico por arista seleccionada.
- `symbolic_formula.txt`: fórmula global (si `symbolic_formula()` está disponible).
- `sproud_edges_layer0.png`: gráfico de aristas (si se pudo renderizar).

## Notas

- El split train/val se hace por `patient_id` para reducir leakage.
- La normalización de embeddings (z-score) se calcula en train y se aplica a val/test.
- Sproud usa `mean(abs(spline_postacts))` por arista en una capa objetivo.

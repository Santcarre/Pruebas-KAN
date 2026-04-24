# Breast Cancer Interpretable ML

> **Tesis de Licenciatura en Matemáticas**  
> Exploración de interpretabilidad en clasificación de mamografías mediante Kolmogorov-Arnold Networks (KAN) sobre features de ResNet-18.

## Descripción

Este proyecto investiga si las **Kolmogorov-Arnold Networks (KAN)** pueden proporcionar interpretabilidad matemática significativa en la clasificación de mamografías del dataset **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM).

La idea central es sustituir el clasificador lineal estándar de ResNet-18 por una KAN, cuyas conexiones son **funciones spline aprendidas** (en lugar de pesos escalares). Esto permite:

- Visualizar la función que cada conexión aprendió
- Aplicar regresión simbólica (`auto_symbolic`) para obtener expresiones matemáticas
- Podar la red y determinar cuántas dimensiones del espacio de features son realmente relevantes

## Arquitectura

```
Mamografía (JPEG)
      ↓
 ResNet-18 pretrenado (ImageNet)
      ↓ GlobalAveragePooling → vector 512-dim
      ↓
  KAN Head (PyKAN)
      ↓
 Clasificación binaria: Maligno / Benigno
```

## Estructura del Proyecto

```
.
├── notebooks/                  # Notebooks del pipeline (en orden)
│   ├── 00_eda.ipynb            # Análisis exploratorio del dataset
│   ├── 01_resnet18_pipeline.ipynb      # Baseline ResNet-18
│   ├── 02_resnet18_tuned.ipynb         # Fine-tuning con class weights
│   ├── 03_resnet18_thresholding.ipynb  # Optimización del umbral por recall
│   ├── 04_resnet18_kan_fast.ipynb      # KAN head (esploración rápida)
│   ├── 05_resnet18_kan_end2end_train.ipynb  # Entrenamiento end-to-end
│   ├── 06_resnet18_kan_interpretability.ipynb  # Análisis de splines
│   ├── 07_resnet18_kan_singlephase_train.ipynb # Entrenamiento single-phase
│   └── 08_resnet18_kan_interpretability.ipynb  # Análisis completo
│
├── scripts/                    # Utilidades compartidas
│   ├── paths.py                # Rutas centralizadas del repo
│   ├── notebook_utils.py       # Dataset, transforms, métricas, threshold
│   ├── kan_plot_utils.py       # Visualización de splines KAN
│   └── build_resnet18_manifest.py  # Genera manifests CSV desde CBIS-DDSM
│
├── src/
│   ├── data/
│   │   └── processed/          # Manifests generados (no en Git)
│   ├── features/               # (futuro) Extracción de embeddings
│   └── visualization/          # (futuro) Plots avanzados
│
├── reports/
│   └── models/                 # Modelos entrenados (no en Git, >44MB)
│
├── requirements.txt
└── README.md
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Santcarre/Pruebas-KAN.git
cd Pruebas-KAN
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Instalar PyTorch con soporte CUDA

> **RTX 5060 (Blackwell / GB207):** requiere PyTorch ≥ 2.6 y CUDA ≥ 12.6.

```bash
# Para RTX 5060 / CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Verificar instalación
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4. Instalar resto de dependencias

```bash
pip install -r requirements.txt
```

### 5. Registrar el kernel de Jupyter

```bash
python -m ipykernel install --user --name breast-cancer-kan
```

## Datos: CBIS-DDSM

Los datos **no se incluyen en el repositorio** por tamaño y licencia.

### Descarga

1. Ir a [TCIA — CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
2. Descargar con el **NBIA Data Retriever**:
   - Imágenes JPEG completas → `src/data/raw/jpeg/`
3. Descargar los **CSV de casos** desde la misma página → `src/data/raw/csv/`

### Generar manifests

```bash
python scripts/build_resnet18_manifest.py
```

Esto genera `src/data/processed/manifest_{all,train,test}.csv`.

## Pipeline de Notebooks

Ejecutar en orden:

| # | Notebook | Descripción |
|---|----------|-------------|
| 00 | `00_eda.ipynb` | Análisis exploratorio del CBIS-DDSM |
| 01 | `01_resnet18_pipeline.ipynb` | Baseline ResNet-18 (sin fine-tuning) |
| 02 | `02_resnet18_tuned.ipynb` | Fine-tuning con class weights y augmentation |
| 03 | `03_resnet18_thresholding.ipynb` | Umbral óptimo por recall clínico |
| 04 | `04_resnet18_kan_fast.ipynb` | KAN head: exploración rápida |
| 05 | `05_resnet18_kan_end2end_train.ipynb` | Entrenamiento end-to-end ResNet + KAN |
| 06 | `06_resnet18_kan_interpretability.ipynb` | Splines, poda, `auto_symbolic` |
| 07 | `07_resnet18_kan_singlephase_train.ipynb` | Entrenamiento single-phase |
| 08 | `08_resnet18_kan_interpretability.ipynb` | Análisis interpretativo completo |

## Notas sobre la GPU (RTX 5060 / Blackwell)

Si PyKAN da errores en GPU, el flujo recomendado es:

```python
# 1. Extraer embeddings con ResNet-18 en GPU (rápido)
embeddings = backbone(images.to('cuda'))
torch.save(embeddings, 'kan_head_embeddings.pt')

# 2. Entrenar KAN head sobre los embeddings guardados (CPU o GPU)
# PyKAN es más estable en este modo con tensores ya extraídos
```

Si persisten errores con einsum en Blackwell:
```python
torch.backends.cuda.matmul.allow_tf32 = False
```

## Interpretabilidad: Funcionalidades Clave de PyKAN

```python
# Regresión simbólica sobre las aristas aprendidas
kan_model.auto_symbolic(lib=['x', 'x^2', 'exp', 'log', 'sin', 'abs'])

# Poda de aristas poco relevantes
kan_model.prune(threshold=0.01)

# Visualización de la red
kan_model.plot()
```

## Citas

- **Liu et al. (2024)** — KAN: Kolmogorov-Arnold Networks. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- **CBIS-DDSM** — Lee et al. (2017). A Curated Mammography Data Set for Use in Computer-Aided Detection and Diagnosis Research. *Scientific Data*.
- **ResNet** — He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

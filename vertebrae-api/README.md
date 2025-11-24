# API de Segmentación de Vértebras

API REST desarrollada en FastAPI. Modelos: **YOLOv8** y **Mask R-CNN**. Segmenta 17 clases de vértebras (T1-T12, L1-L5) en radiografías de columna.

## Características

- **Soporte multi-modelo**: YOLOv8 (Ultralytics) y Mask R-CNN (Detectron2)
- **17 clases de vértebras**: T1-T12 (torácicas), L1-L5 (lumbares)
- **Inferencia en CPU**
- **Almacenamiento en S3** con descarga automática y almacenamiento en caché
- **Múltiples endpoints**: Predicciones JSON e imágenes visualizadas.

## Arquitectura

```
vertebrae-api/
├── app/
│   ├── main.py              # Aplicación FastAPI con endpoints
│   ├── config.py            # Gestión de configuración
│   ├── services.py          # Capa de lógica de negocio
│   ├── schemas.py           # Modelos Pydantic para validación
│   ├── utils.py             # S3, procesamiento de imágenes
│   └── models/
│       ├── __init__.py      
│       ├── base.py          # Interfaz base para modelos
│       ├── factory.py       # Factoría y gestión de modelos
│       ├── yolo.py          # Implementación YOLOv8
│       └── maskrcnn.py      # Implementación Mask R-CNN
├── Dockerfile               # Definición del contenedor
├── docker-compose.yml       # Configuración Docker Compose
├── requirements.txt         # Dependencias Python
└── .env.example             # Plantilla de variables de entorno
```

## Prerequisitos

- Python 3.10+
- Docker y Docker Compose (para despliegue en contenedor)
- Credenciales AWS con acceso a S3 (opcional si los modelos están en caché local)
- Archivos de modelos en S3:
  - YOLOv8: `s3://vertebrae-artifacts/yolo_best.pt`
  - Mask R-CNN: `s3://vertebrae-artifacts/model_final.pth`

## Instalación

### 1. Clonar y Configurar

```bash
cd vertebrae-api
cp .env.example .env
```

### 2. Configurar Variables de Entorno

Edite `.env` con su configuración:

```bash
# Configuración S3
S3_BUCKET=vertebrae-artifacts
MODEL_CACHE_DIR=/tmp/model_cache

# Selección de Modelo (yolo o maskrcnn)
DEFAULT_MODEL=yolo

# Configuración Modelo YOLO
YOLO_MODEL_KEY=yolo_best.pt

# Configuración Modelo Mask R-CNN
MASKRCNN_MODEL_KEY=model_final.pth
MASKRCNN_BACKBONE=COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

# Configuración API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Configuración de Inferencia del Modelo
CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.5
MAX_DETECTIONS=100
DEVICE=cpu

# Registro
LOG_LEVEL=INFO
```

### 3. Opción A: Despliegue con Docker (Recomendado)

```bash
# Construir y ejecutar con Docker Compose
docker-compose up --build

# O con Docker directamente
docker build -t vertebrae-api .
docker run -p 8000:8000 --env-file .env vertebrae-api
```

### 3. Opción B: Desarrollo Local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias usando el script de instalación (recomendado)
chmod +x install.sh
./install.sh

# O instalar manualmente (requiere orden específico)
# 1. Instalar PyTorch primero
pip install torch==2.1.2 torchvision==0.16.2

# 2. Instalar Detectron2 (requiere --no-build-isolation)
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 3. Instalar dependencias restantes
pip install -r requirements.txt

# Ejecutar la API
python -m app.main

# O con uvicorn
uvicorn app.main:app --reload
```

**Nota**: El script [install.sh](install.sh) maneja el orden correcto de instalación para todas las dependencias, especialmente Detectron2 que requiere flags especiales.

La API estará disponible en `http://localhost:8000`

## Endpoints de la API

### 1. Información Raíz
```bash
GET /
```

Devuelve información de la API, modelos soportados y endpoints disponibles.

**Respuesta:**
```json
{
  "name": "Vertebrae Segmentation API",
  "version": "0.2.0",
  "description": "Multi-model segmentation for 17 vertebrae classes",
  "supported_models": ["yolo", "maskrcnn"],
  "default_model": "yolo"
}
```

### 2. Información de Todos los Modelos
```bash
GET /models
```

Obtiene información sobre todos los modelos cargados.

**Respuesta:**
```json
{
  "models": {
    "yolo": {
      "model_name": "YOLOv8-seg",
      "model_type": "yolo",
      "num_classes": 17,
      "classes": ["T1", "T2", ..., "L5"],
      "backbone": "YOLOv8",
      "device": "cpu",
      "confidence_threshold": 0.5,
      "nms_threshold": 0.5,
      "framework": "Ultralytics"
    },
    "maskrcnn": {
      "model_name": "Mask R-CNN",
      "model_type": "maskrcnn",
      "num_classes": 17,
      "classes": ["T1", "T2", ..., "L5"],
      "backbone": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
      "device": "cpu",
      "confidence_threshold": 0.5,
      "nms_threshold": 0.5,
      "framework": "Detectron2"
    }
  },
  "default_model": "yolo"
}
```

### 5. Predicción (Respuesta JSON)
```bash
POST /predict
POST /predict?model=yolo
POST /predict?model=maskrcnn
```

Suba una imagen y reciba predicciones JSON con cajas delimitadoras, máscaras (RLE), puntuaciones y clases.

**Solicitud:**
```bash
curl -X POST "http://localhost:8000/predict?model=yolo" \
  -F "file=@spine_xray.jpg"
```

**Respuesta:**
```json
{
  "detections": [
    {
      "bbox": {
        "x1": 245.3,
        "y1": 102.7,
        "x2": 312.8,
        "y2": 156.4
      },
      "mask": {
        "size": [512, 512],
        "counts": "RLE_encoded_string..."
      },
      "score": 0.95,
      "class_name": "T1",
      "class_id": 0
    }
  ],
  "num_detections": 14,
  "image_shape": [512, 512, 3],
  "processing_time_ms": 234.56,
  "model_used": "yolo"
}
```

### 6. Predicción con Visualización
```bash
POST /predict/visualize
POST /predict/visualize?model=yolo
POST /predict/visualize?model=maskrcnn
```

Al subir una imágen el API devuelve la imágen anotada.

**Solicitud:**
```bash
curl -X POST "http://localhost:8000/predict/visualize?model=maskrcnn" \
  -F "file=@spine_xray.jpg" \
  --output annotated.png
```

**Respuesta:**
- Content-Type: `image/png`
- Encabezados:
  - `X-Num-Detections`: Número de vértebras detectadas
  - `X-Processing-Time-Ms`: Tiempo de inferencia en milisegundos
  - `X-Model-Used`: Modelo usado para la inferencia

## Configuración de Modelos

Los modelos se configuran en [app/config.py](app/config.py) y pueden personalizarse mediante variables de entorno:

| Variable | Valor por defecto | Descripción |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `yolo` | Modelo predeterminado a usar (`yolo` o `maskrcnn`) |
| `YOLO_MODEL_KEY` | `yolo_best.pt` | Clave S3 para pesos del modelo YOLO |
| `MASKRCNN_MODEL_KEY` | `model_final.pth` | Clave S3 para pesos de Mask R-CNN |
| `MASKRCNN_BACKBONE` | `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` | Configuración Detectron2 |
| `CONFIDENCE_THRESHOLD` | `0.5` | Puntuación de confianza mínima para detecciones |
| `NMS_THRESHOLD` | `0.5` | Umbral de supresión no máxima |
| `MAX_DETECTIONS` | `100` | Número máximo de detecciones por imagen |
| `MODEL_CACHE_DIR` | `/tmp/model_cache` | Directorio para almacenar en caché modelos descargados |
| `DEVICE` | `cpu` | Dispositivo para inferencia (`cpu` o `cuda`) |

## Desarrollo

### Estructura del Proyecto

La API sigue una arquitectura modular:

- **[app/main.py](app/main.py)**: Aplicación FastAPI con definiciones de endpoints
- **[app/services.py](app/services.py)**: Capa de lógica de negocio que maneja predicciones y visualizaciones
- **[app/models/](app/models/)**: Implementaciones de modelos
  - **[base.py](app/models/base.py)**: Clase base abstracta para todos los modelos
  - **[factory.py](app/models/factory.py)**: Factoría de modelos para gestionar múltiples modelos
  - **[yolo.py](app/models/yolo.py)**: Implementación YOLOv8 usando Ultralytics
  - **[maskrcnn.py](app/models/maskrcnn.py)**: Implementación Mask R-CNN usando Detectron2
- **[app/utils.py](app/utils.py)**: Funciones de utilidad para S3, procesamiento de imágenes y codificación de máscaras
- **[app/schemas.py](app/schemas.py)**: Modelos Pydantic para validación de solicitudes/respuestas
- **[app/config.py](app/config.py)**: Gestión de configuración usando Pydantic settings


## Rendimiento

### YOLOv8
- **Inferencia CPU**: ~150-300ms por imagen (512x512)
- **Inferencia GPU**: ~20-50ms por imagen
- **Tamaño del Modelo**: ~50-100MB (depende de la variante)
- **Uso de Memoria**: ~1-2GB RAM

### Mask R-CNN
- **Inferencia CPU**: ~300-600ms por imagen (512x512)
- **Inferencia GPU**: ~50-100ms por imagen
- **Tamaño del Modelo**: ~170MB
- **Uso de Memoria**: ~2-3GB RAM

Para inferencia más rápida, configure `DEVICE=cuda` en `.env` y asegúrese de que CUDA esté disponible.

### Modelos Soportados

| Modelo | Framework | Fortalezas | Caso de Uso |
|-------|-----------|-----------|----------|
| **YOLOv8** | Ultralytics | Inferencia rápida, menor memoria | Aplicaciones en tiempo real, entornos con recursos limitados |
| **Mask R-CNN** | Detectron2 | Alta precisión, máscaras detalladas | Aplicaciones críticas en precisión, investigación |

### Diseño de la API
- **RESTful**: Métodos y códigos de estado HTTP estándar
- **Validación**: Esquemas Pydantic para validación de solicitudes/respuestas
- **CORS Habilitado**: Listo para integración con aplicaciones web
- **Documentación Interactiva**: Swagger UI y ReDoc disponibles
- **Manejo de Errores**: Mensajes de error completos y excepciones HTTP

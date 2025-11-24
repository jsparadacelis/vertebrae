# Frontend de Segmentación de Vértebras (React)

Cliente WEB para la API de Segmentación de Vértebras, construida con Vite y React.

## Características

- **React**
- **Carga por Arrastrar y Soltar**: Carga intuitiva de imágenes con retroalimentación visual
- **Selección de Modelo**: Cambia entre modelos YOLO y Mask R-CNN
  - **Analizar**: Obtén resultados detallados en JSON con datos de detección
  - **Visualizar**: Obtén imágenes anotadas con predicciones dibujadas
- **Diseño Responsivo**: Funciona perfectamente en escritorio y teléfonos.

## Stack Tecnológico

- **React 18** - Framework WEB
- **Vite** - Herramienta de construcción y servidor de desarrollo
- **Fetch API** - Cliente HTTP para llamadas a la API

## Prerequisitos

- Node.js 20.x o superior
- npm o yarn
- API de Segmentación de Vértebras en ejecución local (predeterminado: `http://localhost:8000`)

## Inicio Rápido

### Instalación

```bash
# Instalar dependencias
npm install

# Copiar variables de entorno
cp .env.example .env

# Editar .env si no se corre el api en localhost:8000
# VITE_API_URL=http://tu-host-api:puerto
```

### Desarrollo

```bash
# Iniciar servidor de desarrollo
npm run dev
```

La aplicación se abrirá en `http://localhost:5173`

### Compilación de Producción

```bash
# Compilar para producción
npm run build

# Previsualizar compilación de producción
npm run preview
```

## Estructura del Proyecto

```
vertebrae-frontend-react/
├── src/
│   ├── components/           # Componentes de React
│   │   ├── StatusIndicator.jsx       # Healthcheck de la api
│   │   ├── ModelSelector.jsx         # Menú desplegable de selección de modelo
│   │   ├── ModelInfoModal.jsx        # Modal de información del modelo
│   │   ├── ImageUpload.jsx           # Carga por arrastrar y soltar
│   │   ├── AnalysisActions.jsx       # Botones de análisis
│   │   ├── Results.jsx               # Visualización de resultados
│   │   └── *.css                     # Estilos de componentes
│   ├── services/            # Capa de servicio de API
│   │   └── api.js                    # Cliente de API
│   ├── App.jsx              # Componente principal de la aplicación
│   ├── App.css              # Estilos de la aplicación
│   ├── main.jsx             # Punto de entrada
│   └── index.css            # Estilos globales
├── .env                     # Variables de entorno
├── .env.example             # Plantilla de entorno
├── package.json             # Dependencias
├── vite.config.js           # Configuración de Vite
└── README.md                # Este archivo
```

## Uso

### 1. Iniciar la Aplicación

```bash
npm run dev
```

### 2. Verificar Estado de la API

El indicador de estado en la parte superior muestra:
- **Punto verde + "API Connected"**: La API está arriba
- **Punto rojo + "API Disconnected"**: No se puede comunicar con la API

### 3. Seleccionar Modelo

Elige entre:
- **YOLO**: Inferencia más rápida, bueno para análisis rápido
- **Mask R-CNN**: Mayor precisión, mejor segmentación

Haz clic en "Model Info" para especificaciones detalladas.

### 4. Cargar Imagen

**Arrastrar y Soltar**: Arrastra una imagen a la zona de carga

**Explorar**: Haz clic en "Browse Files" para seleccionar del sistema de archivos

Formatos soportados: JPEG, PNG

### 5. Ejecutar Análisis

**Analyze Image**: Puedes obtener la imágen con segmentaciónes y las métricas de confianza por cada vertebra

### 6. Ver Resultados

Los resultados muestran:
- Número de vértebras detectadas
- Tiempo de procesamiento
- Modelo utilizado
- Tabla detallada de todas las detecciones
- Imagen anotada (si se visualizó)

## Configuración

### Variables de Entorno

Crea un archivo `.env`:

```bash
VITE_API_URL=http://localhost:8000
```

## Compilación y Despliegue

### Compilación de Producción

```bash
npm run build
```

## Scripts

```bash
npm run dev       # Iniciar servidor de desarrollo
npm run build     # Compilar para producción
npm run preview   # Previsualizar compilación de producción
```
# Vertebrae Segmentation Frontend (React)

A modern React application for the Vertebrae Segmentation API, built with Vite for fast development and optimized production builds.

## Features

- **Modern React**: Built with React 18+ and Vite
- **Component-Based Architecture**: Modular, reusable components
- **Real-time API Health Monitoring**: Automatic health checks every 30 seconds
- **Drag-and-Drop Upload**: Intuitive image upload with visual feedback
- **Model Selection**: Switch between YOLO and Mask R-CNN models
- **Dual Analysis Modes**:
  - **Analyze**: Get detailed JSON results with detection data
  - **Visualize**: Get annotated images with drawn predictions
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Professional UI**: Clean, medical-grade interface with smooth animations

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **CSS Modules** - Component-scoped styling
- **Fetch API** - HTTP client for API calls

## Prerequisites

- Node.js 20.x or higher
- npm or yarn
- Vertebrae Segmentation API running (default: `http://localhost:8000`)

## Quick Start

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Edit .env if your API is not at localhost:8000
# VITE_API_URL=http://your-api-host:port
```

### Development

```bash
# Start development server
npm run dev
```

The app will open at `http://localhost:5173`

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
vertebrae-frontend-react/
├── src/
│   ├── components/           # React components
│   │   ├── StatusIndicator.jsx       # API health status
│   │   ├── ModelSelector.jsx         # Model selection dropdown
│   │   ├── ModelInfoModal.jsx        # Model info modal
│   │   ├── ImageUpload.jsx           # Drag-and-drop upload
│   │   ├── AnalysisActions.jsx       # Analysis buttons
│   │   ├── Results.jsx               # Results display
│   │   └── *.css                     # Component styles
│   ├── services/            # API service layer
│   │   └── api.js                    # API client
│   ├── App.jsx              # Main app component
│   ├── App.css              # App styles
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── .env                     # Environment variables
├── .env.example             # Environment template
├── package.json             # Dependencies
├── vite.config.js           # Vite configuration
└── README.md                # This file
```

## Usage

### 1. Start the Application

```bash
npm run dev
```

### 2. Check API Status

The status indicator at the top shows:
- **Green dot + "API Connected"**: API is healthy
- **Red dot + "API Disconnected"**: Cannot reach API

### 3. Select Model

Choose between:
- **YOLO**: Faster inference, good for quick analysis
- **Mask R-CNN**: Higher accuracy, better segmentation

Click "Model Info" for detailed specifications.

### 4. Upload Image

**Drag & Drop**: Drag an image onto the upload zone

**Browse**: Click "Browse Files" to select from file system

Supported formats: JPEG, PNG

### 5. Run Analysis

**Analyze Image**: Get detailed JSON results

**Get Annotated Image**: Get visual results with drawn predictions

### 6. View Results

Results show:
- Number of vertebrae detected
- Processing time
- Model used
- Detailed table of all detections
- Annotated image (if visualized)

## Configuration

### Environment Variables

Create a `.env` file:

```bash
VITE_API_URL=http://localhost:8000
```

## Build and Deployment

### Production Build

```bash
npm run build
```

Creates optimized build in `dist/` directory.

### Deploy to Netlify

```bash
npm install -g netlify-cli
npm run build
netlify deploy --prod --dir=dist
```

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

## Troubleshooting

### API Connection Issues

1. Ensure API is running
2. Check `.env` file has correct `VITE_API_URL`
3. Verify CORS is enabled in the API

### CORS Errors

Ensure FastAPI has CORS middleware:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Scripts

```bash
npm run dev       # Start development server
npm run build     # Build for production
npm run preview   # Preview production build
```

## License

Part of the Vertebrae Segmentation project.

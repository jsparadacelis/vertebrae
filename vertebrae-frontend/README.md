# Vertebrae Segmentation Frontend

A modern, responsive web interface for the Vertebrae Segmentation API. This application provides an intuitive UI for uploading medical images and performing vertebrae segmentation using YOLO or Mask R-CNN models.

## Features

- **Drag-and-Drop Upload**: Easy image upload with drag-and-drop support
- **Model Selection**: Choose between YOLO (faster) and Mask R-CNN (more accurate)
- **Real-time API Status**: Visual indicator showing API connection status
- **Dual Analysis Modes**:
  - **Analyze**: Get detailed JSON results with bounding boxes, masks, and confidence scores
  - **Visualize**: Get annotated images with drawn predictions
- **Detailed Results Display**:
  - Summary statistics (number of detections, processing time, model used)
  - Interactive table with all detected vertebrae
  - Confidence score badges with color coding
  - Bounding box coordinates
- **Model Information**: View detailed information about each model
- **Download Support**: Download annotated images

## Screenshots

### Main Interface
![Main Interface](docs/screenshot-main.png)

### Results View
![Results](docs/screenshot-results.png)

## Prerequisites

- A modern web browser (Chrome, Firefox, Safari, Edge)
- The Vertebrae Segmentation API running (default: `http://localhost:8000`)

## Quick Start

### Option 1: Simple HTTP Server (Recommended for Development)

Using Python:
```bash
cd vertebrae-frontend
python3 -m http.server 8080
```

Then open your browser to: `http://localhost:8080`

Using Node.js:
```bash
cd vertebrae-frontend
npx http-server -p 8080
```

### Option 2: Open Directly

Simply open `index.html` in your web browser. Note: Some browsers may restrict certain features when opening files directly.

### Option 3: Using VS Code Live Server

1. Install the "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

## Configuration

### API Endpoint

By default, the frontend connects to `http://localhost:8000`. To change this, edit the `API_BASE_URL` constant in [app.js](app.js):

```javascript
const API_BASE_URL = 'http://your-api-host:port';
```

## Usage Guide

### 1. Check API Status

When the page loads, the status indicator at the top will show:
- **Green dot + "API Connected"**: API is healthy and ready
- **Red dot + "API Disconnected"**: Cannot reach the API

### 2. Select Model

Choose between two models:
- **YOLO**: Faster inference, good for quick analysis
- **Mask R-CNN**: Higher accuracy, better segmentation quality

Click "Model Info" to view detailed model specifications.

### 3. Upload Image

**Method A: Drag and Drop**
- Drag an image file into the upload zone
- The zone will highlight when you hover over it

**Method B: Browse**
- Click "Browse Files" button
- Select an image from your file system

Supported formats: JPEG, PNG

### 4. Run Analysis

**Option A: Get JSON Results**
- Click "Analyze Image"
- View detailed results table with:
  - Vertebra class (T1-T12, L1-L5)
  - Confidence score (color-coded)
  - Bounding box coordinates
  - Segmentation masks (RLE format)

**Option B: Get Annotated Image**
- Click "Get Annotated Image"
- View the image with drawn predictions
- Download the annotated image using the "Download Image" button

You can run both analyses on the same image.

### 5. Clear and Restart

Click "Clear Image" to remove the current image and upload a new one.

## Project Structure

```
vertebrae-frontend/
├── index.html          # Main HTML structure
├── styles.css          # Styling and layout
├── app.js             # Application logic and API integration
└── README.md          # This file
```

## API Integration

The frontend integrates with the following API endpoints:

### GET `/health`
Check API health status

### GET `/model-info?model={yolo|maskrcnn}`
Get detailed model information

### POST `/predict?model={yolo|maskrcnn}`
Run segmentation and get JSON results

**Request**: Multipart form data with image file

**Response**:
```json
{
  "detections": [
    {
      "bbox": {"x1": 100, "y1": 200, "x2": 150, "y2": 250},
      "mask": {"size": [512, 512], "counts": "..."},
      "score": 0.95,
      "class_name": "T1",
      "class_id": 0
    }
  ],
  "num_detections": 17,
  "image_shape": [512, 512, 3],
  "processing_time_ms": 234.5,
  "model_used": "yolo"
}
```

### POST `/predict/visualize?model={yolo|maskrcnn}`
Run segmentation and get annotated image

**Request**: Multipart form data with image file

**Response**: PNG image with metadata in headers:
- `X-Num-Detections`: Number of vertebrae detected
- `X-Processing-Time-Ms`: Processing time in milliseconds
- `X-Model-Used`: Model used for inference

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

### API Connection Issues

**Problem**: "API Disconnected" status

**Solutions**:
1. Ensure the API is running: `docker-compose up` or `uvicorn app.main:app`
2. Check the API URL in `app.js` matches your setup
3. Verify CORS is enabled in the API
4. Check browser console for CORS or network errors

### Upload Not Working

**Problem**: Cannot upload images

**Solutions**:
1. Verify file is a valid image format (JPEG/PNG)
2. Check file size (very large images may take time)
3. Check browser console for errors

### Results Not Displaying

**Problem**: Analysis completes but no results shown

**Solutions**:
1. Check browser console for JavaScript errors
2. Verify API response format matches expected schema
3. Try refreshing the page

### CORS Errors

**Problem**: Browser console shows CORS policy errors

**Solution**: Ensure CORS middleware is configured in the FastAPI backend:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Development

### Customization

**Colors and Styling**
Edit CSS variables in [styles.css](styles.css):
```css
:root {
    --primary-color: #2563eb;
    --success-color: #10b981;
    --danger-color: #ef4444;
    /* ... */
}
```

**API Configuration**
Modify `API_BASE_URL` in [app.js](app.js)

**UI Components**
Edit HTML structure in [index.html](index.html)

### Adding Features

The code is modular and well-commented. Common modifications:

1. **Add new visualization options**: Extend `visualizeImage()` function
2. **Add filters/preprocessing**: Modify `handleFile()` function
3. **Add export formats**: Extend `downloadAnnotatedImage()` function
4. **Add batch processing**: Create new upload handler for multiple files

## Production Deployment

### Using Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /path/to/vertebrae-frontend;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Environment-Specific Configuration

Create a `config.js` file:

```javascript
const config = {
    development: {
        API_BASE_URL: 'http://localhost:8000'
    },
    production: {
        API_BASE_URL: 'https://api.your-domain.com'
    }
};

const API_BASE_URL = config[ENV] || config.development;
```

## Security Considerations

For production deployments:

1. **HTTPS**: Always use HTTPS in production
2. **CORS**: Configure specific allowed origins instead of `"*"`
3. **File Size Limits**: Implement client-side file size validation
4. **Content Security Policy**: Add CSP headers
5. **Authentication**: Implement user authentication if needed

## Performance Tips

1. **Image Optimization**: Resize very large images before upload
2. **Caching**: Enable browser caching for static assets
3. **CDN**: Serve static files via CDN for production
4. **Compression**: Enable gzip/brotli compression

## License

This frontend application is part of the Vertebrae Segmentation project.

## Support

For issues, questions, or contributions:
- Check the API documentation
- Review browser console for errors
- Ensure API and frontend versions are compatible

## Changelog

### Version 1.0.0 (2025-01-22)
- Initial release
- Drag-and-drop upload
- YOLO and Mask R-CNN model support
- JSON and visualization modes
- Model information modal
- Responsive design
- Real-time API health check

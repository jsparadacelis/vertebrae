import { useState } from 'react';
import './ImageUpload.css';

export default function ImageUpload({ onFileSelect, selectedFile, onClear, annotatedImage, originalImageUrl, onDownload }) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFileInput = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (JPEG, PNG, etc.)');
      return;
    }

    onFileSelect(file);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewUrl(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  const handleClear = () => {
    setPreviewUrl(null);
    onClear();
  };

  const handleBrowseClick = (e) => {
    e.stopPropagation();
    document.getElementById('fileInput').click();
  };

  return (
    <div className="card">
      <h2>{annotatedImage ? 'Results' : 'Upload Image'}</h2>

      {!previewUrl && !annotatedImage ? (
        <div
          className={`upload-zone ${isDragging ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="fileInput"
            accept="image/*"
            hidden
            onChange={handleFileInput}
          />
          <div className="upload-content">
            <svg
              className="upload-icon"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="upload-text">Drag and drop an image here</p>
            <p className="upload-subtext">or</p>
            <button className="btn-primary" onClick={handleBrowseClick}>
              Browse Files
            </button>
            <p className="upload-hint">Supported formats: JPEG, PNG</p>
          </div>
        </div>
      ) : annotatedImage ? (
        <div className="images-comparison-upload">
          <div className="images-grid">
            <div className="image-container">
              <h4>Original Image</h4>
              <img
                src={originalImageUrl}
                alt="Original"
                className="comparison-image"
              />
            </div>
            <div className="image-container">
              <h4>Annotated Image</h4>
              <img
                src={annotatedImage.imageUrl}
                alt="Annotated"
                className="comparison-image"
              />
            </div>
          </div>
          <div className="action-buttons">
            <button className="btn-secondary" onClick={onDownload}>
              Download Annotated Image
            </button>
            <button className="btn-danger" onClick={handleClear}>
              Clear & Upload New Image
            </button>
          </div>
        </div>
      ) : (
        <div className="preview-section">
          <h3>Original Image</h3>
          <img src={previewUrl} alt="Preview" className="preview-image" />
          <button className="btn-danger" onClick={handleClear}>
            Clear Image
          </button>
        </div>
      )}
    </div>
  );
}

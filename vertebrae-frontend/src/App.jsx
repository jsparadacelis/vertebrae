import { useState } from 'react';
import StatusIndicator from './components/StatusIndicator';
import ModelSelector from './components/ModelSelector';
import ImageUpload from './components/ImageUpload';
import AnalysisActions from './components/AnalysisActions';
import Results from './components/Results';
import api from './services/api';
import './App.css';

function App() {
  const [selectedModel, setSelectedModel] = useState('yolo');
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImageUrl, setOriginalImageUrl] = useState(null);
  const [visualizing, setVisualizing] = useState(false);
  const [results, setResults] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
    setAnnotatedImage(null);

    // Create preview URL for original image
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImageUrl(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  const handleClearImage = () => {
    setSelectedFile(null);
    setOriginalImageUrl(null);
    setResults(null);
    setAnnotatedImage(null);
  };

  const handleVisualize = async () => {
    if (!selectedFile) return;

    setVisualizing(true);
    try {
      // Run both requests in parallel
      const [visualData, detectionData] = await Promise.all([
        api.predictVisualize(selectedFile, selectedModel),
        api.predict(selectedFile, selectedModel),
      ]);

      // Display both results at the same time
      setAnnotatedImage(visualData);
      setResults(detectionData);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert(`Analysis failed: ${error.message}`);
    } finally {
      setVisualizing(false);
    }
  };

  const handleDownload = () => {
    if (!annotatedImage) return;

    const link = document.createElement('a');
    link.href = annotatedImage.imageUrl;
    link.download = `vertebrae_segmentation_${Date.now()}.png`;
    link.click();
  };

  return (
    <div className="app">
      <div className="container">
        <header>
          <div className="header-content">
            <img src="/logo.png" alt="Logo" className="header-logo" />
            <div className="header-text">
              <h1>Vertebrae Segmentation Tool</h1>
              <p className="subtitle">
                AI-powered medical image analysis for T1-T12 and L1-L5 vertebrae
              </p>
            </div>
          </div>
        </header>

        <main>
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />
          <ImageUpload
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onClear={handleClearImage}
            annotatedImage={annotatedImage}
            originalImageUrl={originalImageUrl}
            onDownload={handleDownload}
          />

          {selectedFile && !annotatedImage && (
            <AnalysisActions
              onVisualize={handleVisualize}
              disabled={!selectedFile}
              visualizing={visualizing}
            />
          )}

          {results?.detections?.length > 0 && (
            <Results
              results={results}
              annotatedImage={null}
              originalImageUrl={originalImageUrl}
              onDownload={handleDownload}
            />
          )}
        </main>

        <footer>
          <StatusIndicator />
          <p>Vertebrae Segmentation API v0.2.0</p>
        </footer>
      </div>
    </div>
  );
}

export default App;

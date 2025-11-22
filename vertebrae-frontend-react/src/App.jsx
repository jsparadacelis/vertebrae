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
  const [analyzing, setAnalyzing] = useState(false);
  const [visualizing, setVisualizing] = useState(false);
  const [results, setResults] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
    setAnnotatedImage(null);
  };

  const handleClearImage = () => {
    setSelectedFile(null);
    setResults(null);
    setAnnotatedImage(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    try {
      const data = await api.predict(selectedFile, selectedModel);
      setResults(data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert(`Analysis failed: ${error.message}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleVisualize = async () => {
    if (!selectedFile) return;

    setVisualizing(true);
    try {
      const data = await api.predictVisualize(selectedFile, selectedModel);
      setAnnotatedImage(data);

      // Also set minimal results if not already set
      if (!results) {
        setResults({
          num_detections: parseInt(data.metadata.numDetections) || 0,
          processing_time_ms: parseFloat(data.metadata.processingTime) || 0,
          model_used: data.metadata.modelUsed || selectedModel,
          detections: [],
        });
      }
    } catch (error) {
      console.error('Visualization failed:', error);
      alert(`Visualization failed: ${error.message}`);
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
          <h1>Vertebrae Segmentation Tool</h1>
          <p className="subtitle">
            AI-powered medical image analysis for T1-T12 and L1-L5 vertebrae
          </p>
        </header>

        <main>
          <StatusIndicator />
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />
          <ImageUpload
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onClear={handleClearImage}
          />

          {selectedFile && (
            <AnalysisActions
              onAnalyze={handleAnalyze}
              onVisualize={handleVisualize}
              disabled={!selectedFile}
              analyzing={analyzing}
              visualizing={visualizing}
            />
          )}

          {(results || annotatedImage) && (
            <Results
              results={results?.detections?.length > 0 ? results : null}
              annotatedImage={annotatedImage}
              onDownload={handleDownload}
            />
          )}
        </main>

        <footer>
          <p>Vertebrae Segmentation API v0.2.0</p>
        </footer>
      </div>
    </div>
  );
}

export default App;

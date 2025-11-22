import { useState } from 'react';
import api from '../services/api';
import ModelInfoModal from './ModelInfoModal';
import './ModelSelector.css';

export default function ModelSelector({ selectedModel, onModelChange }) {
  const [showModal, setShowModal] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleShowInfo = async () => {
    setLoading(true);
    try {
      const info = await api.getModelInfo(selectedModel);
      setModelInfo(info);
      setShowModal(true);
    } catch (error) {
      console.error('Failed to fetch model info:', error);
      alert('Failed to load model information');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="card">
        <h2>Configuration</h2>
        <div className="form-group">
          <label htmlFor="modelSelect">Select Model:</label>
          <select
            id="modelSelect"
            className="model-select"
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
          >
            <option value="yolo">YOLO</option>
            <option value="maskrcnn">Mask R-CNN</option>
          </select>
          <button
            className="btn-secondary"
            onClick={handleShowInfo}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Model Info'}
          </button>
        </div>
      </div>

      {showModal && modelInfo && (
        <ModelInfoModal
          modelInfo={modelInfo}
          onClose={() => setShowModal(false)}
        />
      )}
    </>
  );
}

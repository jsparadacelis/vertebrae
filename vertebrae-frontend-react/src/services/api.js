const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class VertebraeAPI {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async checkHealth() {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return await response.json();
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  async getModelInfo(model = null) {
    try {
      const url = model
        ? `${this.baseURL}/model-info?model=${model}`
        : `${this.baseURL}/model-info`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to fetch model info: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Get model info failed: ${error.message}`);
    }
  }

  async getAllModelsInfo() {
    try {
      const response = await fetch(`${this.baseURL}/models`);

      if (!response.ok) {
        throw new Error(`Failed to fetch models info: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Get models info failed: ${error.message}`);
    }
  }

  async predict(file, model = null) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const url = model
        ? `${this.baseURL}/predict?model=${model}`
        : `${this.baseURL}/predict`;

      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }

  async predictVisualize(file, model = null) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const url = model
        ? `${this.baseURL}/predict/visualize?model=${model}`
        : `${this.baseURL}/predict/visualize`;

      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Visualization failed');
      }

      // Get metadata from headers
      const metadata = {
        numDetections: response.headers.get('X-Num-Detections'),
        processingTime: response.headers.get('X-Processing-Time-Ms'),
        modelUsed: response.headers.get('X-Model-Used'),
      };

      // Get image blob
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);

      return {
        imageUrl,
        blob,
        metadata,
      };
    } catch (error) {
      throw new Error(`Visualization failed: ${error.message}`);
    }
  }
}

export default new VertebraeAPI();

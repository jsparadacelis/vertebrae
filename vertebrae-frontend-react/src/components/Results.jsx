import './Results.css';

export default function Results({ results, annotatedImage, originalImageUrl, onDownload }) {
  if (!results && !annotatedImage) return null;

  const getConfidenceClass = (score) => {
    if (score >= 0.8) return 'confidence-high';
    if (score >= 0.6) return 'confidence-medium';
    return 'confidence-low';
  };

  return (
    <div className="card">
      <h2>Results</h2>

      {results && (
        <>
          <div className="results-meta">
            <div className="meta-item">
              <span className="meta-label">Detections:</span>
              <span className="meta-value">{results.num_detections}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Processing Time:</span>
              <span className="meta-value">
                {results.processing_time_ms.toFixed(2)} ms
              </span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Model Used:</span>
              <span className="meta-value">{results.model_used.toUpperCase()}</span>
            </div>
          </div>

          <div className="table-container">
            <table className="detections-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Vertebra</th>
                  <th>Confidence</th>
                  <th>Bounding Box</th>
                </tr>
              </thead>
              <tbody>
                {results.detections.map((detection, index) => (
                  <tr key={index}>
                    <td>{index + 1}</td>
                    <td>
                      <strong>{detection.class_name}</strong>
                    </td>
                    <td>
                      <span
                        className={`confidence-badge ${getConfidenceClass(
                          detection.score
                        )}`}
                      >
                        {(detection.score * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="bbox-coords">
                      ({Math.round(detection.bbox.x1)},{' '}
                      {Math.round(detection.bbox.y1)}) - (
                      {Math.round(detection.bbox.x2)},{' '}
                      {Math.round(detection.bbox.y2)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {annotatedImage && (
        <div className="images-comparison">
          <h3>Image Comparison</h3>
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
          <button className="btn-secondary" onClick={onDownload}>
            Download Annotated Image
          </button>
        </div>
      )}
    </div>
  );
}

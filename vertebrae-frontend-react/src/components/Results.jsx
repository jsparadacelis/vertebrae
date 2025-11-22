import './Results.css';

export default function Results({ results, annotatedImage, originalImageUrl, onDownload }) {
  if (!results && !annotatedImage) return null;

  const getConfidenceClass = (score) => {
    if (score >= 0.8) return 'confidence-high';
    if (score >= 0.6) return 'confidence-medium';
    return 'confidence-low';
  };

  // Sort detections by vertebra name in anatomical order (T1-T12, L1-L5)
  const sortedDetections = results?.detections ? [...results.detections].sort((a, b) => {
    const parseVertebra = (name) => {
      const match = name.match(/([TL])(\d+)/);
      if (!match) return { type: 'Z', num: 0 }; // Handle unexpected format
      return { type: match[1], num: parseInt(match[2]) };
    };

    const aVert = parseVertebra(a.class_name);
    const bVert = parseVertebra(b.class_name);

    // Sort T before L
    if (aVert.type !== bVert.type) {
      return aVert.type === 'T' ? -1 : 1;
    }

    // Sort by number within same type
    return aVert.num - bVert.num;
  }) : [];

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
                {sortedDetections.map((detection, index) => (
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

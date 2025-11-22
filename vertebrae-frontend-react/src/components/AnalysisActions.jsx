import './AnalysisActions.css';

export default function AnalysisActions({ onAnalyze, onVisualize, disabled, analyzing, visualizing }) {
  return (
    <div className="card actions">
      <button
        className="btn-primary btn-large"
        onClick={onAnalyze}
        disabled={disabled || analyzing}
      >
        {analyzing ? (
          <>
            <span className="spinner"></span> Analyzing...
          </>
        ) : (
          'Analyze Image'
        )}
      </button>
      <button
        className="btn-secondary btn-large"
        onClick={onVisualize}
        disabled={disabled || visualizing}
      >
        {visualizing ? (
          <>
            <span className="spinner"></span> Processing...
          </>
        ) : (
          'Get Annotated Image'
        )}
      </button>
    </div>
  );
}

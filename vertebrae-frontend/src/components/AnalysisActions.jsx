import './AnalysisActions.css';

export default function AnalysisActions({ onVisualize, disabled, visualizing }) {
  return (
    <div className="card actions">
      <button
        className="btn-primary btn-large"
        onClick={onVisualize}
        disabled={disabled || visualizing}
      >
        {visualizing ? (
          <>
            <span className="spinner"></span> Analyzing...
          </>
        ) : (
          'Analyze Image'
        )}
      </button>
    </div>
  );
}

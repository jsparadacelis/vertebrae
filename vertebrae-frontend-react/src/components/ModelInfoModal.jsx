import './ModelInfoModal.css';

export default function ModelInfoModal({ modelInfo, onClose }) {
  return (
    <div className="modal" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <span className="close" onClick={onClose}>
          &times;
        </span>
        <h2>Model Information</h2>
        <div className="model-info-content">
          <dl>
            <dt>Model Name:</dt>
            <dd>{modelInfo.model_name}</dd>

            <dt>Framework:</dt>
            <dd>{modelInfo.framework}</dd>

            <dt>Backbone:</dt>
            <dd>{modelInfo.backbone}</dd>

            <dt>Device:</dt>
            <dd>{modelInfo.device.toUpperCase()}</dd>

            <dt>Number of Classes:</dt>
            <dd>{modelInfo.num_classes}</dd>

            <dt>Confidence Threshold:</dt>
            <dd>{modelInfo.confidence_threshold}</dd>

            <dt>NMS Threshold:</dt>
            <dd>{modelInfo.nms_threshold}</dd>

            <dt>Classes:</dt>
            <dd>{modelInfo.classes.join(', ')}</dd>
          </dl>
        </div>
      </div>
    </div>
  );
}

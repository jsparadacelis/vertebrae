// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let selectedFile = null;
let currentResults = null;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const clearBtn = document.getElementById('clearBtn');
const modelSelect = document.getElementById('modelSelect');
const modelInfoBtn = document.getElementById('modelInfoBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const visualizeBtn = document.getElementById('visualizeBtn');
const actionsCard = document.getElementById('actionsCard');
const resultsCard = document.getElementById('resultsCard');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const analyzeBtnText = document.getElementById('analyzeBtnText');
const visualizeBtnText = document.getElementById('visualizeBtnText');
const numDetections = document.getElementById('numDetections');
const processingTime = document.getElementById('processingTime');
const modelUsed = document.getElementById('modelUsed');
const detectionsTableBody = document.getElementById('detectionsTableBody');
const annotatedSection = document.getElementById('annotatedSection');
const annotatedImage = document.getElementById('annotatedImage');
const downloadBtn = document.getElementById('downloadBtn');
const modelInfoModal = document.getElementById('modelInfoModal');
const modelInfoContent = document.getElementById('modelInfoContent');
const closeModal = document.querySelector('.close');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    // Upload zone click
    uploadZone.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);

    // Clear button
    clearBtn.addEventListener('click', clearImage);

    // Model info button
    modelInfoBtn.addEventListener('click', showModelInfo);

    // Analysis buttons
    analyzeBtn.addEventListener('click', analyzeImage);
    visualizeBtn.addEventListener('click', visualizeImage);

    // Download button
    downloadBtn.addEventListener('click', downloadAnnotatedImage);

    // Modal close
    closeModal.addEventListener('click', () => {
        modelInfoModal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === modelInfoModal) {
            modelInfoModal.style.display = 'none';
        }
    });
}

// API Health Check
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            statusDot.classList.add('online');
            statusText.textContent = 'API Connected';
        } else {
            statusDot.classList.add('offline');
            statusText.textContent = 'API Unhealthy';
        }
    } catch (error) {
        statusDot.classList.add('offline');
        statusText.textContent = 'API Disconnected';
        console.error('Health check failed:', error);
    }
}

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// File Selection Handler
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle File
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPEG, PNG, etc.)');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        actionsCard.style.display = 'block';
        resultsCard.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Clear Image
function clearImage() {
    selectedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    actionsCard.style.display = 'none';
    resultsCard.style.display = 'none';
    annotatedSection.style.display = 'none';
}

// Show Model Info
async function showModelInfo() {
    const model = modelSelect.value;
    modelInfoModal.style.display = 'block';
    modelInfoContent.innerHTML = '<p>Loading...</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/model-info?model=${model}`);
        if (!response.ok) throw new Error('Failed to fetch model info');

        const data = await response.json();

        modelInfoContent.innerHTML = `
            <dl>
                <dt>Model Name:</dt>
                <dd>${data.model_name}</dd>

                <dt>Framework:</dt>
                <dd>${data.framework}</dd>

                <dt>Backbone:</dt>
                <dd>${data.backbone}</dd>

                <dt>Device:</dt>
                <dd>${data.device.toUpperCase()}</dd>

                <dt>Number of Classes:</dt>
                <dd>${data.num_classes}</dd>

                <dt>Confidence Threshold:</dt>
                <dd>${data.confidence_threshold}</dd>

                <dt>NMS Threshold:</dt>
                <dd>${data.nms_threshold}</dd>

                <dt>Classes:</dt>
                <dd>${data.classes.join(', ')}</dd>
            </dl>
        `;
    } catch (error) {
        console.error('Failed to fetch model info:', error);
        modelInfoContent.innerHTML = '<p style="color: var(--danger-color);">Failed to load model information.</p>';
    }
}

// Analyze Image
async function analyzeImage() {
    if (!selectedFile) return;

    const model = modelSelect.value;
    analyzeBtn.disabled = true;
    visualizeBtn.disabled = true;
    analyzeBtnText.innerHTML = '<span class="spinner"></span> Analyzing...';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_BASE_URL}/predict?model=${model}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const data = await response.json();
        currentResults = data;
        displayResults(data);

    } catch (error) {
        console.error('Analysis failed:', error);
        alert(`Analysis failed: ${error.message}`);
    } finally {
        analyzeBtn.disabled = false;
        visualizeBtn.disabled = false;
        analyzeBtnText.textContent = 'Analyze Image';
    }
}

// Visualize Image
async function visualizeImage() {
    if (!selectedFile) return;

    const model = modelSelect.value;
    analyzeBtn.disabled = true;
    visualizeBtn.disabled = true;
    visualizeBtnText.innerHTML = '<span class="spinner"></span> Processing...';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_BASE_URL}/predict/visualize?model=${model}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Visualization failed');
        }

        // Get metadata from headers
        const numDets = response.headers.get('X-Num-Detections');
        const procTime = response.headers.get('X-Processing-Time-Ms');
        const modelUsedValue = response.headers.get('X-Model-Used');

        // Get image blob
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        // Display annotated image
        annotatedImage.src = imageUrl;
        annotatedSection.style.display = 'block';
        resultsCard.style.display = 'block';

        // Show metadata
        numDetections.textContent = numDets || '-';
        processingTime.textContent = procTime ? `${parseFloat(procTime).toFixed(2)} ms` : '-';
        modelUsed.textContent = modelUsedValue ? modelUsedValue.toUpperCase() : '-';

        // Hide table if not already populated
        if (!currentResults) {
            document.querySelector('.table-container').style.display = 'none';
        }

    } catch (error) {
        console.error('Visualization failed:', error);
        alert(`Visualization failed: ${error.message}`);
    } finally {
        analyzeBtn.disabled = false;
        visualizeBtn.disabled = false;
        visualizeBtnText.textContent = 'Get Annotated Image';
    }
}

// Display Results
function displayResults(data) {
    resultsCard.style.display = 'block';
    annotatedSection.style.display = 'none';

    // Update metadata
    numDetections.textContent = data.num_detections;
    processingTime.textContent = `${data.processing_time_ms.toFixed(2)} ms`;
    modelUsed.textContent = data.model_used.toUpperCase();

    // Show table
    document.querySelector('.table-container').style.display = 'block';

    // Populate detections table
    detectionsTableBody.innerHTML = '';
    data.detections.forEach((detection, index) => {
        const row = document.createElement('tr');

        const confidenceClass =
            detection.score >= 0.8 ? 'confidence-high' :
            detection.score >= 0.6 ? 'confidence-medium' :
            'confidence-low';

        row.innerHTML = `
            <td>${index + 1}</td>
            <td><strong>${detection.class_name}</strong></td>
            <td>
                <span class="confidence-badge ${confidenceClass}">
                    ${(detection.score * 100).toFixed(1)}%
                </span>
            </td>
            <td class="bbox-coords">
                (${Math.round(detection.bbox.x1)}, ${Math.round(detection.bbox.y1)}) -
                (${Math.round(detection.bbox.x2)}, ${Math.round(detection.bbox.y2)})
            </td>
        `;

        detectionsTableBody.appendChild(row);
    });

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Download Annotated Image
function downloadAnnotatedImage() {
    const link = document.createElement('a');
    link.href = annotatedImage.src;
    link.download = `vertebrae_segmentation_${Date.now()}.png`;
    link.click();
}

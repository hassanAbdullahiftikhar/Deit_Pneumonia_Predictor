document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const pageSections = document.querySelectorAll('.page-section');

    // DOM Elements - Upload & Analysis
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const samples = document.querySelectorAll('.sample-thumb');

    // DOM Elements - Results View
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsView = document.getElementById('results-view');
    const statusCard = document.getElementById('status-card');
    const statusIcon = document.getElementById('status-icon');
    const predictionLabel = document.getElementById('prediction-label');
    const predictionDesc = document.getElementById('prediction-desc');
    const confidenceScore = document.getElementById('confidence-score');
    const originalPreview = document.getElementById('original-preview');
    const gradcamPreview = document.getElementById('gradcam-preview');
    const probNormalText = document.getElementById('prob-normal-text');
    const probNormalBar = document.getElementById('prob-normal-bar');
    const probPneumoniaText = document.getElementById('prob-pneumonia-text');
    const probPneumoniaBar = document.getElementById('prob-pneumonia-bar');

    let selectedFile = null;

    // --- Navigation Logic ---
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            const page = item.getAttribute('data-page');
            if (!page) return;

            e.preventDefault();

            // Update Nav UI
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            // Switch Pages
            pageSections.forEach(section => {
                section.classList.add('hidden');
                if (section.id === `${page}-page`) {
                    section.classList.remove('hidden');
                    section.classList.add('animate-in');
                }
            });
        });
    });

    // --- Upload Logic ---
    dropZone.addEventListener('click', () => imageInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Sample Image Selection
    samples.forEach(sample => {
        sample.addEventListener('click', async () => {
            try {
                const response = await fetch(sample.src);
                const blob = await response.blob();
                const file = new File([blob], "sample.jpg", { type: "image/jpeg" });
                handleFile(file);
            } catch (err) {
                console.error("Failed to load sample image", err);
            }
        });
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        selectedFile = file;
        analyzeBtn.disabled = false;

        // Preview original
        const reader = new FileReader();
        reader.onload = (e) => {
            originalPreview.src = e.target.result;
            // Clear previous results when new file is uploaded
            resultsView.classList.add('hidden');
            resultsPlaceholder.classList.remove('hidden');
        };
        reader.readAsDataURL(file);

        // Visual feedback on drop zone
        dropZone.innerHTML = `
            <div class="upload-icon"><i class="fas fa-check-circle"></i></div>
            <div class="upload-text">
                <p>${file.name}</p>
                <span>Image ready for analysis</span>
            </div>
        `;
    }

    // --- Analysis Logic ---
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI Loading State
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ANALYZING...';
        resultsPlaceholder.innerHTML = `
            <i class="fas fa-cog fa-spin" style="font-size: 3rem; margin-bottom: var(--spacing-md); color: var(--electric-violet);"></i>
            <p>Model is processing image patches...<br>Generating Grad-CAM attention maps.</p>
        `;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Analysis failed');

            const data = await response.json();
            displayResults(data);
        } catch (err) {
            console.error(err);
            alert('Error during analysis. Please check if the backend is running.');
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> RUN ANALYSIS';
        }
    });

    function displayResults(data) {
        // Reset Button
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> RUN ANALYSIS';

        // Toggle Views
        resultsPlaceholder.classList.add('hidden');
        resultsView.classList.remove('hidden');
        resultsView.classList.add('animate-in');

        // Update Text & Scores
        const isPneumonia = data.prediction === 'PNEUMONIA';
        predictionLabel.innerText = data.prediction;
        confidenceScore.innerText = `${(data.confidence * 100).toFixed(1)}%`;
        
        // Probabilities
        const probNormal = data.probabilities['NORMAL'] || 0;
        const probPneumonia = data.probabilities['PNEUMONIA'] || 0;

        probNormalText.innerText = `${(probNormal * 100).toFixed(1)}%`;
        probNormalBar.style.width = `${probNormal * 100}%`;
        probPneumoniaText.innerText = `${(probPneumonia * 100).toFixed(1)}%`;
        probPneumoniaBar.style.width = `${probPneumonia * 100}%`;

        // Style based on result
        if (isPneumonia) {
            statusCard.className = 'card prediction-card pneumonia';
            statusIcon.innerHTML = '<i class="fas fa-exclamation-triangle" style="color: var(--color-pneumonia);"></i>';
            predictionDesc.innerText = 'High probability of pulmonary infiltration detected.';
            confidenceScore.style.color = 'var(--color-pneumonia)';
        } else {
            statusCard.className = 'card prediction-card normal';
            statusIcon.innerHTML = '<i class="fas fa-check-circle" style="color: var(--color-normal);"></i>';
            predictionDesc.innerText = 'No signs of pulmonary infiltration detected.';
            confidenceScore.style.color = 'var(--color-normal)';
        }

        // Grad-CAM Image
        gradcamPreview.src = `data:image/png;base64,${data.gradcam}`;
    }
});

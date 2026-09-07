/**
 * SmartTrace — Image Preview & Batch Staging Controller
 * Handles interactive client-side image previews for single inspection
 * and multi-file staging galleries with individual item removal before execution.
 */

(function () {
    'use strict';

    // Standard pre-verified sample plates
    const SAMPLES = {
        'sample_1': {
            id: 'sample_1',
            name: 'Sample_Plate_01_Pitted.jpg',
            url: '/media/samples/sample_1.jpg',
            size: '104 KB',
            dimensions: '1600 × 256 px',
            tag: 'Defect: Pitted Surface',
            tagType: 'defect'
        },
        'sample_2': {
            id: 'sample_2',
            name: 'Sample_Plate_02_Clean.jpg',
            url: '/media/samples/sample_2.jpg',
            size: '19 KB',
            dimensions: '1600 × 256 px',
            tag: 'Compliant: Clean Surface',
            tagType: 'pass'
        },
        'sample_3': {
            id: 'sample_3',
            name: 'Sample_Plate_03_Scratch.jpg',
            url: '/media/samples/sample_3.jpg',
            size: '81 KB',
            dimensions: '1600 × 256 px',
            tag: 'Defect: Surface Scratch',
            tagType: 'defect'
        },
        'sample_4': {
            id: 'sample_4',
            name: 'Sample_Plate_04_Inclusion.jpg',
            url: '/media/samples/sample_4.jpg',
            size: '138 KB',
            dimensions: '1600 × 256 px',
            tag: 'Defect: Inclusion',
            tagType: 'defect'
        }
    };

    /**
     * Format bytes to readable size string (e.g. 1.4 MB, 320 KB)
     */
    function formatFileSize(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    /* ═══════════════════════════════════════════════════════
       1. SINGLE IMAGE INSPECTION PREVIEW
       ═══════════════════════════════════════════════════════ */
    function initSingleImagePreview() {
        const fileInput = document.getElementById('image-upload-input');
        const uploadZone = document.getElementById('upload-zone');
        const previewContainer = document.getElementById('single-image-preview');
        const inspectBtn = document.getElementById('inspect-btn');
        const stagedSampleInput = document.getElementById('staged-sample-id');

        if (!previewContainer) return;

        const previewImg = document.getElementById('preview-img');
        const fileNameEl = document.getElementById('preview-filename');
        const fileSizeEl = document.getElementById('preview-filesize');
        const fileDimEl = document.getElementById('preview-dimensions');
        const fileTagEl = document.getElementById('preview-tag');
        const sampleCards = document.querySelectorAll('[data-sample-choice]');

        function updateActiveSample(sampleId) {
            sampleCards.forEach(card => {
                if (card.dataset.sampleChoice === sampleId) {
                    card.classList.add('active-sample');
                } else {
                    card.classList.remove('active-sample');
                }
            });
        }

        function stageSample(sampleId) {
            const sample = SAMPLES[sampleId] || SAMPLES['sample_1'];
            if (!sample) return;

            if (stagedSampleInput) stagedSampleInput.value = sample.id;
            if (fileInput) fileInput.value = '';

            previewImg.src = sample.url;
            fileNameEl.textContent = sample.name;
            fileSizeEl.innerHTML = `<i class="bi bi-hdd me-1"></i> ${sample.size}`;
            fileDimEl.innerHTML = `<i class="bi bi-aspect-ratio me-1"></i> ${sample.dimensions}`;

            if (fileTagEl) {
                fileTagEl.textContent = sample.tag;
                fileTagEl.className = sample.tagType === 'pass' 
                    ? 'badge bg-success-subtle text-success border border-success-subtle px-2 py-1'
                    : 'badge bg-danger-subtle text-danger border border-danger-subtle px-2 py-1';
            }

            previewContainer.classList.remove('d-none');
            updateActiveSample(sampleId);

            if (inspectBtn) {
                inspectBtn.disabled = false;
                inspectBtn.innerHTML = `<i class="bi bi-play-circle-fill me-1"></i> Run Quality Inspection (${sample.name})`;
            }
        }

        function stageUserFile(file) {
            if (!file || !file.type.startsWith('image/')) {
                alert('Please select a valid image file (JPG, PNG, WebP).');
                return;
            }

            if (stagedSampleInput) stagedSampleInput.value = '';
            updateActiveSample(null);

            const objectUrl = URL.createObjectURL(file);
            previewImg.src = objectUrl;
            fileNameEl.textContent = file.name;
            fileSizeEl.innerHTML = `<i class="bi bi-hdd me-1"></i> ${formatFileSize(file.size)}`;
            fileDimEl.innerHTML = `<i class="bi bi-aspect-ratio me-1"></i> Calculating...`;

            if (fileTagEl) {
                fileTagEl.textContent = 'Custom Uploaded Plate';
                fileTagEl.className = 'badge bg-primary-subtle text-primary border border-primary-subtle px-2 py-1';
            }

            const img = new Image();
            img.onload = function () {
                fileDimEl.innerHTML = `<i class="bi bi-aspect-ratio me-1"></i> ${this.naturalWidth} × ${this.naturalHeight} px`;
            };
            img.src = objectUrl;

            previewContainer.classList.remove('d-none');

            if (inspectBtn) {
                inspectBtn.disabled = false;
                inspectBtn.innerHTML = `<i class="bi bi-play-circle-fill me-1"></i> Run Quality Inspection (${file.name})`;
            }
        }

        // Attach click handlers to sample preview selector cards
        sampleCards.forEach(card => {
            card.addEventListener('click', function (e) {
                e.preventDefault();
                const choice = this.dataset.sampleChoice;
                stageSample(choice);
            });
        });

        // Attach file input change
        if (fileInput) {
            fileInput.addEventListener('change', function () {
                if (this.files && this.files[0]) {
                    stageUserFile(this.files[0]);
                }
            });
        }

        // Drag and drop on uploadZone
        if (uploadZone) {
            ['dragenter', 'dragover'].forEach(evt => {
                uploadZone.addEventListener(evt, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    uploadZone.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(evt => {
                uploadZone.addEventListener(evt, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    uploadZone.classList.remove('dragover');
                }, false);
            });

            uploadZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                if (dt.files && dt.files.length > 0) {
                    if (fileInput) fileInput.files = dt.files;
                    stageUserFile(dt.files[0]);
                }
            });
        }

        // Initialize: If no inspection result is currently shown, pre-stage Sample 1 by default
        const resultPanel = document.querySelector('.result-details-panel');
        if (!resultPanel) {
            stageSample('sample_1');
        }
    }

    /* ═══════════════════════════════════════════════════════
       2. BATCH IMAGES STAGING GALLERY & PREVIEW
       ═══════════════════════════════════════════════════════ */
    function initBatchPreview() {
        const fileInput = document.getElementById('batch-upload-input');
        const uploadZone = document.getElementById('batch-upload-zone');
        const stagingArea = document.getElementById('batch-staging-area');
        const galleryGrid = document.getElementById('batch-gallery-grid');
        const countBadge = document.getElementById('batch-count-badge');
        const totalSizeBadge = document.getElementById('batch-total-size');
        const clearAllBtn = document.getElementById('batch-clear-all-btn');
        const addMoreBtn = document.getElementById('batch-add-more-btn');
        const submitBtn = document.getElementById('batch-submit-btn');
        const isSampleBatchInput = document.getElementById('is-sample-batch-input');
        const loadSampleBatchBtn = document.getElementById('load-sample-batch-btn');

        if (!stagingArea || !galleryGrid) return;

        let stagedFiles = [];
        let isSampleLotActive = false;

        function updateInputFiles() {
            if (window.DataTransfer && fileInput) {
                const dt = new DataTransfer();
                stagedFiles.forEach(file => dt.items.add(file));
                fileInput.files = dt.files;
            }
        }

        function renderGallery() {
            galleryGrid.innerHTML = '';

            if (stagedFiles.length === 0 && !isSampleLotActive) {
                stagingArea.classList.add('d-none');
                if (uploadZone) uploadZone.classList.remove('d-none');
                if (isSampleBatchInput) isSampleBatchInput.value = '0';
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i> Run Batch Inspection';
                }
                return;
            }

            stagingArea.classList.remove('d-none');

            if (isSampleLotActive) {
                // Render sample lot preview
                const sampleKeys = ['sample_1', 'sample_2', 'sample_3', 'sample_4'];
                if (countBadge) countBadge.textContent = '4 Sample Plates Queued';
                if (totalSizeBadge) totalSizeBadge.textContent = '342 KB';
                if (isSampleBatchInput) isSampleBatchInput.value = '1';
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i> Run Batch Inspection (4 Sample Plates)';
                }

                sampleKeys.forEach((key, index) => {
                    const sample = SAMPLES[key];
                    const card = document.createElement('div');
                    card.className = 'batch-preview-card';
                    card.innerHTML = `
                        <div class="batch-preview-thumb-wrap">
                            <img src="${sample.url}" alt="${sample.name}" class="batch-preview-thumb" loading="lazy">
                        </div>
                        <div class="batch-preview-info">
                            <div class="batch-preview-name" title="${sample.name}">${sample.name}</div>
                            <div class="batch-preview-meta">
                                <span class="badge-size">${sample.size}</span>
                                <span class="badge-index">#${index + 1}</span>
                            </div>
                        </div>
                    `;
                    galleryGrid.appendChild(card);
                });
                return;
            }

            // Custom uploaded files
            if (isSampleBatchInput) isSampleBatchInput.value = '0';
            let totalBytes = 0;
            stagedFiles.forEach(f => totalBytes += f.size);

            if (countBadge) countBadge.textContent = `${stagedFiles.length} plates queued`;
            if (totalSizeBadge) totalSizeBadge.textContent = formatFileSize(totalBytes);
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = `<i class="bi bi-play-fill me-1"></i> Run Batch Inspection (${stagedFiles.length} Plates)`;
            }

            stagedFiles.forEach((file, index) => {
                const card = document.createElement('div');
                card.className = 'batch-preview-card';
                const objectUrl = URL.createObjectURL(file);

                card.innerHTML = `
                    <div class="batch-preview-thumb-wrap">
                        <img src="${objectUrl}" alt="${file.name}" class="batch-preview-thumb" loading="lazy">
                        <button type="button" class="batch-remove-item-btn" data-index="${index}" title="Remove plate">
                            <i class="bi bi-x"></i>
                        </button>
                    </div>
                    <div class="batch-preview-info">
                        <div class="batch-preview-name" title="${file.name}">${file.name}</div>
                        <div class="batch-preview-meta">
                            <span class="badge-size">${formatFileSize(file.size)}</span>
                            <span class="badge-index">#${index + 1}</span>
                        </div>
                    </div>
                `;

                const removeBtn = card.querySelector('.batch-remove-item-btn');
                removeBtn.addEventListener('click', function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    stagedFiles = stagedFiles.filter((_, idx) => idx !== index);
                    updateInputFiles();
                    renderGallery();
                });

                galleryGrid.appendChild(card);
            });
        }

        // 1-Click Load & Preview Sample Lot
        if (loadSampleBatchBtn) {
            loadSampleBatchBtn.addEventListener('click', function (e) {
                e.preventDefault();
                isSampleLotActive = true;
                stagedFiles = [];
                if (fileInput) fileInput.value = '';
                renderGallery();
                // Scroll smoothly to staging area
                stagingArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            });
        }

        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', function (e) {
                e.preventDefault();
                isSampleLotActive = false;
                stagedFiles = [];
                if (fileInput) fileInput.value = '';
                updateInputFiles();
                renderGallery();
            });
        }

        if (addMoreBtn && fileInput) {
            addMoreBtn.addEventListener('click', function (e) {
                e.preventDefault();
                isSampleLotActive = false;
                fileInput.click();
            });
        }

        if (fileInput) {
            fileInput.addEventListener('change', function () {
                if (this.files && this.files.length > 0) {
                    isSampleLotActive = false;
                    const validImages = Array.from(this.files).filter(f => f.type.startsWith('image/'));
                    stagedFiles = stagedFiles.concat(validImages);
                    updateInputFiles();
                    renderGallery();
                }
            });
        }

        // Auto-load sample lot if no batches currently exist
        const emptyState = document.querySelector('.content-card .text-center h4');
        if (emptyState && emptyState.textContent.includes('No Lot Analysis')) {
            isSampleLotActive = true;
            renderGallery();
        }
    }

    /* ═══════════════════════════════════════════════════════
       3. THEME TOGGLE
       ═══════════════════════════════════════════════════════ */
    function initThemeToggle() {
        const toggleBtn = document.getElementById('theme-toggle-btn');
        const savedTheme = localStorage.getItem('smarttrace-theme') || 'light';

        function applyTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            document.documentElement.setAttribute('data-bs-theme', theme === 'dark' ? 'dark' : 'light');
            localStorage.setItem('smarttrace-theme', theme);

            if (toggleBtn) {
                const icon = toggleBtn.querySelector('i');
                const text = toggleBtn.querySelector('.theme-text');
                if (theme === 'dark') {
                    if (icon) icon.className = 'bi bi-sun-fill text-warning';
                    if (text) text.textContent = 'Light Mode';
                } else {
                    if (icon) icon.className = 'bi bi-moon-stars-fill text-secondary';
                    if (text) text.textContent = 'Dark Mode';
                }
            }

            window.dispatchEvent(new Event('resize'));
        }

        applyTheme(savedTheme);

        if (toggleBtn) {
            toggleBtn.addEventListener('click', function () {
                const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
                const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
                applyTheme(nextTheme);
            });
        }
    }

    document.addEventListener('DOMContentLoaded', function () {
        initThemeToggle();
        initSingleImagePreview();
        initBatchPreview();
    });

})();

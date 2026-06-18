/**
 * Smart Trace — Plotly Chart Helpers
 * Industrial-themed chart configurations
 */

// Dark theme layout base
const DARK_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
        family: 'Inter, sans-serif',
        color: '#8b95a8',
        size: 12,
    },
    margin: { t: 40, r: 20, b: 40, l: 40 },
    showlegend: true,
    legend: {
        font: { size: 11, color: '#8b95a8' },
        bgcolor: 'rgba(0,0,0,0)',
    },
};

// Color palette
const COLORS = {
    blue: '#00d4ff',
    purple: '#7c3aed',
    emerald: '#00e676',
    amber: '#ffb800',
    red: '#ff1744',
    cyan: '#06b6d4',
    palette: ['#00d4ff', '#7c3aed', '#00e676', '#ffb800', '#ff1744', '#06b6d4'],
};

const PLOTLY_CONFIG = {
    displayModeBar: false,
    responsive: true,
};

/**
 * Render a donut chart for defect or severity distribution.
 */
function renderDonutChart(elementId, labels, values, title) {
    const el = document.getElementById(elementId);
    if (!el || !labels || labels.length === 0) return;

    const data = [{
        type: 'pie',
        labels: labels,
        values: values,
        hole: 0.55,
        marker: {
            colors: COLORS.palette.slice(0, labels.length),
            line: { color: '#0a0e17', width: 2 },
        },
        textinfo: 'label+percent',
        textfont: { size: 11, color: '#e8edf5' },
        hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
    }];

    const layout = {
        ...DARK_LAYOUT,
        title: {
            text: title,
            font: { size: 14, color: '#e8edf5' },
            x: 0.5,
        },
        height: 320,
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}

/**
 * Render a horizontal bar chart.
 */
function renderBarChart(elementId, labels, values, title, color) {
    const el = document.getElementById(elementId);
    if (!el || !labels || labels.length === 0) return;

    const data = [{
        type: 'bar',
        x: values,
        y: labels,
        orientation: 'h',
        marker: {
            color: color || COLORS.blue,
            opacity: 0.85,
            line: { color: color || COLORS.blue, width: 1 },
        },
        hovertemplate: '<b>%{y}</b>: %{x}<extra></extra>',
    }];

    const layout = {
        ...DARK_LAYOUT,
        title: {
            text: title,
            font: { size: 14, color: '#e8edf5' },
            x: 0.5,
        },
        height: 320,
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.04)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
        },
        yaxis: {
            automargin: true,
        },
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}

/**
 * Render a quality/risk gauge chart.
 */
function renderGauge(elementId, value, title, maxVal) {
    const el = document.getElementById(elementId);
    if (!el) return;

    maxVal = maxVal || 100;

    let gaugeColor;
    if (title.toLowerCase().includes('risk')) {
        gaugeColor = value > 50 ? COLORS.red : value > 25 ? COLORS.amber : COLORS.emerald;
    } else {
        gaugeColor = value >= 80 ? COLORS.emerald : value >= 50 ? COLORS.amber : COLORS.red;
    }

    const data = [{
        type: 'indicator',
        mode: 'gauge+number',
        value: value,
        title: {
            text: title,
            font: { size: 14, color: '#e8edf5' },
        },
        number: {
            suffix: '%',
            font: { size: 32, color: '#e8edf5', family: 'Inter' },
        },
        gauge: {
            axis: {
                range: [0, maxVal],
                tickwidth: 1,
                tickcolor: '#5a6478',
                dtick: 25,
            },
            bar: { color: gaugeColor, thickness: 0.7 },
            bgcolor: 'rgba(255,255,255,0.04)',
            borderwidth: 0,
            steps: [
                { range: [0, maxVal * 0.33], color: 'rgba(255,255,255,0.02)' },
                { range: [maxVal * 0.33, maxVal * 0.66], color: 'rgba(255,255,255,0.04)' },
                { range: [maxVal * 0.66, maxVal], color: 'rgba(255,255,255,0.06)' },
            ],
        },
    }];

    const layout = {
        ...DARK_LAYOUT,
        height: 220,
        margin: { t: 60, r: 20, b: 10, l: 20 },
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}

/**
 * Show loading overlay.
 */
function showLoading(message) {
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay active';
        overlay.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">${message || 'Processing...'}</div>
        `;
        document.body.appendChild(overlay);
    } else {
        overlay.classList.add('active');
        const text = overlay.querySelector('.loading-text');
        if (text) text.textContent = message || 'Processing...';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.classList.remove('active');
}

/**
 * Attach form loading on submit.
 */
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form[data-loading]');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            showLoading(form.dataset.loading || 'Processing...');
        });
    });
});

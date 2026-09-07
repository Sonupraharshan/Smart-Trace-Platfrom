/**
 * SmartTrace — Plotly Chart Helpers
 * Clean enterprise-grade chart configurations with dual theme support.
 */

// Colors for enterprise manufacturing analytics
const COLORS = {
    blue: '#2563eb',
    blueSubtle: 'rgba(37, 99, 235, 0.15)',
    purple: '#7c3aed',
    emerald: '#059669',
    amber: '#d97706',
    red: '#dc2626',
    cyan: '#0284c7',
    palette: ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0284c7'],
};

/**
 * Returns dynamic theme layout configuration for Plotly
 */
function getChartTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Inter, sans-serif',
            color: isDark ? '#94a3b8' : '#475569',
            size: 11,
        },
        margin: { t: 36, r: 16, b: 36, l: 36 },
        showlegend: true,
        legend: {
            font: { size: 11, color: isDark ? '#cbd5e1' : '#334155' },
            bgcolor: 'rgba(0,0,0,0)',
        },
        gridColor: isDark ? 'rgba(255, 255, 255, 0.06)' : '#f1f5f9',
        textColor: isDark ? '#f8fafc' : '#0f172a',
        sliceBorderColor: isDark ? '#131b2e' : '#ffffff',
    };
}

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

    const theme = getChartTheme();

    const data = [{
        type: 'pie',
        labels: labels,
        values: values,
        hole: 0.58,
        marker: {
            colors: COLORS.palette.slice(0, labels.length),
            line: { color: theme.sliceBorderColor, width: 2 },
        },
        textinfo: 'label+percent',
        textfont: { size: 11, color: '#ffffff' },
        hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
    }];

    const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        margin: theme.margin,
        showlegend: false,
        title: {
            text: title,
            font: { size: 13, color: theme.textColor, family: 'Inter' },
            x: 0.05,
            y: 0.96,
        },
        height: 270,
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}

/**
 * Render a horizontal bar chart.
 */
function renderBarChart(elementId, labels, values, title, color) {
    const el = document.getElementById(elementId);
    if (!el || !labels || labels.length === 0) return;

    const theme = getChartTheme();
    const barColor = color || COLORS.blue;

    const data = [{
        type: 'bar',
        x: values,
        y: labels,
        orientation: 'h',
        marker: {
            color: barColor,
            opacity: 0.9,
            line: { color: barColor, width: 1 },
            cornerradius: 4,
        },
        hovertemplate: '<b>%{y}</b>: %{x}<extra></extra>',
    }];

    const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        margin: { t: 36, r: 20, b: 36, l: 80 },
        title: {
            text: title,
            font: { size: 13, color: theme.textColor, family: 'Inter' },
            x: 0.05,
            y: 0.96,
        },
        height: 270,
        xaxis: {
            gridcolor: theme.gridColor,
            zerolinecolor: theme.gridColor,
            tickfont: { color: theme.font.color },
        },
        yaxis: {
            automargin: true,
            tickfont: { color: theme.font.color },
        },
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}

/**
 * Render a quality or risk gauge chart.
 */
function renderGauge(elementId, value, title, maxVal) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const theme = getChartTheme();
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
            font: { size: 13, color: theme.textColor, family: 'Inter' },
        },
        number: {
            suffix: '%',
            font: { size: 28, color: theme.textColor, family: 'Inter' },
        },
        gauge: {
            axis: {
                range: [0, maxVal],
                tickwidth: 1,
                tickcolor: theme.font.color,
                dtick: 25,
            },
            bar: { color: gaugeColor, thickness: 0.65 },
            bgcolor: theme.gridColor,
            borderwidth: 0,
            steps: [
                { range: [0, maxVal * 0.33], color: 'rgba(0,0,0,0.02)' },
                { range: [maxVal * 0.33, maxVal * 0.66], color: 'rgba(0,0,0,0.04)' },
                { range: [maxVal * 0.66, maxVal], color: 'rgba(0,0,0,0.06)' },
            ],
        },
    }];

    const layout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        height: 200,
        margin: { t: 40, r: 24, b: 10, l: 24 },
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}

/**
 * Show loading overlay with message.
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
 * Form submission loading handler
 */
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form[data-loading]');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            showLoading(form.dataset.loading || 'Processing...');
        });
    });
});

/**
 * Vehicle Analytics Dashboard — Client-Side Application
 *
 * Handles:
 * - REST API communication
 * - WebSocket real-time updates
 * - Chart.js timeline visualization
 * - Dynamic DOM updates for events, stats, and visit table
 */

// ── State ────────────────────────────────────────────────
let ws = null;
let pollingInterval = null;
let timelineChart = null;
let events = [];
let visits = [];

// ── API Functions ────────────────────────────────────────

async function apiCall(endpoint, method = 'GET', body = null) {
    const options = { method };
    if (body) {
        if (body instanceof FormData) {
            options.body = body;
        } else {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(body);
        }
    }
    const response = await fetch(`/api/${endpoint}`, options);
    return response.json();
}

// ── Processing Control ───────────────────────────────────

async function startProcessing() {
    const btnProcess = document.getElementById('btnProcess');
    const btnStop = document.getElementById('btnStop');
    const progressContainer = document.getElementById('progressContainer');

    btnProcess.disabled = true;
    btnStop.disabled = false;

    setStatus('active', 'Processing...');
    progressContainer.style.display = 'flex';

    // Clear previous state
    events = [];
    visits = [];
    updateEventList();
    updateVisitTable();
    resetStats();

    try {
        const result = await apiCall('process', 'POST');
        console.log('Processing started:', result);

        // Start polling for progress
        startPolling();
    } catch (err) {
        console.error('Failed to start processing:', err);
        setStatus('idle', 'Error');
        btnProcess.disabled = false;
        btnStop.disabled = true;
    }
}

async function stopProcessing() {
    try {
        await apiCall('stop', 'POST');
        stopPolling();
        setStatus('idle', 'Stopped');
        document.getElementById('btnProcess').disabled = false;
        document.getElementById('btnStop').disabled = true;
    } catch (err) {
        console.error('Failed to stop processing:', err);
    }
}

async function uploadVideo(input) {
    if (!input.files || !input.files[0]) return;

    const formData = new FormData();
    formData.append('file', input.files[0]);

    try {
        const result = await apiCall('upload', 'POST', formData);
        console.log('Upload result:', result);
        alert(`Video uploaded: ${result.path}`);
    } catch (err) {
        console.error('Upload failed:', err);
        alert('Upload failed. Please try again.');
    }
}

// ── Polling ──────────────────────────────────────────────

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            // Get progress
            const progressData = await apiCall('progress');
            updateProgress(progressData);

            // Get latest frame
            const frameData = await apiCall('frame');
            if (frameData.frame) {
                showFrame(frameData.frame);
            }

            // Check if complete
            if (progressData.is_complete) {
                stopPolling();
                onProcessingComplete();
            }

            if (progressData.error) {
                stopPolling();
                setStatus('idle', 'Error');
                document.getElementById('btnProcess').disabled = false;
                document.getElementById('btnStop').disabled = true;
                alert(`Processing error: ${progressData.error}`);
            }
        } catch (err) {
            console.error('Polling error:', err);
        }
    }, 800);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// ── Progress Updates ─────────────────────────────────────

function updateProgress(data) {
    const progress = data.progress || {};
    const percent = progress.percent || 0;
    const current = progress.current_frame || 0;
    const total = progress.total_frames || 0;

    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = `${percent.toFixed(1)}%`;
}

function showFrame(base64Frame) {
    const img = document.getElementById('videoFrame');
    const placeholder = document.getElementById('videoPlaceholder');

    img.src = `data:image/jpeg;base64,${base64Frame}`;
    img.style.display = 'block';
    placeholder.style.display = 'none';
}

// ── Processing Complete ──────────────────────────────────

async function onProcessingComplete() {
    setStatus('complete', 'Complete');
    document.getElementById('btnProcess').disabled = false;
    document.getElementById('btnStop').disabled = true;

    // Load all results
    try {
        const eventsData = await apiCall('events');
        events = eventsData.events || [];
        updateEventList();

        const visitsData = await apiCall('visits');
        visits = visitsData.visits || [];
        updateVisitTable();

        const summaryData = await apiCall('summary');
        updateStats(summaryData.summary || {});

        updateTimelineChart();
        loadAnalytics();
    } catch (err) {
        console.error('Failed to load results:', err);
    }
}

// ── UI Update Functions ──────────────────────────────────

function setStatus(state, text) {
    const badge = document.getElementById('statusBadge');
    badge.className = `status-badge ${state}`;
    badge.querySelector('.status-text').textContent = text;
}

function resetStats() {
    document.getElementById('statEntries').textContent = '0';
    document.getElementById('statExits').textContent = '0';
    document.getElementById('statInside').textContent = '0';
    document.getElementById('statUnique').textContent = '0';
}

function updateStats(summary) {
    animateValue('statEntries', summary.total_entries || 0);
    animateValue('statExits', summary.total_exits || 0);
    animateValue('statInside', summary.vehicles_inside || 0);
    animateValue('statUnique', summary.unique_vehicles || 0);
}

function animateValue(elementId, endValue) {
    const el = document.getElementById(elementId);
    const start = parseInt(el.textContent) || 0;
    const duration = 600;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.round(start + (endValue - start) * eased);
        el.textContent = current;
        if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
}

function updateEventList() {
    const list = document.getElementById('eventList');
    const count = document.getElementById('eventCount');

    count.textContent = `${events.length} events`;

    if (events.length === 0) {
        list.innerHTML = '<div class="event-empty">No events detected yet</div>';
        return;
    }

    list.innerHTML = events.map((event, index) => {
        const type = event.event.toLowerCase();
        const time = event.timestamp ? event.timestamp.split('T')[1] || event.timestamp : '';
        return `
            <div class="event-item ${type}" style="animation-delay: ${index * 50}ms">
                <span class="event-badge ${type}">${event.event}</span>
                <div class="event-details">
                    <div class="event-plate">${event.vehicle_number}</div>
                    <div class="event-time">${time} · ${event.camera}</div>
                </div>
            </div>
        `;
    }).reverse().join('');  // Most recent first
}

function updateVisitTable() {
    const tbody = document.getElementById('visitTableBody');

    if (visits.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="table-empty">No visit records yet</td></tr>';
        return;
    }

    tbody.innerHTML = visits.map(visit => {
        const statusClass = visit.status === 'Inside' ? 'inside' : 'completed';
        return `
            <tr>
                <td><strong style="font-family: 'Courier New', monospace;">${visit.vehicle_number}</strong></td>
                <td>${visit.entry_time || '—'}</td>
                <td>${visit.exit_time || '—'}</td>
                <td>${visit.duration || '—'}</td>
                <td>${visit.visit_no}</td>
                <td><span class="status-tag ${statusClass}">${visit.status}</span></td>
            </tr>
        `;
    }).join('');
}

// ── Chart.js Timeline ────────────────────────────────────

function updateTimelineChart() {
    const ctx = document.getElementById('timelineChart').getContext('2d');

    if (timelineChart) {
        timelineChart.destroy();
    }

    if (events.length === 0) return;

    // Group events by time (minute buckets)
    const entryByTime = {};
    const exitByTime = {};

    events.forEach(event => {
        const time = event.timestamp ? event.timestamp.split('T')[1] || '00:00' : '00:00';
        const minute = time.substring(0, 5); // HH:MM

        if (event.event === 'ENTRY') {
            entryByTime[minute] = (entryByTime[minute] || 0) + 1;
        } else {
            exitByTime[minute] = (exitByTime[minute] || 0) + 1;
        }
    });

    const allTimes = [...new Set([...Object.keys(entryByTime), ...Object.keys(exitByTime)])].sort();

    timelineChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: allTimes,
            datasets: [
                {
                    label: 'Entries',
                    data: allTimes.map(t => entryByTime[t] || 0),
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: '#10b981',
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'Exits',
                    data: allTimes.map(t => exitByTime[t] || 0),
                    backgroundColor: 'rgba(239, 68, 68, 0.6)',
                    borderColor: '#ef4444',
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#8896ab', font: { family: 'Inter' } },
                },
            },
            scales: {
                x: {
                    ticks: { color: '#5a6a80', font: { size: 11 } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#5a6a80',
                        stepSize: 1,
                        font: { size: 11 },
                    },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
            },
        },
    });
}

// ── AI Analytics ─────────────────────────────────────────

async function loadAnalytics() {
    const container = document.getElementById('aiReport');

    try {
        const data = await apiCall('analytics');
        if (data.report) {
            // Simple markdown to HTML conversion
            container.innerHTML = markdownToHtml(data.report);
        } else {
            container.innerHTML = '<p class="report-placeholder">AI analytics report will be generated after processing completes.</p>';
        }
    } catch (err) {
        console.error('Failed to load analytics:', err);
    }
}

function markdownToHtml(md) {
    // Basic markdown to HTML conversion
    let html = md
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^- (.*$)/gm, '<li>$1</li>')
        .replace(/^(\d+)\. (.*$)/gm, '<li>$2</li>')
        .replace(/\n/g, '<br>');

    // Wrap consecutive <li> elements in <ul>
    html = html.replace(/(<li>.*?<\/li>(<br>)?)+/g, match => {
        return '<ul>' + match.replace(/<br>/g, '') + '</ul>';
    });

    return html;
}

// ── Initialize ───────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Check initial status
    apiCall('status').then(data => {
        if (data.has_results) {
            onProcessingComplete();
        }
    });

    // Initialize empty chart
    updateTimelineChart();
});

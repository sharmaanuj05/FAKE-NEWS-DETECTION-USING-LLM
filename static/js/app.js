/**
 * VERITAS AI — Fake News Forensics Web Application Logic
 * Interactive Attention Heatmaps, Fallacy Detection, and Telemetry
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const inputTitle = document.getElementById('inputTitle');
  const inputText = document.getElementById('inputText');
  const charCounter = document.getElementById('charCounter');
  const wordCounter = document.getElementById('wordCounter');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const btnClear = document.getElementById('btnClear');
  const btnExport = document.getElementById('btnExport');
  const analyzeSpinner = document.getElementById('analyzeSpinner');
  const analyzeBtnText = document.getElementById('analyzeBtnText');
  const presetsContainer = document.getElementById('presetsContainer');
  const resultsSection = document.getElementById('resultsSection');

  // Benchmark Elements
  const btnRunBenchmark = document.getElementById('btnRunBenchmark');
  const benchmarkSpinner = document.getElementById('benchmarkSpinner');

  let currentAnalysisData = null;
  let cachedSamples = [];

  // ==========================================
  // Tab Switching
  // ==========================================
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');

      if (targetId === 'tab-benchmark') {
        loadBenchmarkMetrics();
      }
    });
  });

  // ==========================================
  // Character & Word Counter
  // ==========================================
  function updateCounters() {
    const text = inputText.value || '';
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    charCounter.textContent = `${chars} chars`;
    wordCounter.textContent = `${words} words`;
  }
  inputText.addEventListener('input', updateCounters);

  // Clear Form
  btnClear.addEventListener('click', () => {
    inputTitle.value = '';
    inputText.value = '';
    updateCounters();
    resultsSection.style.display = 'none';
    currentAnalysisData = null;
  });

  // ==========================================
  // Load Preset Samples
  // ==========================================
  async function fetchSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      cachedSamples = data.samples || [];
      renderPresets(cachedSamples);
    } catch (err) {
      console.error('Failed to load samples:', err);
    }
  }

  function renderPresets(samples) {
    if (!presetsContainer) return;
    presetsContainer.innerHTML = '';

    samples.slice(0, 6).forEach(sample => {
      const chip = document.createElement('button');
      chip.className = `preset-chip ${sample.ground_truth === 'FAKE' ? 'badge-fake' : 'badge-real'}`;
      const icon = sample.ground_truth === 'FAKE' ? '⚠️' : '✓';
      chip.innerHTML = `<span>${icon}</span> <span>${sample.category}: ${sample.title.substring(0, 38)}...</span>`;
      chip.addEventListener('click', () => {
        inputTitle.value = sample.title;
        inputText.value = sample.content;
        updateCounters();
        // Smooth scroll to card
        document.getElementById('inspectorCard').scrollIntoView({ behavior: 'smooth' });
        // Automatically analyze
        triggerAnalysis();
      });
      presetsContainer.appendChild(chip);
    });
  }

  // ==========================================
  // Primary Article Analysis
  // ==========================================
  async function triggerAnalysis() {
    const text = inputText.value.trim();
    const title = inputTitle.value.trim();

    if (!text || text.length < 10) {
      alert('Please enter or select at least 10 characters of article text to analyze.');
      return;
    }

    setAnalyzing(true);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, text, include_saliency: true })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Analysis request failed.');
      }

      const data = await response.json();
      currentAnalysisData = data;
      renderAnalysisResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      alert('Analysis error: ' + err.message);
    } finally {
      setAnalyzing(false);
    }
  }

  btnAnalyze.addEventListener('click', triggerAnalysis);

  function setAnalyzing(isBusy) {
    btnAnalyze.disabled = isBusy;
    if (isBusy) {
      analyzeSpinner.style.display = 'inline-block';
      analyzeBtnText.textContent = 'Deconstructing Text & LLM Reasoning...';
    } else {
      analyzeSpinner.style.display = 'none';
      analyzeBtnText.textContent = 'Analyze Veracity with LLM';
    }
  }

  // ==========================================
  // Render Results Dashboard
  // ==========================================
  function renderAnalysisResults(data) {
    resultsSection.style.display = 'block';

    const cls = data.classification;
    const reasoning = data.reasoning;
    const pillars = reasoning.pillars;

    // 1. Verdict Badge
    const verdictBadge = document.getElementById('verdictBadge');
    const isFake = cls.verdict.includes('FAKE') || cls.verdict.includes('MISINFORMATION');

    verdictBadge.className = `verdict-badge ${isFake ? 'verdict-fake' : 'verdict-real'}`;
    verdictBadge.innerHTML = isFake ? '⚠️ ' + cls.verdict : '🛡️ ' + cls.verdict;

    // Confidence
    document.getElementById('confidenceVal').textContent = `${cls.confidence_percent.toFixed(1)}%`;
    document.getElementById('riskIndexVal').textContent = `${cls.risk_index}/100`;

    // Probability Meter
    const realPct = Math.round(cls.real_probability * 100);
    const fakePct = Math.round(cls.fake_probability * 100);
    document.getElementById('probFillReal').style.width = `${realPct}%`;
    document.getElementById('probFillFake').style.width = `${fakePct}%`;
    document.getElementById('labelRealProb').textContent = `Credible: ${realPct}%`;
    document.getElementById('labelFakeProb').textContent = `Misinformation: ${fakePct}%`;

    // Runtime metadata
    document.getElementById('metaLatency').textContent = `${data.runtime.latency_ms} ms`;
    document.getElementById('metaEngine').textContent = data.runtime.engine;
    document.getElementById('metaDevice').textContent = data.runtime.device;
    document.getElementById('metaMemory').textContent = `${data.runtime.ram_used_gb} GB / ${data.runtime.ram_total_gb} GB`;

    // 2. Pillars
    renderPillar('sensationalism', pillars.sensationalism.score, '#f43f5e');
    renderPillar('sourceRigor', pillars.source_rigor.score, '#10b981');
    renderPillar('linguisticBias', pillars.linguistic_bias.score, '#f59e0b');
    renderPillar('fallacyScore', pillars.logical_fallacies.score, '#ec4899');

    // 3. Token Saliency Heatmap
    renderHeatmap(data.token_saliency || []);

    // 4. Reasoning Narrative
    renderReasoningNarrative(reasoning);

    // Smooth scroll down to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  function renderPillar(id, score, color) {
    const valEl = document.getElementById(`score_${id}`);
    const barEl = document.getElementById(`bar_${id}`);
    if (valEl) valEl.textContent = `${score}/100`;
    if (barEl) {
      barEl.style.width = `${score}%`;
      barEl.style.backgroundColor = color;
    }
  }

  function renderHeatmap(tokens) {
    const container = document.getElementById('heatmapContainer');
    if (!container) return;
    container.innerHTML = '';

    tokens.forEach(tok => {
      const span = document.createElement('span');
      span.className = 'heat-token';
      span.textContent = tok.token + ' ';

      if (tok.sentiment === 'suspicious') {
        span.classList.add('suspicious');
        if (tok.weight > 0.6) span.classList.add('high');
        span.title = `Misinformation Driver (Saliency: +${tok.weight})`;
      } else if (tok.sentiment === 'credible') {
        span.classList.add('credible');
        if (tok.weight < -0.5) span.classList.add('high');
        span.title = `Credibility Marker (Saliency: ${tok.weight})`;
      }

      container.appendChild(span);
    });
  }

  function renderReasoningNarrative(reasoning) {
    const summaryEl = document.getElementById('reasoningSummary');
    const stepsList = document.getElementById('reasoningStepsList');
    const checklist = document.getElementById('factCheckChecklist');

    if (summaryEl) summaryEl.textContent = reasoning.summary;

    if (stepsList) {
      stepsList.innerHTML = '';
      (reasoning.deductive_steps || []).forEach(step => {
        const li = document.createElement('li');
        li.className = 'reasoning-item';
        li.innerHTML = `<span class="reasoning-icon">🔍</span> <div>${step}</div>`;
        stepsList.appendChild(li);
      });
    }

    if (checklist) {
      checklist.innerHTML = '';
      (reasoning.media_literacy_checklist || []).forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        checklist.appendChild(li);
      });
    }
  }

  // ==========================================
  // Export Report as Markdown
  // ==========================================
  if (btnExport) {
    btnExport.addEventListener('click', () => {
      if (!currentAnalysisData) return;
      const d = currentAnalysisData;
      const md = `# Veritas AI Misinformation Forensic Report
Generated: ${new Date().toISOString()}

## Classification Verdict
- Verdict: ${d.classification.verdict}
- Confidence: ${d.classification.confidence_percent}%
- Risk Index: ${d.classification.risk_index}/100
- Credible Probability: ${(d.classification.real_probability * 100).toFixed(1)}%
- Fake Probability: ${(d.classification.fake_probability * 100).toFixed(1)}%

## Forensic Pillars
- Sensationalism Score: ${d.reasoning.pillars.sensationalism.score}/100
- Source Rigor Score: ${d.reasoning.pillars.source_rigor.score}/100
- Linguistic Bias Score: ${d.reasoning.pillars.linguistic_bias.score}/100
- Logical Fallacy Score: ${d.reasoning.pillars.logical_fallacies.score}/100

## Reasoning Chain
${d.reasoning.deductive_steps.map(s => `- ${s}`).join('\n')}

## Media Literacy Recommendations
${d.reasoning.media_literacy_checklist.map(c => `- [ ] ${c}`).join('\n')}

---
Engine: ${d.runtime.engine} | Device: ${d.runtime.device} | Latency: ${d.runtime.latency_ms} ms
`;

      navigator.clipboard.writeText(md).then(() => {
        alert('Forensic Report successfully copied to clipboard in Markdown format!');
      }).catch(err => {
        console.error('Copy failed:', err);
      });
    });
  }

  // ==========================================
  // Benchmark Evaluation Suite
  // ==========================================
  async function loadBenchmarkMetrics() {
    if (benchmarkSpinner) benchmarkSpinner.style.display = 'inline-block';
    try {
      const res = await fetch('/api/evaluate');
      const data = await res.json();
      renderBenchmarkData(data);
    } catch (err) {
      console.error('Benchmark fetch failed:', err);
    } finally {
      if (benchmarkSpinner) benchmarkSpinner.style.display = 'none';
    }
  }

  if (btnRunBenchmark) {
    btnRunBenchmark.addEventListener('click', loadBenchmarkMetrics);
  }

  function renderBenchmarkData(data) {
    if (!data || data.error) return;

    document.getElementById('benchAccuracy').textContent = `${data.accuracy}%`;
    document.getElementById('benchF1').textContent = `${data.f1_score}%`;
    document.getElementById('benchPrecision').textContent = `${data.precision}%`;
    document.getElementById('benchRecall').textContent = `${data.recall}%`;
    document.getElementById('benchRocAuc').textContent = `${data.roc_auc}`;
    document.getElementById('benchLatency').textContent = `${data.performance.avg_latency_ms} ms`;

    // Confusion Matrix cells
    const cm = data.confusion_matrix;
    document.getElementById('cmTN').textContent = cm.true_negatives;
    document.getElementById('cmFP').textContent = cm.false_positives;
    document.getElementById('cmFN').textContent = cm.false_negatives;
    document.getElementById('cmTP').textContent = cm.true_positives;
  }

  // Initial load
  fetchSamples();
});

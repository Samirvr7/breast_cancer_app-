(function () {
  const form = document.getElementById('patient-form');
  const segButtons = document.querySelectorAll('.seg');
  const modelInput = document.getElementById('model-input');
  const submitBtn = document.getElementById('submit-btn');

  const resultEmpty = document.getElementById('result-empty');
  const resultFilled = document.getElementById('result-filled');
  const gaugeFill = document.getElementById('gauge-fill');
  const gaugeNeedle = document.getElementById('gauge-needle');
  const gaugeValue = document.getElementById('gauge-value');
  const riskBadge = document.getElementById('risk-badge');
  const modelLabel = document.getElementById('result-model-label');
  const factorChartCanvas = document.getElementById('factor-chart');
  const staleNotice = document.getElementById('stale-notice');

  const GAUGE_CIRCUMFERENCE = 251.2;
  let lastScoredModel = null;
  let factorChart = null;

  segButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      segButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      modelInput.value = btn.dataset.model;
      if (!resultFilled.hidden && lastScoredModel && lastScoredModel !== modelInput.value) {
        staleNotice.hidden = false;
      } else {
        staleNotice.hidden = true;
      }
    });
  });

  function riskLabel(tier) {
    return { low: 'Low risk', moderate: 'Moderate risk', high: 'High risk' }[tier] || tier;
  }

  function renderFactorChart(factors) {
    if (!factors || !factors.length || typeof Chart === 'undefined') return;

    const labels = factors.map((f) => f.feature.replace(/_/g, ' '));
    const values = factors.map((f) => f.value);
    const barColors = factors.map((f) => (f.direction === 'up' ? '#B3392C' : '#2C7A56'));

    if (factorChart) factorChart.destroy();
    factorChart = new Chart(factorChartCanvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: barColors,
          borderRadius: 4,
          barThickness: 18,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const dir = ctx.raw >= 0 ? 'pushes risk UP' : 'pushes risk DOWN';
                return `${Math.abs(ctx.raw).toFixed(3)} — ${dir}`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: '← lowers risk    raises risk →', font: { size: 10 } },
            grid: { color: '#EFF3F4' },
          },
          y: { grid: { display: false } },
        },
      },
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = 'Scoring…';
    staleNotice.hidden = true;

    const payload = { model: modelInput.value };
    new FormData(form).forEach((value, key) => { payload[key] = value; });

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');

      resultEmpty.hidden = true;
      resultFilled.hidden = false;
      lastScoredModel = data.model_key;

      const pct = Math.round(data.probability_malignant * 100);
      gaugeValue.textContent = pct + '%';

      const offset = GAUGE_CIRCUMFERENCE * (1 - data.probability_malignant);
      gaugeFill.style.strokeDashoffset = offset;

      const angle = -90 + (180 * data.probability_malignant);
      gaugeNeedle.style.transform = `rotate(${angle}deg)`;

      const colorMap = { low: '#2C7A56', moderate: '#C97E2B', high: '#B3392C' };
      gaugeFill.style.stroke = colorMap[data.risk_tier] || '#0E7C86';

      riskBadge.textContent = `${riskLabel(data.risk_tier)} · ${data.prediction}`;
      riskBadge.className = 'risk-badge ' + data.risk_tier;

      modelLabel.textContent = `Scored with ${data.model_used}`;

      renderFactorChart(data.top_factors);
    } catch (err) {
      alert(err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Run assessment';
    }
  });
})();

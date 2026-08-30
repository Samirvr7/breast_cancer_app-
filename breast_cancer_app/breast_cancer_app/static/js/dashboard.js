(function () {
  if (typeof Chart === 'undefined') {
    console.error('Chart.js failed to load — check static/js/vendor/chart.umd.min.js exists and the network tab for a 404.');
    return;
  }
  const metrics = window.__METRICS__;
  const importances = window.__IMPORTANCES__;
  if (!metrics || !importances) {
    console.error('Dashboard data missing — window.__METRICS__ or window.__IMPORTANCES__ is empty.');
    return;
  }

  const TEAL = '#0E7C86';
  const NAVY = '#10202E';
  const AMBER = '#C97E2B';
  const GREEN = '#2C7A56';
  const GRID = '#DCE3E7';

  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.color = '#4B5D6B';

  // ---- ROC curve ----
  const rocCtx = document.getElementById('roc-chart');
  new Chart(rocCtx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: `Random Forest (AUC ${metrics.random_forest.roc_auc})`,
          data: metrics.random_forest.roc_curve.fpr.map((x, i) => ({ x, y: metrics.random_forest.roc_curve.tpr[i] })),
          borderColor: TEAL, backgroundColor: 'transparent', tension: 0.2, pointRadius: 0, borderWidth: 2,
        },
        {
          label: `Logistic Regression (AUC ${metrics.logistic_regression.roc_auc})`,
          data: metrics.logistic_regression.roc_curve.fpr.map((x, i) => ({ x, y: metrics.logistic_regression.roc_curve.tpr[i] })),
          borderColor: NAVY, backgroundColor: 'transparent', tension: 0.2, pointRadius: 0, borderWidth: 2,
        },
        {
          label: 'Random guess',
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: '#B7C4CC', borderDash: [4, 4], pointRadius: 0, borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False Positive Rate' }, grid: { color: GRID } },
        y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate' }, grid: { color: GRID } },
      },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } } },
    },
  });

  // ---- class balance ----
  const balCtx = document.getElementById('balance-chart');
  new Chart(balCtx, {
    type: 'doughnut',
    data: {
      labels: ['Benign', 'Malignant'],
      datasets: [{ data: [metrics.class_balance.benign, metrics.class_balance.malignant], backgroundColor: [GREEN, '#B3392C'] }],
    },
    options: { plugins: { legend: { position: 'bottom' } }, cutout: '62%' },
  });

  // ---- feature importance bars ----
  function importanceChart(id, entries, color) {
    const ctx = document.getElementById(id);
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: entries.map((e) => e.feature.replace(/_/g, ' ')),
        datasets: [{ data: entries.map((e) => e.value), backgroundColor: color, borderRadius: 4 }],
      },
      options: {
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: { x: { grid: { color: GRID } }, y: { grid: { display: false } } },
      },
    });
  }

  importanceChart('rf-importance-chart', importances.random_forest, TEAL);
  importanceChart('lr-importance-chart', importances.logistic_regression, AMBER);
})();

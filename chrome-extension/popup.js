const DEFAULT_API_BASE = 'http://localhost:8000/api';
let API_BASE = DEFAULT_API_BASE;

async function loadApiBase() {
  try {
    const { apiBaseUrl } = await chrome.storage.sync.get('apiBaseUrl');
    API_BASE = (apiBaseUrl && apiBaseUrl.trim()) || DEFAULT_API_BASE;
  } catch (err) {
    API_BASE = DEFAULT_API_BASE;
  }
}

let selectedText = '';

// Check for selected text when popup opens
document.addEventListener('DOMContentLoaded', async () => {
  await loadApiBase();
  setupSettingsUI();

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString()
    });
    
    if (result && result.trim().length > 0) {
      selectedText = result;
      showSelection(result);
    } else {
      document.getElementById('no-selection').style.display = 'block';
    }
  } catch (err) {
    document.getElementById('no-selection').style.display = 'block';
  }
});

function showSelection(text) {
  document.getElementById('no-selection').style.display = 'none';
  document.getElementById('selection-ready').style.display = 'block';
  document.getElementById('selected-preview').textContent = text.substring(0, 200) + (text.length > 200 ? '...' : '');
}

document.getElementById('analyze-btn')?.addEventListener('click', async () => {
  if (!selectedText) return;
  
  document.getElementById('selection-ready').style.display = 'none';
  document.getElementById('loading').style.display = 'block';
  
  try {
    const response = await fetch(`${API_BASE}/analyze-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: selectedText,
        check_sources: true,
        extract_claims: false
      })
    });
    
    const data = await response.json();
    showResult(data);
  } catch (err) {
    alert('Analysis failed: ' + err.message);
    document.getElementById('loading').style.display = 'none';
    document.getElementById('selection-ready').style.display = 'block';
  }
});

function showResult(data) {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('result').style.display = 'block';
  
  const score = Math.round(data.credibility_score);
  const scoreEl = document.getElementById('score-value');
  scoreEl.textContent = `${score}%`;
  
  let color = '#00C853';
  if (score < 70) color = '#FFC107';
  if (score < 40) color = '#FF2A2A';
  scoreEl.style.color = color;
  
  const badge = document.getElementById('prediction-badge');
  badge.textContent = data.prediction;
  badge.className = 'prediction ' + data.prediction.toLowerCase();
  
  document.getElementById('explanation-text').textContent = data.explanation || 'Analysis completed.';
}

document.getElementById('new-analysis-btn')?.addEventListener('click', () => {
  document.getElementById('result').style.display = 'none';
  document.getElementById('selection-ready').style.display = 'block';
});

function setupSettingsUI() {
  const toggle = document.getElementById('settings-toggle');
  const panel = document.getElementById('settings-panel');
  const input = document.getElementById('api-base-input');
  const status = document.getElementById('settings-status');

  if (!toggle || !panel || !input) return;

  input.value = API_BASE;

  toggle.addEventListener('click', () => {
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  });

  document.getElementById('save-settings-btn')?.addEventListener('click', async () => {
    const value = input.value.trim().replace(/\/+$/, '');
    if (!value) return;
    await chrome.storage.sync.set({ apiBaseUrl: value });
    API_BASE = value;
    if (status) {
      status.textContent = 'Saved.';
      setTimeout(() => { status.textContent = ''; }, 2000);
    }
  });

  document.getElementById('reset-settings-btn')?.addEventListener('click', async () => {
    await chrome.storage.sync.remove('apiBaseUrl');
    API_BASE = DEFAULT_API_BASE;
    input.value = DEFAULT_API_BASE;
    if (status) {
      status.textContent = 'Reset to default.';
      setTimeout(() => { status.textContent = ''; }, 2000);
    }
  });
}

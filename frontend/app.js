'use strict';

const $ = (id) => document.getElementById(id);
const MAX_SIZE = 10 * 1024 * 1024;
const allowedTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);
let selectedFile = null;
let previewUrl = null;
let busy = false;
let ready = false;
let selectionVersion = 0;

function showError(message) {
  $('upload-error').textContent = message;
  $('upload-error').hidden = !message;
}

function showResultState(state) {
  $('empty-result').hidden = state !== 'empty';
  $('loading-result').hidden = state !== 'loading';
  $('prediction-result').hidden = state !== 'result';
  $('result-tag').textContent = {empty: 'Ждём фото', loading: 'Анализируем', result: 'Готово'}[state];
}

function updateControls() {
  $('predict-button').disabled = busy || !ready || !selectedFile;
  $('predict-text').textContent = busy ? 'Распознаём…' : 'Распознать фото';
  $('predict-arrow').hidden = busy;
  $('predict-spinner').hidden = !busy;
  $('remove-file').disabled = busy;
  $('dropzone').disabled = busy;
  $('file-input').disabled = busy;
  document.querySelectorAll('.example').forEach((button) => { button.disabled = busy; });
  $('another-photo').disabled = busy;
}

function resetPhoto() {
  if (busy) return;
  selectionVersion++;
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('preview').removeAttribute('src');
  $('preview-box').hidden = true;
  $('dropzone').hidden = false;
  $('file-input').value = '';
  showError('');
  showResultState('empty');
  updateControls();
}

async function selectPhoto(file) {
  if (!file || busy) return;
  const version = ++selectionVersion;
  showError('');
  if (file.size === 0) return showError('Файл пустой. Выберите другое фото.');
  if (file.size > MAX_SIZE) return showError('Файл слишком большой: максимум 10 МБ.');
  if (!allowedTypes.has(file.type) && (file.type || !/\.(jpe?g|png|webp)$/i.test(file.name))) {
    return showError('Выберите фото в формате JPG, PNG или WebP.');
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    if (version !== selectionVersion) { URL.revokeObjectURL(url); return; }
    if (image.naturalWidth * image.naturalHeight > 20_000_000) {
      URL.revokeObjectURL(url);
      return showError('Фото слишком большое: максимум 20 мегапикселей.');
    }
  } catch {
    URL.revokeObjectURL(url);
    if (version === selectionVersion) showError('Не удалось прочитать фото. Возможно, файл повреждён.');
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = url;
  selectedFile = file;
  $('preview').src = previewUrl;
  $('file-name').textContent = file.name;
  $('file-size').textContent = file.size < 1024 * 1024
    ? `${Math.max(1, Math.round(file.size / 1024))} КБ`
    : `${(file.size / 1024 / 1024).toFixed(1)} МБ`;
  $('preview-box').hidden = false;
  $('dropzone').hidden = true;
  showResultState('empty');
  updateControls();
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health', {signal: AbortSignal.timeout(8000), cache: 'no-store'});
    if (!response.ok) throw new Error('Unavailable');
    const data = await response.json();
    ready = data.status === 'ready';
  } catch { ready = false; }
  $('model-status').className = `status ${ready ? 'ready' : 'offline'}`;
  $('status-text').textContent = ready ? 'Модель готова к работе' : 'Нет связи с моделью';
  updateControls();
}

const percent = (value) => `${(value * 100).toLocaleString('ru-RU', {minimumFractionDigits: 1, maximumFractionDigits: 1})}%`;

async function predict() {
  if (busy || !selectedFile || !ready) return;
  selectionVersion++;
  busy = true;
  showError('');
  showResultState('loading');
  updateControls();
  try {
    const form = new FormData();
    form.append('file', selectedFile);
    const response = await fetch('/api/predict', {method: 'POST', body: form, signal: AbortSignal.timeout(60000)});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Сервис временно недоступен. Попробуйте снова.');
    $('result-label').textContent = data.label_ru;
    $('result-icon').setAttribute('href', data.label === 'cat' ? '#icon-cat' : '#icon-dog');
    $('confidence').textContent = percent(data.confidence);
    $('cat-value').textContent = percent(data.probabilities.cat);
    $('dog-value').textContent = percent(data.probabilities.dog);
    $('cat-progress').value = data.probabilities.cat;
    $('dog-progress').value = data.probabilities.dog;
    $('inference-time').textContent = `${Math.round(data.inference_ms)} мс · ${data.image.width} × ${data.image.height}`;
    $('low-confidence').hidden = data.confidence >= 0.65;
    showResultState('result');
  } catch (error) {
    showResultState('empty');
    const timeout = error.name === 'TimeoutError' || error.name === 'AbortError';
    showError(timeout ? 'Анализ занял слишком много времени. Попробуйте ещё раз.' : error instanceof TypeError ? 'Нет связи с сервером. Проверьте подключение и попробуйте снова.' : error.message);
    void checkHealth();
  } finally {
    busy = false;
    updateControls();
  }
}

$('dropzone').addEventListener('click', () => { $('file-input').value = ''; $('file-input').click(); });
$('file-input').addEventListener('change', (event) => void selectPhoto(event.target.files[0]));
$('remove-file').addEventListener('click', resetPhoto);
$('predict-button').addEventListener('click', () => void predict());
$('another-photo').addEventListener('click', () => { resetPhoto(); $('dropzone').focus(); });

for (const name of ['dragenter', 'dragover']) {
  $('dropzone').addEventListener(name, (event) => { event.preventDefault(); if (!busy) $('dropzone').classList.add('dragging'); });
}
for (const name of ['dragleave', 'drop']) {
  $('dropzone').addEventListener(name, (event) => { event.preventDefault(); $('dropzone').classList.remove('dragging'); });
}
$('dropzone').addEventListener('drop', (event) => {
  if (event.dataTransfer.files.length > 1) return showError('Выберите одно фото для распознавания.');
  void selectPhoto(event.dataTransfer.files[0]);
});
document.addEventListener('dragover', (event) => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); });
document.addEventListener('drop', (event) => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); });

document.querySelectorAll('.example').forEach((button) => {
  button.addEventListener('click', async () => {
    if (busy) return;
    const version = ++selectionVersion;
    const animal = button.dataset.example;
    try {
      const response = await fetch(`/assets/${animal}.jpg`);
      if (!response.ok) throw new Error('Example unavailable');
      const blob = await response.blob();
      if (version !== selectionVersion) return;
      await selectPhoto(new File([blob], animal === 'cat' ? 'Пример — кошка.jpg' : 'Пример — собака.jpg', {type: 'image/jpeg'}));
    } catch { if (version === selectionVersion) showError('Не удалось открыть пример. Выберите фото с устройства.'); }
  });
});
void checkHealth();
setInterval(() => { if (!busy && !document.hidden) void checkHealth(); }, 30000);
document.addEventListener('visibilitychange', () => { if (!document.hidden && !busy) void checkHealth(); });
window.addEventListener('pagehide', () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });

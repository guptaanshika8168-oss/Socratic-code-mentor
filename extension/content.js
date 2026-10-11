(() => {
  function getCode() {
    try {
      if (window.monaco?.editor?.getModels) {
        const m = window.monaco.editor.getModels()[0];
        if (m?.getValue()?.trim()) return m.getValue();
      }
    } catch (_) {}
    try {
      const cm = document.querySelector('.CodeMirror');
      if (cm?.CodeMirror?.getValue) return cm.CodeMirror.getValue();
    } catch (_) {}
    for (const selector of ['.cm-content', 'textarea.ace_text-input', '.ace_text-input', 'textarea', '[contenteditable="true"]']) {
      for (const el of document.querySelectorAll(selector)) {
        const val = ('value' in el ? el.value : el.innerText || el.textContent || '').trim();
        if (val.length > 20 && (val.includes('\n') || val.includes('{') || val.includes('def '))) return val;
      }
    }
    const lines = [...document.querySelectorAll('.view-lines .view-line')].map(x => x.innerText || x.textContent || '');
    const fallback = lines.join('\n').trim();
    return fallback.length > 20 ? fallback : '';
  }
  function getProblem() {
    for (const selector of ['[data-track-load="description_content"]', '[data-testid="question-content"]', '.challenge-body-html', '.problem-statement', 'article']) {
      const el = document.querySelector(selector);
      if (el?.innerText?.trim().length > 80) return el.innerText.trim().slice(0, 12000);
    }
    return ((document.querySelector('main') || document.body).innerText || '').trim().slice(0, 12000);
  }
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg?.type !== 'SCM_EXTRACT_PAGE') return;
    sendResponse({ok:true, url:location.href, title:document.title, problem:getProblem(), code:getCode()});
    return true;
  });
})();

// Clearchart frontend: plain JavaScript, no build step.
// Every response shape used here is documented in API_CONTRACT.md.

const API_BASE = ""; // same-origin; change if backend is deployed separately

const CHAR_WARN_LIMIT = 20000;
// The client must wait longer than the server can take, or we give up on a request the
// server still finishes (and bills). Each Claude call can take up to 150 s and is retried
// once (backend ANTHROPIC_TIMEOUT_SECONDS / ANTHROPIC_MAX_RETRIES), so about 5 minutes per call.
// OCR makes one call; translate makes two (explain, then fact-check) plus quick lookups.
// If you raise those settings in backend/.env, raise these too.
const TIMEOUTS_MS = { quick: 15000, ocr: 330000, translate: 660000 };
// localStorage key for "I understand" on the demo privacy notice. Change the version suffix
// if the notice's wording changes, so everyone sees the new text once.
const DEMO_NOTICE_KEY = "clearchart.demoNoticeDismissed.v1";

// --- display labels for values the API returns ---
const SAMPLE_GROUPS = [
  ["lab", "Lab reports"],
  ["discharge", "Discharge summaries"],
  ["radiology", "Radiology reports"],
  ["tricky", "Tricky cases"],
  ["other", "Other"],
];
const ENGINE_LABELS = { tesseract: "Tesseract OCR", claude: "Claude vision", "pdf-text": "PDF text layer (no OCR needed)" };
const LEVEL_LABELS = { "5th": "5th-grade reading level", "8th": "8th-grade reading level", adult: "adult reading level" };
const SOURCE_KIND_LABELS = { lab_test: "Lab test", topic: "Health topic", drug: "Medicine" };
const CLAIM_KIND_LABELS = { patient_specific: "About you", general: "General info" };
const RANGE_SOURCE_NOTES = {
  report: "from your report",
  typical: "typical adult range — your lab's may differ",
  report_flag: "flag printed on report",
};
const FLAG_BADGES = {
  high: { label: "High", icon: "↑" },
  low: { label: "Low", icon: "↓" },
  normal: { label: "Normal", icon: "" },
  unclear: { label: "Can't compare", icon: "" }, // has a range, but the value can't be placed in it (e.g. "<15")
  unknown: { label: "No range", icon: "" }, // no range at all
};
const STATUS_MESSAGES = {
  413: "That document is too large. Try a shorter section or a smaller file.",
  429: "Too many requests right now. Wait a minute and try again.",
  500: "The server hit an unexpected error. Please try again.",
  502: "The AI service had a problem. Please try again in a moment.",
  503: "The AI service is unavailable right now. Please try again later.",
};
const COPY_DISCLAIMER =
  "Educational explanation from Clearchart, not medical advice. Talk with your doctor or care team about your results.";

// --- app state ---
let selectedFile = null; // file chosen in the Upload tab
let selectedFileRead = false; // is that file's OCR text what's in the textbox now?
let thumbUrl = null; // object URL for the image preview (revoked when replaced)
let lastResult = null; // most recent /api/translate response
let lastResultText = null; // the document text that lastResult explains
let loadingTimer = null; // interval id for the elapsed-seconds counter
let copyStatusTimer = null; // timeout id that clears the "Copied." message
let translating = false; // true while a /api/translate request is running
let ocrBusy = false; // true while the latest /api/ocr request is running
let ocrRequestId = 0; // bumped to make an OCR answer that is still on its way be ignored
let samplesReady = false; // has the list of examples loaded?
let sampleLoading = false; // true while one example is being fetched
let loadedSampleText = ""; // what the last example put in the textbox (to spot user edits)

// --- element lookups ---
const $ = (id) => document.getElementById(id);
const documentText = $("documentText");
const sampleSelect = $("sampleSelect");
const fileInput = $("fileInput");
const cameraInput = $("cameraInput");
const dropZone = $("dropZone");
const ocrBtn = $("ocrBtn");
const ocrNotice = $("ocrNotice");
const translateBtn = $("translateBtn");
const outputPanel = $("outputPanel");
const explanationBody = $("explanationBody");

// --- small helpers ---
function escapeHtml(value) {
  const replacements = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(value ?? "").replace(/[&<>"']/g, (ch) => replacements[ch]);
}

// Only allow http(s) links from the API (blocks "javascript:" and similar)
function safeUrl(url) {
  try {
    const parsed = new URL(url, window.location.href);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : "#";
  } catch {
    return "#";
  }
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// Read a message aloud to screen-reader users without showing it on screen
function announce(message) {
  const region = $("srStatus");
  region.textContent = "";
  setTimeout(() => (region.textContent = message), 50);
}

// --- talking to the backend ---
async function request(path, options = {}, timeoutMs = TIMEOUTS_MS.quick) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(API_BASE + path, { ...options, signal: controller.signal });
  } catch (err) {
    // After a timeout the server may still be working (and each try counts toward the
    // hourly limit), so don't invite an instant retry.
    throw new Error(
      err.name === "AbortError"
        ? "This is taking much longer than expected. The server may still be working on it, so wait a few minutes before trying again."
        : "Couldn't reach the Clearchart server. Check that it's running, then try again."
    );
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) throw new Error(await errorMessageFrom(res));
  try {
    return await res.json();
  } catch {
    throw new Error("The server sent a response the app couldn't read.");
  }
}

// Errors carry {"detail": "..."}; fall back to a friendly message per status code
async function errorMessageFrom(res) {
  const body = await res.json().catch(() => null);
  const detail = body && body.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((d) => d && d.msg).filter(Boolean);
    if (messages.length) return messages.join(" ");
  }
  return STATUS_MESSAGES[res.status] || `Request failed (${res.status}).`;
}

// --- accessible tabs (shared by the input and output tab bars) ---
function setupTabs(tablist) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      const keyTargets = {
        ArrowRight: tabs[(index + 1) % tabs.length],
        ArrowLeft: tabs[(index - 1 + tabs.length) % tabs.length],
        Home: tabs[0],
        End: tabs[tabs.length - 1],
      };
      const next = keyTargets[event.key];
      if (!next) return;
      event.preventDefault();
      selectTab(next);
      next.focus();
    });
  });
}

function selectTab(tab) {
  const tabs = tab.closest('[role="tablist"]').querySelectorAll('[role="tab"]');
  tabs.forEach((t) => {
    const selected = t === tab;
    t.setAttribute("aria-selected", String(selected));
    t.tabIndex = selected ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !selected;
  });
}

function isTabSelected(tabId) {
  return $(tabId).getAttribute("aria-selected") === "true";
}

// Small count badge on a tab, e.g. "Lab results 14"
function setTabCount(tabId, count, noun, warn = false) {
  const badge = $(tabId).querySelector(".tab-count");
  badge.innerHTML = `${count}<span class="visually-hidden"> ${escapeHtml(noun)}</span>`;
  badge.hidden = !count;
  badge.classList.toggle("warn", warn && count > 0);
}

// --- demo privacy notice (shown until "I understand"; remembered in this browser) ---
// Storage can throw (private windows, blocked site data). Then the notice just shows on every
// visit, which is the safe side for a privacy warning.
function demoNoticeDismissed() {
  try {
    return window.localStorage.getItem(DEMO_NOTICE_KEY) === "1";
  } catch {
    return false;
  }
}

function rememberDemoNoticeDismissed() {
  try {
    window.localStorage.setItem(DEMO_NOTICE_KEY, "1");
  } catch {
    // Not saved: it hides for now and comes back next visit
  }
}

function setupDemoNotice() {
  $("demoNotice").hidden = demoNoticeDismissed();
  $("demoNoticeBtn").addEventListener("click", () => {
    rememberDemoNoticeDismissed();
    $("demoNotice").hidden = true;
    // The focused button just disappeared: continue at the first control in the workspace
    document.querySelector('.input-tabs [aria-selected="true"]').focus();
  });
}

// --- server health (API key banner, OCR availability, demo rate limit) ---
async function checkHealth() {
  try {
    const health = await request("/api/health");
    $("healthBanner").hidden = health.api_key_configured !== false;
    $("ocrEngineNote").hidden = health.ocr_engine !== "none";
    showRateLimit(health.rate_limit_per_hour);
  } catch {
    // Server unreachable: the translate button will show a clear error when used.
  }
}

// Line under the Translate button. Older servers don't send the field, and 0 means no limit.
function showRateLimit(limit) {
  const hasLimit = typeof limit === "number" && Number.isFinite(limit) && limit > 0;
  // The server counts translations and file reads in the same hourly allowance.
  const noun = limit === 1 ? "request" : "requests";
  $("rateLimitNote").textContent = hasLimit ? `Demo limit: ${limit.toLocaleString("en-US")} ${noun} per hour.` : "";
  $("rateLimitNote").hidden = !hasLimit;
}

// --- keeping buttons and inputs in step with what's running ---
// Each "disabled" state is worked out from the flags above, so no code path can forget one.
function updateSampleSelect() {
  sampleSelect.disabled = !samplesReady || sampleLoading || translating;
}

function updateOcrButton() {
  ocrBtn.disabled = !selectedFile || ocrBusy || translating;
}

// While a translation runs the text stays fixed, so the result always matches what's on screen
function setTranslating(busy) {
  translating = busy;
  documentText.readOnly = busy; // readOnly (not disabled) keeps the text readable and selectable
  updateSampleSelect();
  updateOcrButton();
}

// Setting .value from code doesn't fire the "input" event, so refresh what depends on the text here
function setDocumentText(text) {
  documentText.value = text;
  updateCharCount();
  updateStaleNotice();
}

// Form uploads can turn "\n" into "\r\n", so compare text with line endings evened out
function sameText(a, b) {
  const normalize = (s) => String(s ?? "").replace(/\r\n?/g, "\n").trim();
  return normalize(a) === normalize(b);
}

// Warn when the explanation on screen was made from different text than what's in the box now
function updateStaleNotice() {
  const stale = lastResultText !== null && !sameText(lastResultText, documentText.value);
  $("staleNotice").hidden = !stale;
}

// --- character count ---
function updateCharCount() {
  const count = documentText.value.length;
  const over = count > CHAR_WARN_LIMIT;
  const formatted = count.toLocaleString("en-US");
  $("charCount").textContent = over
    ? `${formatted} characters — over ${CHAR_WARN_LIMIT.toLocaleString("en-US")}, so the server may reject it. Try trimming to the part you need explained.`
    : `${formatted} characters`;
  $("charCount").classList.toggle("warn", over);
}

// --- "Try an example" samples ---
async function loadSampleList() {
  try {
    const samples = await request("/api/samples");
    fillSampleSelect(Array.isArray(samples) ? samples : []);
  } catch {
    sampleSelect.innerHTML = `<option value="">Examples unavailable</option>`;
  }
}

function fillSampleSelect(samples) {
  if (!samples.length) {
    sampleSelect.innerHTML = `<option value="">No examples available</option>`;
    return;
  }
  const knownCategories = SAMPLE_GROUPS.map(([key]) => key);
  const groupOf = (sample) => (knownCategories.includes(sample.category) ? sample.category : "other");

  let html = `<option value="">Choose a sample document…</option>`;
  for (const [category, label] of SAMPLE_GROUPS) {
    const inGroup = samples.filter((s) => groupOf(s) === category);
    if (!inGroup.length) continue;
    const options = inGroup.map((s) => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.title)}</option>`);
    html += `<optgroup label="${escapeHtml(label)}">${options.join("")}</optgroup>`;
  }
  sampleSelect.innerHTML = html;
  samplesReady = true;
  updateSampleSelect();
}

async function loadSample(id) {
  if (!id) return;
  // Ask before replacing text the user typed or corrected after OCR. Setting .value from code
  // also wipes the textbox's undo history, so there'd be no way back. An untouched example
  // can be swapped without asking.
  const current = documentText.value.trim();
  if (current && current !== loadedSampleText.trim() &&
      !window.confirm("Replace the text in the box with this example? What you typed or corrected will be lost.")) {
    sampleSelect.value = ""; // the box doesn't hold an example, so show the placeholder again
    return;
  }
  sampleLoading = true;
  updateSampleSelect();
  hideInputError();
  try {
    const sample = await request(`/api/samples/${encodeURIComponent(id)}`);
    abandonOcr(); // an OCR answer arriving later must not overwrite the example
    selectedFileRead = false; // the box no longer holds the uploaded file's text
    loadedSampleText = sample.text || "";
    setDocumentText(loadedSampleText);
    hideOcrNotice();
    announce(`Loaded example: ${sample.title}.`);
  } catch (err) {
    sampleSelect.value = ""; // so picking the same example again retries it
    showInputError(`Couldn't load that example. ${err.message}`);
  } finally {
    sampleLoading = false;
    updateSampleSelect();
  }
}

// --- file upload: picker, camera, drag-and-drop ---
function isSupportedFile(file) {
  return file.type.startsWith("image/") || file.type === "application/pdf" || /\.pdf$/i.test(file.name);
}

function setSelectedFile(file) {
  $("ocrError").hidden = true;
  if (file && !isSupportedFile(file)) {
    showOcrError("That file type isn't supported. Choose a photo (JPG, PNG, WebP or GIF) or a PDF.");
    file = null;
  }
  abandonOcr(); // text still being read from the previous file must not be shown as this one's
  selectedFile = file;
  selectedFileRead = false;
  if (thumbUrl) URL.revokeObjectURL(thumbUrl);
  thumbUrl = null;

  $("filePreview").hidden = !file;
  updateOcrButton();
  if (!file) {
    fileInput.value = ""; // lets the same file be picked again after "Remove"
    cameraInput.value = "";
    return;
  }
  const isImage = file.type.startsWith("image/");
  if (isImage) {
    thumbUrl = URL.createObjectURL(file);
    $("fileThumb").src = thumbUrl;
  }
  $("fileThumb").hidden = !isImage;
  $("filePdfBadge").hidden = isImage;
  $("fileName").textContent = file.name;
  $("fileMeta").textContent = `${isImage ? "Photo" : "PDF"} · ${formatBytes(file.size)}`;
}

function setupDropZone() {
  ["dragenter", "dragover"].forEach((type) =>
    dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      dropZone.classList.add("dragging");
    })
  );
  dropZone.addEventListener("dragleave", (event) => {
    if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove("dragging");
  });
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    const file = event.dataTransfer.files[0];
    if (file) setSelectedFile(file);
  });

  // Stop the browser from opening a file that misses the drop zone
  // (text dragged into the textarea still works because we only block files).
  const blockFileDrop = (event) => {
    const types = Array.from((event.dataTransfer && event.dataTransfer.types) || []);
    if (types.includes("Files")) event.preventDefault();
  };
  window.addEventListener("dragover", blockFileDrop);
  window.addEventListener("drop", blockFileDrop);
}

// --- OCR: read text from the file, then let the user check it ---
async function readTextFromFile() {
  if (!selectedFile || ocrBusy || translating) return;
  const requestId = ++ocrRequestId; // this request's number; a newer one makes it stale
  const form = new FormData();
  form.append("file", selectedFile);
  $("ocrError").hidden = true;
  setOcrBusy(true);
  announce("Reading text from your file. This can take a minute or two.");

  try {
    const data = await request("/api/ocr", { method: "POST", body: form }, TIMEOUTS_MS.ocr);
    if (requestId !== ocrRequestId) return; // file changed or an example loaded meanwhile: drop it
    const text = (data.text || "").trim();
    if (!text) {
      showOcrError("No readable text was found in that file. Try a sharper, well-lit photo, or paste the text instead.");
      return;
    }
    selectedFileRead = true;
    loadedSampleText = ""; // OCR text counts as the user's own: ask before an example replaces it
    sampleSelect.value = "";
    setDocumentText(text);
    hideInputError();
    selectTab($("tab-paste-btn"));
    showOcrNotice(data);
  } catch (err) {
    if (requestId !== ocrRequestId) return; // an error for a file that's no longer selected
    showOcrError(`Couldn't read that file. ${err.message}`);
  } finally {
    // Only the latest request may clear the "Reading…" state
    if (requestId === ocrRequestId) setOcrBusy(false);
  }
}

function setOcrBusy(busy) {
  ocrBusy = busy;
  $("ocrStatus").hidden = !busy;
  ocrBtn.textContent = busy ? "Reading…" : "Read text from file";
  updateOcrButton();
}

// Stop waiting for the current OCR request. The server may still finish it, but its answer
// is ignored, because it no longer matches what's on screen.
function abandonOcr() {
  ocrRequestId++;
  setOcrBusy(false);
}

function showOcrNotice({ engine, warning }) {
  const engineLabel = ENGINE_LABELS[engine] || engine || "unknown";
  ocrNotice.innerHTML = `
    <p><strong>Check the extracted text against your paper document and fix any mistakes before translating.</strong></p>
    ${warning ? `<p class="notice-warn">${escapeHtml(warning)}</p>` : ""}
    <p class="notice-meta">Extracted with: ${escapeHtml(engineLabel)}</p>
    <button type="button" class="ghost-btn notice-close" data-action="dismiss">Dismiss</button>`;
  ocrNotice.hidden = false;
  ocrNotice.tabIndex = -1;
  ocrNotice.focus(); // screen readers read it; no keyboard pop-up on phones
}

function hideOcrNotice() {
  ocrNotice.hidden = true;
  ocrNotice.innerHTML = "";
}

function showOcrError(message) {
  $("ocrError").textContent = message;
  $("ocrError").hidden = false;
}

// --- input validation messages (shown under the Translate button) ---
function showInputError(message) {
  $("inputError").textContent = message;
  $("inputError").hidden = false;
}

function hideInputError() {
  $("inputError").hidden = true;
}

// --- translate ---
function selectedReadingLevel() {
  return document.querySelector('input[name="readingLevel"]:checked').value;
}

async function translate() {
  // The disabled button stops clicks, but not the Ctrl/Cmd+Enter shortcut, so check here too.
  // Each request is paid, and a second one could finish first and be overwritten by the older one.
  if (translating) return;
  hideInputError();
  if (ocrBusy || sampleLoading) {
    showInputError("Wait for the text to finish loading, then check it before translating.");
    return;
  }
  if (isTabSelected("tab-upload-btn")) {
    // The Paste tab is hidden here, so don't send text the user can't see
    if (!selectedFile) {
      showInputError("Choose a photo or PDF first, or switch to “Paste text” to translate typed text.");
      return;
    }
    if (!selectedFileRead) {
      showInputError("Click “Read text from file” first, then check the text before translating.");
      return;
    }
    selectTab($("tab-paste-btn")); // the file's text is in the box: show exactly what will be sent
  }
  const text = documentText.value.trim();
  if (!text) {
    showInputError("Paste a document, choose an example, or upload a photo first.");
    return;
  }

  const form = new FormData();
  form.append("document_text", text);
  form.append("reading_level", selectedReadingLevel());

  setTranslating(true);
  startLoading();
  try {
    const data = await request("/api/translate", { method: "POST", body: form }, TIMEOUTS_MS.translate);
    renderResult(data, text);
    announce("Your explanation is ready.");
  } catch (err) {
    showOutputError(err.message);
  } finally {
    stopLoading();
    setTranslating(false);
  }
}

function startLoading() {
  clearInterval(loadingTimer); // never leave an older counter running
  translateBtn.disabled = true;
  translateBtn.textContent = "Translating…";
  $("errorBanner").hidden = true;
  $("emptyState").hidden = true;
  $("resultState").hidden = true;
  $("loadingState").hidden = false;
  outputPanel.setAttribute("aria-busy", "true");

  // Honest timer: real seconds since the request started
  const startedAt = Date.now();
  $("elapsed").textContent = "0s";
  loadingTimer = setInterval(() => {
    $("elapsed").textContent = `${Math.floor((Date.now() - startedAt) / 1000)}s`;
  }, 1000);
  announce("Translating your document. This usually takes 15 to 40 seconds.");

  // On phones the output panel is below the button, so bring it into view
  if (window.matchMedia("(max-width: 860px)").matches) {
    outputPanel.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }
}

function stopLoading() {
  clearInterval(loadingTimer);
  loadingTimer = null;
  $("loadingState").hidden = true;
  outputPanel.removeAttribute("aria-busy");
  translateBtn.disabled = false;
  translateBtn.textContent = "Translate document →";
}

function showOutputError(message) {
  // startLoading only hid the earlier result, so it's still intact: show it again rather
  // than throwing away an explanation the user already has
  const keepPrevious = Boolean(lastResult);
  $("errorBanner").innerHTML =
    `<strong>Couldn't translate this document.</strong> ${escapeHtml(message)}` +
    (keepPrevious ? " Your previous explanation is still shown below." : "");
  $("errorBanner").hidden = false;
  $("resultState").hidden = !keepPrevious;
  $("emptyState").hidden = keepPrevious;
}

// --- results (callable from the console with mock data: renderResult({...})) ---
// sourceText is the text that was sent; it's compared with the textbox to spot later edits.
function renderResult(data, sourceText = data.source_text) {
  lastResult = data;
  lastResultText = typeof sourceText === "string" ? sourceText : null;
  const faithfulness = data.faithfulness || {};
  $("errorBanner").hidden = true;
  $("emptyState").hidden = true;
  $("loadingState").hidden = true;
  $("resultState").hidden = false;

  renderMetrics(data);
  renderExplanation(data);
  renderLabs(data.lab_values || []);
  renderSources(data.references || [], data.learn_more || []);
  renderFactCheck(faithfulness.numeric || {}, faithfulness.judge || {});
  renderResultMeta(data);
  updateStaleNotice();
  selectTab($("otab-explanation-btn"));
}

// --- metric chips ---
function renderMetrics(data) {
  const faithfulness = data.faithfulness || {};
  const readability = data.readability || {};
  const grade = (value) => (typeof value === "number" ? value.toFixed(1) : "—");
  const flagged = (data.lab_values || []).filter((lab) => lab.flag === "high" || lab.flag === "low").length;

  $("metricRow").innerHTML = [
    metricChip("Reading grade", `${grade(readability.source_grade_level)} → ${grade(readability.explanation_grade_level)}`),
    numbersChip(faithfulness.numeric || {}),
    claimsChip(faithfulness.judge || {}),
    metricChip("Flagged results", String(flagged)),
  ].join("");
}

function metricChip(label, value, tone = "neutral") {
  return `<li class="metric-chip tone-${tone}">${escapeHtml(label)}: <strong>${escapeHtml(value)}</strong></li>`;
}

function numbersChip(numeric) {
  const total = numeric.total_numbers_in_explanation || 0;
  const unsupported = (numeric.unsupported_numbers || []).length;
  if (!total) return metricChip("Numbers verified", "none to check");
  return metricChip("Numbers verified", `${Math.max(0, total - unsupported)}/${total}`, unsupported ? "warn" : "good");
}

function claimsChip(judge) {
  if (judge.error) return `<li class="metric-chip">Fact-check unavailable</li>`;
  const total = judge.total_claims_checked || 0;
  const supported = judge.supported_count || 0;
  const anyUnsupported = (judge.unsupported_claims || []).length > 0 || supported < total;
  if (!total) return metricChip("Claims verified", "none checked");
  return metricChip("Claims verified", `${supported}/${total}`, anyUnsupported ? "warn" : "good");
}

// --- explanation tab: markdown -> sanitized HTML -> citation links ---
function configureMarkdown() {
  if (!window.marked) return;
  // The explanation's "## Section" headings become <h3>, so they sit under the panel's <h2>
  marked.use({
    renderer: {
      heading(text, level) {
        const tag = `h${Math.min(Math.max(level + 1, 3), 6)}`;
        return `<${tag}>${text}</${tag}>\n`;
      },
    },
  });
}

// Only plain text formatting is allowed through: no images, forms, styles or scripts
const SANITIZE_OPTIONS = {
  ALLOWED_TAGS: ["h2", "h3", "h4", "h5", "h6", "p", "br", "hr", "ul", "ol", "li", "strong", "em", "b", "i",
    "a", "blockquote", "code", "pre", "table", "thead", "tbody", "tr", "th", "td", "sup", "sub", "del"],
  ALLOWED_ATTR: ["href", "title"],
};

function markdownToSafeHtml(markdown) {
  if (!window.marked || !window.DOMPurify) {
    // The CDN scripts didn't load: show plain escaped text rather than unsanitized HTML
    return `<p class="plain-text">${escapeHtml(markdown)}</p>`;
  }
  return DOMPurify.sanitize(marked.parse(markdown), SANITIZE_OPTIONS);
}

function renderExplanation(data) {
  const explanation = (data.explanation || "").trim();
  explanationBody.innerHTML = explanation
    ? markdownToSafeHtml(explanation)
    : `<p class="empty-note">The server returned an empty explanation. Please try again.</p>`;
  linkCitations(explanationBody, data.references || []);

  // Links written by the model open in a new tab
  explanationBody.querySelectorAll('a[href^="http"]').forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
  $("printLevel").textContent = LEVEL_LABELS[data.reading_level] ? `(${LEVEL_LABELS[data.reading_level]})` : "";
}

// Citation markers look like [1] or [2, 3]; ids refer to references[].id
const CITATION_PATTERN = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function linkCitations(container, references) {
  const titles = new Map(references.map((ref) => [String(ref.id), ref.title]));
  if (!titles.size) return;

  // Collect text nodes first, then replace them (editing while walking confuses the walker)
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node.parentElement.closest("a, code, pre") && node.nodeValue.match(CITATION_PATTERN)) {
      textNodes.push(node);
    }
  }
  textNodes.forEach((node) => node.replaceWith(buildCitationFragment(node.nodeValue, titles)));
}

function buildCitationFragment(text, titles) {
  const fragment = document.createDocumentFragment();
  let lastIndex = 0;
  for (const match of text.matchAll(CITATION_PATTERN)) {
    const ids = match[1].split(",").map((id) => id.trim());
    if (!ids.some((id) => titles.has(id))) continue; // no matching sources: leave the text alone
    // Move the space before the marker into the citation, so hiding citations (print) leaves no gap
    const before = text.slice(lastIndex, match.index);
    const trimmed = before.replace(/\s+$/, "");
    fragment.append(trimmed, buildCitation(ids, titles, trimmed !== before));
    lastIndex = match.index + match[0].length;
  }
  fragment.append(text.slice(lastIndex));
  return fragment;
}

// Built with DOM methods (not HTML strings), so nothing here needs escaping
function buildCitation(ids, titles, leadingSpace) {
  const sup = document.createElement("sup");
  sup.className = "cite";
  sup.append(leadingSpace ? " [" : "[");
  ids.forEach((id, index) => {
    if (index > 0) sup.append(", ");
    if (!titles.has(id)) {
      sup.append(id); // unknown id: show it, but don't link it
      return;
    }
    const link = document.createElement("a");
    link.className = "cite-link";
    link.href = `#ref-${id}`;
    link.dataset.ref = id;
    link.textContent = id;
    link.setAttribute("aria-label", `Source ${id}: ${titles.get(id)}`);
    sup.append(link);
  });
  sup.append("]");
  return sup;
}

// Clicking a citation switches to Sources, scrolls to the card, and highlights it
function jumpToReference(id) {
  const card = $(`ref-${id}`);
  if (!card) return;
  selectTab($("otab-sources-btn"));
  card.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "center" });
  card.focus({ preventScroll: true });
  card.classList.remove("flash");
  void card.offsetWidth; // restart the highlight animation if it is already running
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1600);
}

// --- copy + print ---
function explanationAsPlainText() {
  let text = explanationBody.innerText.trim();
  const refs = (lastResult && lastResult.references) || [];
  if (refs.length) {
    text += "\n\nSources (MedlinePlus):\n" + refs.map((ref) => `[${ref.id}] ${ref.title} — ${ref.url}`).join("\n");
  }
  // Copied text leaves the app, so it carries the fact-check warnings with it
  const faithfulness = (lastResult && lastResult.faithfulness) || {};
  const issues = factCheckIssues(faithfulness.numeric || {}, faithfulness.judge || {});
  const lines = [];
  if (issues.judgeFailed) lines.push("- The automatic fact-check couldn't run for this explanation.");
  if (issues.numbers.length) lines.push(`- Numbers not found in your document: ${issues.numbers.join(", ")}`);
  issues.claims.forEach((claim) => lines.push(`- Not backed up by your document (${claimKindLabel(claim)}): "${claim.sentence}"`));
  if (lines.length) text += "\n\nCheck these with your care team:\n" + lines.join("\n");
  return `${text}\n\n${COPY_DISCLAIMER}`;
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for older browsers and plain-http pages (e.g. testing on a phone over Wi-Fi)
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.className = "visually-hidden";
    document.body.append(area);
    area.select();
    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      area.remove();
    }
  }
}

async function copyExplanation() {
  if (!lastResult) return;
  const copied = await copyToClipboard(explanationAsPlainText());
  $("copyStatus").textContent = copied ? "Copied." : "Couldn't copy. Select the text and copy it manually.";
  $("copyStatus").classList.toggle("warn", !copied);
  clearTimeout(copyStatusTimer);
  copyStatusTimer = setTimeout(() => ($("copyStatus").textContent = ""), 4000);
}

// --- lab results tab ---
function renderLabs(labs) {
  setTabCount("otab-labs-btn", labs.length, "values");
  if (!labs.length) {
    $("otab-labs").innerHTML = `<p class="empty-note">No structured lab values were detected in this document.</p>`;
    return;
  }
  $("otab-labs").innerHTML = `
    <p class="panel-intro">${labSummary(labs)}</p>
    <div class="table-wrap">
      <table class="lab-table">
        <caption class="visually-hidden">Lab results found in your document</caption>
        <thead>
          <tr><th scope="col">Test</th><th scope="col">Result</th><th scope="col">Reference range</th><th scope="col">Flag</th></tr>
        </thead>
        <tbody>${labs.map(labRow).join("")}</tbody>
      </table>
    </div>`;
}

function hasRange(lab) {
  const has = (n) => n !== null && n !== undefined;
  return Boolean(lab.range_text) || has(lab.ref_low) || has(lab.ref_high);
}

// The API flags a result "unknown" both when there's no range and when there is one but the
// value can't be placed in it (like "<15", or a lab "*" marker), so tell those apart here
function flagKey(lab) {
  if (lab.flag === "high" || lab.flag === "low" || lab.flag === "normal") return lab.flag;
  return hasRange(lab) ? "unclear" : "unknown";
}

function labSummary(labs) {
  const counts = { high: 0, low: 0, normal: 0, unclear: 0, unknown: 0 };
  labs.forEach((lab) => counts[flagKey(lab)]++);
  const parts = [];
  if (counts.high) parts.push(`<strong class="text-high">${counts.high} high</strong>`);
  if (counts.low) parts.push(`<strong class="text-low">${counts.low} low</strong>`);
  if (counts.normal) parts.push(`${counts.normal} in range`);
  if (counts.unclear) parts.push(`${counts.unclear} that can't be compared with the range`);
  if (counts.unknown) parts.push(`${counts.unknown} without a range`);
  let summary = `${parts.join(" · ")}. Each result is compared with its reference range.`;
  if (counts.unclear) {
    summary += ` “Can't compare” means the result is written as a limit (like &lt;15) or was marked by
      the lab, so we can't say whether it's in range. Ask your care team.`;
  }
  return summary;
}

function labRow(lab) {
  const flag = flagKey(lab);
  const value = lab.value_text ?? lab.value;
  const result = [value, lab.unit].filter((part) => part !== null && part !== undefined && part !== "").join(" ");
  const range = lab.range_text || rangeFromBounds(lab.ref_low, lab.ref_high) || "—";
  const note = RANGE_SOURCE_NOTES[lab.range_source];
  return `
    <tr class="flag-${flag}">
      <th scope="row" class="cell-test">${escapeHtml(lab.name)}</th>
      <td class="cell-result" data-label="Result">${escapeHtml(result || "—")}</td>
      <td class="cell-range" data-label="Reference range">
        <span class="range-text">${escapeHtml(range)}</span>
        ${note ? `<span class="range-note">${escapeHtml(note)}</span>` : ""}
      </td>
      <td class="cell-flag">${flagBadge(flag)}</td>
    </tr>`;
}

function rangeFromBounds(low, high) {
  const has = (n) => n !== null && n !== undefined;
  if (has(low) && has(high)) return `${low}–${high}`;
  if (has(high)) return `up to ${high}`;
  if (has(low)) return `at least ${low}`;
  return "";
}

function flagBadge(flag) {
  const { label, icon } = FLAG_BADGES[flag];
  const iconHtml = icon ? `<span aria-hidden="true">${icon}</span> ` : "";
  return `<span class="flag-badge flag-badge-${flag}">${iconHtml}${label}</span>`;
}

// --- sources tab (MedlinePlus) ---
function renderSources(references, learnMore) {
  setTabCount("otab-sources-btn", references.length + learnMore.length, "pages");
  let html = "";
  if (references.length) {
    html += `<h3 class="section-title">References</h3>
      <ul class="ref-list" role="list">${references.map(referenceCard).join("")}</ul>`;
  }
  if (learnMore.length) {
    html += `<h3 class="section-title">Learn more</h3>
      <p class="section-note">Related MedlinePlus pages for further reading.</p>
      <ul class="ref-list" role="list">${learnMore.map(learnMoreCard).join("")}</ul>`;
  }
  if (!html) html = `<p class="empty-note">No MedlinePlus pages matched this document.</p>`;
  html += `<p class="source-footer">Source: MedlinePlus, U.S. National Library of Medicine.</p>`;
  $("otab-sources").innerHTML = html;
}

function referenceCard(ref) {
  return `
    <li class="ref-card" id="ref-${escapeHtml(ref.id)}" tabindex="-1">
      <div class="ref-head">
        <span class="ref-num">[${escapeHtml(ref.id)}]</span>
        <span class="ref-kind">${escapeHtml(SOURCE_KIND_LABELS[ref.kind] || "MedlinePlus")}</span>
        ${ref.cited ? `<span class="ref-cited">Cited in explanation</span>` : ""}
      </div>
      ${sourceBody(ref)}
    </li>`;
}

function learnMoreCard(item) {
  return `
    <li class="ref-card">
      <div class="ref-head">
        <span class="ref-kind">${escapeHtml(SOURCE_KIND_LABELS[item.kind] || "MedlinePlus")}</span>
      </div>
      ${sourceBody(item)}
    </li>`;
}

function sourceBody(item) {
  return `
    <a class="ref-title" href="${escapeHtml(safeUrl(item.url))}" target="_blank" rel="noopener noreferrer">
      ${escapeHtml(item.title)}<span aria-hidden="true"> ↗</span><span class="visually-hidden"> (opens in a new tab)</span>
    </a>
    ${item.snippet ? `<p class="ref-snippet">${escapeHtml(item.snippet)}</p>` : ""}
    ${item.matched ? `<p class="ref-matched">Matched: ${escapeHtml(item.matched)}</p>` : ""}`;
}

// --- fact check tab ---
function renderFactCheck(numeric, judge) {
  const issues = (numeric.unsupported_numbers || []).length + (judge.unsupported_claims || []).length;
  setTabCount("otab-check-btn", issues, issues === 1 ? "issue" : "issues", true);
  $("otab-check").innerHTML = numericSection(numeric) + judgeSection(judge);
  renderPrintCheck(numeric, judge);
}

// Statements about the patient matter most, so they're listed first
const patientClaimsFirst = (a, b) => (b.kind === "patient_specific") - (a.kind === "patient_specific");

function claimKindLabel(claim) {
  return CLAIM_KIND_LABELS[claim.kind] || "General info";
}

// The problems the fact check found, shared by the printout and "Copy text"
function factCheckIssues(numeric, judge) {
  return {
    judgeFailed: Boolean(judge.error),
    numbers: numeric.unsupported_numbers || [],
    claims: [...(judge.unsupported_claims || [])].sort(patientClaimsFirst),
  };
}

// The Fact check tab isn't printed, so a short box under the explanation repeats any problems
// (it stays hidden when everything checked out)
function renderPrintCheck(numeric, judge) {
  const issues = factCheckIssues(numeric, judge);
  const parts = [];
  if (issues.judgeFailed) parts.push(`<p>The automatic fact-check couldn't run for this explanation.</p>`);
  if (issues.numbers.length) {
    parts.push(`<p><strong>Numbers not found in your document:</strong> ${issues.numbers.map(escapeHtml).join(", ")}</p>`);
  }
  if (issues.claims.length) {
    const items = issues.claims.map((claim) =>
      `<li>“${escapeHtml(claim.sentence)}” <em>(${escapeHtml(claimKindLabel(claim))})</em></li>`);
    parts.push(`<p><strong>Statements your document doesn't back up:</strong></p><ul>${items.join("")}</ul>`);
  }
  $("printCheck").innerHTML = parts.length ? `<h3>Check these with your care team</h3>${parts.join("")}` : "";
  $("printCheck").hidden = !parts.length;
}

function checkBox(kind, html) {
  return `<div class="check-box check-${kind}">${html}</div>`;
}

function numberList(numbers) {
  return numbers.map((n) => `<span class="num">${escapeHtml(n)}</span>`).join(", ");
}

function numericSection(numeric) {
  const total = numeric.total_numbers_in_explanation || 0;
  const unsupported = numeric.unsupported_numbers || [];
  const fromRanges = numeric.numbers_from_reference_ranges || [];
  let html = `<h3 class="section-title">Numbers</h3>
    <p class="section-note">Every number in the explanation was looked up in your document.</p>`;

  if (!total) {
    html += checkBox("info", "The explanation doesn't contain any numbers to check.");
  } else if (!unsupported.length) {
    const rangeNote = fromRanges.length ? " or to the typical ranges the app added (below)" : "";
    html += checkBox("ok", `All numbers in the explanation trace back to your document${rangeNote}.`);
  } else {
    html += checkBox(
      "flag",
      `<strong>These numbers in the explanation weren't found in your document:</strong> ${numberList(unsupported)}.
       Double-check them against the original.`
    );
  }
  if (fromRanges.length) {
    html += checkBox("info", `${numberList(fromRanges)} — from the typical ranges the app added, not your document.`);
  }
  return html;
}

function judgeSection(judge) {
  let html = `<h3 class="section-title">Claims</h3>
    <p class="section-note">A second AI pass checked each statement in the explanation against your document.</p>`;
  if (judge.error) {
    return html + checkBox("flag", `<strong>The fact-check couldn't run this time.</strong> ${escapeHtml(judge.error)}`);
  }
  const total = judge.total_claims_checked || 0;
  const supported = judge.supported_count || 0;
  const claims = judge.unsupported_claims || [];
  if (!claims.length) {
    const message = total ? `All ${total} claims checked were supported by your document.` : "No claims were checked.";
    return html + checkBox(total ? "ok" : "info", message);
  }
  const ordered = [...claims].sort(patientClaimsFirst);
  return `${html}
    <p class="check-summary">${supported} of ${total} claims were supported. These were not:</p>
    <ul class="claim-list" role="list">${ordered.map(claimItem).join("")}</ul>
    <p class="section-note"><strong>About you</strong> means a statement about your results or care that your
      document doesn't back up. <strong>General info</strong> means a general health statement that isn't in your
      document. It may be true, but check it with your care team.</p>`;
}

function claimItem(claim) {
  const kind = claim.kind === "patient_specific" ? "patient" : "general";
  return `
    <li class="claim-item claim-${kind}">
      <span class="claim-kind">${escapeHtml(claimKindLabel(claim))}</span>
      <blockquote class="claim-sentence">${escapeHtml(claim.sentence)}</blockquote>
      ${claim.reason ? `<p class="claim-reason">${escapeHtml(claim.reason)}</p>` : ""}
    </li>`;
}

// --- small footer line: level, model, time ---
function renderResultMeta(data) {
  const parts = [];
  if (LEVEL_LABELS[data.reading_level]) parts.push(`Explained at ${LEVEL_LABELS[data.reading_level]}`);
  if (data.model) parts.push(data.model);
  const totalMs = data.timing_ms && data.timing_ms.total;
  if (typeof totalMs === "number") parts.push(`${(totalMs / 1000).toFixed(1)} s`);
  $("resultMeta").textContent = parts.join(" · ");
}

// --- wire everything up ---
function init() {
  setupDemoNotice(); // first, so the notice is in place before anything else renders
  configureMarkdown();
  setupTabs(document.querySelector(".input-tabs"));
  setupTabs(document.querySelector(".output-tabs"));
  setupDropZone();

  documentText.addEventListener("input", () => {
    sampleSelect.value = ""; // the text no longer matches the example, so it can be picked again
    updateCharCount();
    updateStaleNotice();
    hideInputError();
  });
  documentText.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !(event.metaKey || event.ctrlKey)) return; // Ctrl/Cmd + Enter shortcut
    event.preventDefault();
    if (!event.repeat) translate(); // ignore auto-repeat while the keys are held down
  });
  sampleSelect.addEventListener("change", () => loadSample(sampleSelect.value));
  fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0] || null));
  cameraInput.addEventListener("change", () => setSelectedFile(cameraInput.files[0] || null));
  $("fileThumb").addEventListener("error", () => ($("fileThumb").hidden = true)); // e.g. HEIC in Chrome
  $("clearFileBtn").addEventListener("click", () => setSelectedFile(null));
  ocrBtn.addEventListener("click", readTextFromFile);
  ocrNotice.addEventListener("click", (event) => {
    if (!event.target.closest('[data-action="dismiss"]')) return;
    hideOcrNotice();
    documentText.focus();
  });
  translateBtn.addEventListener("click", translate);
  explanationBody.addEventListener("click", (event) => {
    const link = event.target.closest("a.cite-link");
    if (!link) return;
    event.preventDefault();
    jumpToReference(link.dataset.ref);
  });
  $("copyBtn").addEventListener("click", copyExplanation);
  $("printBtn").addEventListener("click", () => window.print());

  updateCharCount();
  updateSampleSelect(); // stays disabled until the examples have loaded
  checkHealth();
  loadSampleList();
}

init();

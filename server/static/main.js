/* g4h live-demo frontend. Form submit → POST /predict → render result. */

const $ = (sel) => document.querySelector(sel);

const urlInput = $("#url");
const fetchDom = $("#fetch-dom");
const button = $("#classify");
const status = $("#status");
const result = $("#result");
const verdictLabel = $("#verdict-label");
const scoresEl = $("#scores");
const indicatorsEl = $("#indicators");
const fetchMetaBlock = $("#fetch-meta-block");
const fetchMetaEl = $("#fetch-meta");
const modelInputEl = $("#model-input");

const INDICATOR_CATEGORIES = ["url", "security", "domain", "intent", "content", "hosting", "fetch", "meta"];

function categoryFor(indicator) {
  // indicator looks like "category:name:{...}"
  const head = indicator.split(":")[0];
  return INDICATOR_CATEGORIES.includes(head) ? head : "meta";
}

function escape(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function renderScores(scores, topLabel) {
  scoresEl.innerHTML = "";
  for (const [label, value] of Object.entries(scores)) {
    const cell = document.createElement("div");
    cell.className = "score-cell" + (label === topLabel ? " top" : "");
    cell.innerHTML = `<div class="score-value">${(value * 100).toFixed(1)}%</div><div class="score-label">${escape(label)}</div>`;
    scoresEl.appendChild(cell);
  }
}

function renderIndicators(indicators) {
  indicatorsEl.innerHTML = "";
  if (!indicators || indicators.length === 0) {
    const span = document.createElement("span");
    span.className = "indicator meta";
    span.textContent = "(none)";
    indicatorsEl.appendChild(span);
    return;
  }
  for (const ind of indicators) {
    const span = document.createElement("span");
    span.className = "indicator " + categoryFor(ind);
    span.textContent = ind;
    indicatorsEl.appendChild(span);
  }
}

function renderFetchMeta(meta) {
  if (!meta) {
    fetchMetaBlock.classList.add("hidden");
    return;
  }
  fetchMetaBlock.classList.remove("hidden");
  fetchMetaEl.innerHTML = "";

  const entries = [
    ["status", meta.status ?? "—"],
    ["final url", meta.final_url ?? "—"],
    ["title", meta.title ?? "—"],
    ["error", meta.error ?? "—"],
  ];
  if (meta.response_headers) {
    for (const [k, v] of Object.entries(meta.response_headers)) {
      entries.push([k, v]);
    }
  }
  for (const [k, v] of entries) {
    const key = document.createElement("div");
    key.className = "key";
    key.textContent = k;
    const val = document.createElement("div");
    val.className = "val";
    val.textContent = String(v);
    fetchMetaEl.appendChild(key);
    fetchMetaEl.appendChild(val);
  }
}

async function classify() {
  const url = urlInput.value.trim();
  if (!url) {
    status.textContent = "ENTER A URL FIRST.";
    status.className = "status-line error";
    return;
  }
  button.disabled = true;
  result.classList.add("hidden");
  status.className = "status-line working";
  status.textContent = fetchDom.checked
    ? "FETCHING DOM AND CLASSIFYING (≤10s) ..."
    : "EXTRACTING AND CLASSIFYING ...";

  try {
    const t0 = performance.now();
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, fetch_dom: fetchDom.checked }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text || resp.statusText}`);
    }
    const data = await resp.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    verdictLabel.textContent = data.label;
    verdictLabel.className = "verdict-tag " + data.label;
    renderScores(data.scores, data.label);
    renderIndicators(data.indicators);
    renderFetchMeta(data.fetch_meta);
    modelInputEl.textContent = data.indicator_text || "(empty)";

    result.classList.remove("hidden");
    status.className = "status-line";
    status.textContent = `DONE IN ${elapsed}s · ${data.indicators?.length || 0} INDICATORS`;
  } catch (e) {
    status.className = "status-line error";
    status.textContent = "ERROR: " + e.message;
  } finally {
    button.disabled = false;
  }
}

button.addEventListener("click", classify);
urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") classify();
});

const statusPill = document.getElementById("status-pill");
const indexCount = document.getElementById("index-count");
const indexPath = document.getElementById("index-path");
const refreshStatsBtn = document.getElementById("refresh-stats");

const ingestForm = document.getElementById("ingest-form");
const ingestOutput = document.getElementById("ingest-output");
const sourceDir = document.getElementById("source-dir");
const patterns = document.getElementById("patterns");
const persist = document.getElementById("persist");

const searchForm = document.getElementById("search-form");
const searchQuery = document.getElementById("search-query");
const searchTopk = document.getElementById("search-topk");
const searchResults = document.getElementById("search-results");

const answerForm = document.getElementById("answer-form");
const answerQuery = document.getElementById("answer-query");
const answerTopk = document.getElementById("answer-topk");
const answerText = document.getElementById("answer-text");
const answerSources = document.getElementById("answer-sources");
const ingestProgressBar = document.getElementById("ingest-progress-bar");
const ingestProgressText = document.getElementById("ingest-progress-text");
const ingestButton = ingestForm.querySelector("button[type='submit']");
const flowSteps = Array.from(document.querySelectorAll("[data-step]"));
let ingestPoller = null;
let ingestJobId = null;

const INGEST_STEPS = ["ingest", "chunk", "embed", "index"];

function clearFlow() {
  flowSteps.forEach((step) => {
    step.classList.remove("is-active", "is-done", "is-error");
  });
}

function setFlowState({ active = [], done = [], error = [] } = {}) {
  clearFlow();
  flowSteps.forEach((step) => {
    const key = step.dataset.step;
    if (done.includes(key)) step.classList.add("is-done");
    if (active.includes(key)) step.classList.add("is-active");
    if (error.includes(key)) step.classList.add("is-error");
  });
}

function setIngestFlow(step, status) {
  if (!step) return;
  if (status === "completed") {
    setFlowState({ done: INGEST_STEPS });
    return;
  }
  if (status === "error") {
    setFlowState({ error: [step] });
    return;
  }
  const stepIndex = INGEST_STEPS.indexOf(step);
  if (stepIndex === -1) {
    setFlowState({ active: ["ingest"] });
    return;
  }
  const done = INGEST_STEPS.slice(0, stepIndex);
  setFlowState({ done, active: [step] });
}

function updateIngestProgress(progress, message) {
  const safe = Math.max(0, Math.min(100, Math.round(progress || 0)));
  ingestProgressBar.style.width = `${safe}%`;
  ingestProgressText.textContent = `${safe}%`;
  if (message) {
    ingestOutput.textContent = message;
  }
}

async function pollIngest(jobId) {
  try {
    const job = await fetchJson(`/ingest/${jobId}`, { method: "GET" });
    updateIngestProgress(job.progress, job.message);
    setIngestFlow(job.step, job.status);

    if (job.status === "completed") {
      ingestButton.disabled = false;
      if (job.result) {
        ingestOutput.textContent = JSON.stringify(job.result, null, 2);
      }
      await refreshStats();
      clearInterval(ingestPoller);
      ingestPoller = null;
    }

    if (job.status === "error") {
      ingestButton.disabled = false;
      ingestOutput.textContent = job.error || job.message || "Ingestion failed.";
      clearInterval(ingestPoller);
      ingestPoller = null;
    }
  } catch (err) {
    ingestButton.disabled = false;
    ingestOutput.textContent = `Error: ${err.message}`;
    setFlowState({ error: ["ingest"] });
    clearInterval(ingestPoller);
    ingestPoller = null;
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Request failed." }));
    throw new Error(err.detail || "Request failed.");
  }
  return response.json();
}

function setStatus(message, ok = true) {
  statusPill.textContent = message;
  statusPill.style.background = ok ? "rgba(45, 93, 123, 0.1)" : "rgba(160, 60, 60, 0.12)";
  statusPill.style.color = ok ? "#2d5d7b" : "#a03c3c";
}

function renderSearchResults(results) {
  searchResults.innerHTML = "";
  if (!results || results.length === 0) {
    searchResults.innerHTML = '<p class="muted">No results found.</p>';
    return;
  }

  results.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "result-item";
    const text = item.text || "";
    const snippet = text.length > 360 ? `${text.slice(0, 360)}...` : text;
    const meta = item.metadata || {};
    card.innerHTML = `
      <strong>#${idx + 1}</strong>
      <div>${snippet.replace(/\n/g, "<br />")}</div>
      <div class="result-meta">source: ${meta.source || "unknown"} | chunk: ${meta.chunk_id ?? "?"} | score: ${item.score?.toFixed?.(4) ?? item.score}</div>
    `;
    searchResults.appendChild(card);
  });
}

function renderSources(sources) {
  answerSources.innerHTML = "";
  if (!sources || sources.length === 0) {
    answerSources.innerHTML = '<li class="muted">No sources available.</li>';
    return;
  }
  sources.forEach((src) => {
    const li = document.createElement("li");
    li.textContent = `${src.source || "unknown"} (chunk ${src.chunk_id ?? "?"})`;
    answerSources.appendChild(li);
  });
}

async function refreshStats() {
  try {
    const stats = await fetchJson("/stats", { method: "GET" });
    indexCount.textContent = stats.indexed_vectors ?? 0;
    indexPath.textContent = stats.persist_path || "storage/faiss_index";
    setStatus("API: Ready", true);
  } catch (err) {
    setStatus("API: Offline", false);
  }
}

refreshStatsBtn.addEventListener("click", refreshStats);

window.addEventListener("load", () => {
  refreshStats();
});

ingestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  ingestOutput.textContent = "Queueing ingestion...";
  updateIngestProgress(0);
  setFlowState({ active: ["ingest"] });
  ingestButton.disabled = true;
  const patternList = patterns.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  const payload = {
    source_dir: sourceDir.value.trim() || "data",
    patterns: patternList.length ? patternList : null,
    persist: persist.checked,
  };

  try {
    const result = await fetchJson("/ingest", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    ingestJobId = result.job_id;
    updateIngestProgress(1, "Ingestion started...");
    if (ingestPoller) clearInterval(ingestPoller);
    ingestPoller = setInterval(() => pollIngest(ingestJobId), 1000);
    await pollIngest(ingestJobId);
  } catch (err) {
    ingestOutput.textContent = `Error: ${err.message}`;
    setFlowState({ error: ["ingest"] });
    ingestButton.disabled = false;
  }
});

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    query: searchQuery.value.trim(),
    top_k: Number(searchTopk.value || 5),
  };

  if (!payload.query) {
    searchResults.innerHTML = '<p class="muted">Enter a search query.</p>';
    return;
  }

  try {
    setFlowState({ active: ["retrieve"] });
    const result = await fetchJson("/search", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderSearchResults(result.results);
    setFlowState({ done: ["retrieve"] });
  } catch (err) {
    searchResults.innerHTML = `<p class="muted">Error: ${err.message}</p>`;
    setFlowState({ error: ["retrieve"] });
  }
});

answerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    query: answerQuery.value.trim(),
    top_k: Number(answerTopk.value || 5),
  };

  if (!payload.query) {
    answerText.textContent = "Enter a question to get an answer.";
    renderSources([]);
    return;
  }

  answerText.textContent = "Generating answer...";
  renderSources([]);
  setFlowState({ active: ["retrieve", "generate"] });

  try {
    const result = await fetchJson("/answer", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    answerText.textContent = result.answer || "No answer returned.";
    renderSources(result.sources || []);
    setFlowState({ done: ["retrieve", "generate"] });
  } catch (err) {
    answerText.textContent = `Error: ${err.message}`;
    renderSources([]);
    setFlowState({ error: ["retrieve", "generate"] });
  }
});

const API_BASE = ""; // same-origin; change if backend is deployed separately

let selectedLevel = "8th";
let selectedFile = null;

// --- input tab switching ---
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

// --- reading level dial ---
document.querySelectorAll(".dial-opt").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".dial-opt").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    selectedLevel = btn.dataset.level;
  });
});

// --- file upload / camera capture ---
const fileInput = document.getElementById("fileInput");
fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  document.getElementById("fileName").textContent = selectedFile ? selectedFile.name : "";
});

// --- output tab switching ---
document.querySelectorAll(".otab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".otab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".otab-panel").forEach((p) => p.classList.add("hidden"));
    document.getElementById(`otab-${btn.dataset.otab}`).classList.remove("hidden");
  });
});

// --- main translate action ---
document.getElementById("translateBtn").addEventListener("click", async () => {
  const activeInputTab = document.querySelector(".tab-btn.active").dataset.tab;
  const documentText = document.getElementById("documentText").value;

  if (activeInputTab === "paste" && !documentText.trim()) {
    alert("Paste a document first.");
    return;
  }
  if (activeInputTab === "upload" && !selectedFile) {
    alert("Choose or take a photo first.");
    return;
  }

  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("resultState").classList.add("hidden");
  document.getElementById("loadingState").classList.remove("hidden");
  document.getElementById("translateBtn").disabled = true;

  const formData = new FormData();
  formData.append("reading_level", selectedLevel);
  if (activeInputTab === "paste") {
    formData.append("document_text", documentText);
  } else {
    formData.append("file", selectedFile);
  }

  try {
    const res = await fetch(`${API_BASE}/api/translate`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    document.getElementById("loadingState").classList.add("hidden");
    document.getElementById("emptyState").classList.remove("hidden");
    alert(`Something went wrong: ${e.message}`);
  } finally {
    document.getElementById("translateBtn").disabled = false;
  }
});

function renderResult(data) {
  document.getElementById("loadingState").classList.add("hidden");
  document.getElementById("resultState").classList.remove("hidden");

  // metrics
  const numericScore = data.faithfulness.numeric.numeric_faithfulness_score;
  const flaggedCount = (data.lab_values || []).filter((l) => l.flag !== "normal").length;
  document.getElementById("metricRow").innerHTML = `
    <span class="metric-chip">Source grade level: <strong>${data.readability.source_grade_level}</strong></span>
    <span class="metric-chip">Explanation grade level: <strong>${data.readability.explanation_grade_level}</strong></span>
    <span class="metric-chip">Numeric faithfulness: <strong>${(numericScore * 100).toFixed(0)}%</strong></span>
    <span class="metric-chip">Flagged values: <strong>${flaggedCount}</strong></span>
  `;

  // explanation
  document.getElementById("otab-explanation").innerHTML = marked.parse(data.explanation);

  // labs
  if (!data.lab_values || data.lab_values.length === 0) {
    document.getElementById("otab-labs").innerHTML =
      "<p>No structured lab values were detected in this document.</p>";
  } else {
    document.getElementById("otab-labs").innerHTML = data.lab_values
      .map(
        (l) => `
      <div class="lab-row flag-${l.flag}">
        <span>${escapeHtml(l.name)}: ${l.value} ${l.unit || ""}</span>
        <span class="lab-flag-tag">${l.flag}</span>
      </div>`
      )
      .join("");
  }

  // faithfulness detail
  const unsupportedNums = data.faithfulness.numeric.unsupported_numbers;
  const judge = data.faithfulness.judge;
  let checkHtml = "";
  if (unsupportedNums.length > 0) {
    checkHtml += `<div class="claim-flag"><strong>Numbers in the explanation not found in the source:</strong> ${unsupportedNums.join(", ")}</div>`;
  } else {
    checkHtml += `<div class="claim-ok">All numbers in the explanation trace back to the source document.</div>`;
  }
  if (judge.unsupported_claims && judge.unsupported_claims.length > 0) {
    checkHtml += judge.unsupported_claims
      .map((c) => `<div class="claim-flag">${escapeHtml(c)}</div>`)
      .join("");
  } else if (!judge.judge_parse_error) {
    checkHtml += `<div class="claim-ok">The fact-checking pass found no unsupported claims (${judge.supported_count}/${judge.total_claims_checked} claims verified).</div>`;
  }
  document.getElementById("otab-check").innerHTML = checkHtml;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

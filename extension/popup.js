const BRIDGE_URL = "http://127.0.0.1:8765";

let currentSessionId = null;

// Initialize Session
function updateSessionDisplay() {
  const lbl = document.getElementById("sessionLabel");
  if (currentSessionId) {
    lbl.textContent = "Session: " + currentSessionId.slice(0, 8);
  } else {
    lbl.textContent = "Session: New";
  }
}

chrome.storage.local.get(["sessionId", "profile"], (res) => {
  if (res.sessionId) {
    currentSessionId = res.sessionId;
    updateSessionDisplay();
  }
  if (res.profile) {
    const profSelect = document.getElementById("profileSelect");
    if (profSelect) profSelect.value = res.profile;
  }
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.sessionId) {
    currentSessionId = changes.sessionId.newValue;
    updateSessionDisplay();
  }
});

// Profile change listener
document.getElementById("profileSelect").addEventListener("change", (e) => {
  chrome.storage.local.set({ profile: e.target.value });
});

function getActiveProfile() {
  const sel = document.getElementById("profileSelect");
  return sel ? sel.value : "balanced";
}

// Reset / New Session
document.getElementById("newSessionBtn").addEventListener("click", () => {
  currentSessionId = null;
  chrome.storage.local.remove(["sessionId"], () => {
    updateSessionDisplay();
    flashBtn("newSessionBtn", "Reset!");
  });
});

// Check Server Health
async function checkHealth() {
  const badge = document.getElementById("statusBadge");
  const text = document.getElementById("statusText");
  text.textContent = "Checking...";
  try {
    const res = await fetch(`${BRIDGE_URL}/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      badge.className = "status-badge";
      text.textContent = `Online (${data.profile || "ready"})`;
    } else {
      throw new Error("HTTP " + res.status);
    }
  } catch (err) {
    badge.className = "status-badge offline";
    text.textContent = "Offline (Start Server)";
  }
}

document.getElementById("statusBadge").addEventListener("click", checkHealth);

// Tab Switching
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// Tab 1: Redact Text
document.getElementById("redactBtn").addEventListener("click", async () => {
  const input = document.getElementById("promptInput").value;
  if (!input.trim()) return;

  const btn = document.getElementById("redactBtn");
  btn.textContent = "⏳ Protecting...";
  btn.disabled = true;

  try {
    const res = await fetch(`${BRIDGE_URL}/redact_text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: input,
        session_id: currentSessionId,
        profile: getActiveProfile(),
      }),
    });

    if (!res.ok) throw new Error("Server returned HTTP " + res.status);

    const data = await res.json();
    currentSessionId = data.session_id;
    chrome.storage.local.set({ sessionId: currentSessionId });
    updateSessionDisplay();

    const resultBox = document.getElementById("redactedResult");
    resultBox.textContent = data.redacted_text;
    resultBox.style.display = "block";

    // Show Chips
    const chipsContainer = document.getElementById("entityChips");
    chipsContainer.innerHTML = "";
    if (data.detected_pii && data.detected_pii.length > 0) {
      data.detected_pii.forEach((item) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = `${item.placeholder}: ${item.masked}`;
        chipsContainer.appendChild(chip);
      });
    } else {
      chipsContainer.innerHTML = "<span style='font-size:11px;color:#10b981;'>✅ No PII detected (Text is clean)</span>";
    }
  } catch (err) {
    alert("Could not reach PII Shield local bridge.\n\nMake sure 'python extension_server/server.py' is running.");
  } finally {
    btn.textContent = "🛡️ Redact PII";
    btn.disabled = false;
  }
});

// Copy Redacted
document.getElementById("copyRedactedBtn").addEventListener("click", () => {
  const text = document.getElementById("redactedResult").textContent;
  if (!text) return;
  navigator.clipboard.writeText(text);
  flashBtn("copyRedactedBtn", "Copied!");
});

// Tab 2: Document Sanitize
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

dropzone.addEventListener("click", () => fileInput.click());

// Drag & drop handlers
["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  if (dt && dt.files && dt.files.length > 0) {
    handleFileUpload(dt.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFileUpload(e.target.files[0]);
  }
});

async function handleFileUpload(file) {
  if (!file) return;

  const labelEl = dropzone.querySelector(".dropzone-label");
  labelEl.textContent = "⏳ Parsing & Shielding " + file.name + "...";

  const formData = new FormData();
  formData.append("file", file);
  if (currentSessionId) {
    formData.append("session_id", currentSessionId);
  }
  formData.append("profile", getActiveProfile());

  try {
    const res = await fetch(`${BRIDGE_URL}/redact_file`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error("HTTP " + res.status);

    const data = await res.json();
    currentSessionId = data.session_id;
    chrome.storage.local.set({ sessionId: currentSessionId });
    updateSessionDisplay();

    const docBox = document.getElementById("docResult");
    docBox.textContent = data.redacted_text;
    docBox.style.display = "block";

    document.getElementById("copyDocBtn").style.display = "block";
    const piiCount = data.detected_pii ? data.detected_pii.length : 0;
    labelEl.textContent = `✅ Sanitized: ${file.name} (${piiCount} PII masked)`;
  } catch (err) {
    alert("Error processing document: " + err.message + "\nCheck server logs.");
    labelEl.textContent = "Click or drag & drop PDF / DOCX / PPTX";
  }
}

document.getElementById("copyDocBtn").addEventListener("click", () => {
  const text = document.getElementById("docResult").textContent;
  navigator.clipboard.writeText(text);
  flashBtn("copyDocBtn", "Copied!");
});

// Tab 3: Restore
document.getElementById("restoreBtn").addEventListener("click", async () => {
  const input = document.getElementById("restoreInput").value;
  if (!input.trim()) {
    alert("Please paste the response to decode.");
    return;
  }

  // Ensure current session is populated
  if (!currentSessionId) {
    const stored = await chrome.storage.local.get(["sessionId"]);
    if (stored.sessionId) {
      currentSessionId = stored.sessionId;
    }
  }

  if (!currentSessionId) {
    alert("No active session found. Please redact text or upload a document first.");
    return;
  }

  const btn = document.getElementById("restoreBtn");
  btn.textContent = "⏳ Restoring...";
  btn.disabled = true;

  try {
    const res = await fetch(`${BRIDGE_URL}/restore_text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: input,
        session_id: currentSessionId,
      }),
    });

    if (!res.ok) throw new Error("HTTP " + res.status);

    const data = await res.json();
    const box = document.getElementById("restoredResult");
    box.textContent = data.restored_text;
    box.style.display = "block";
  } catch (err) {
    alert("Could not restore: " + err.message);
  } finally {
    btn.textContent = "🔓 Restore Original Values";
    btn.disabled = false;
  }
});

document.getElementById("copyRestoredBtn").addEventListener("click", () => {
  const text = document.getElementById("restoredResult").textContent;
  navigator.clipboard.writeText(text);
  flashBtn("copyRestoredBtn", "Copied!");
});

function flashBtn(id, text) {
  const el = document.getElementById(id);
  const orig = el.textContent;
  el.textContent = "✓ " + text;
  setTimeout(() => (el.textContent = orig), 1500);
}

// Initial check
checkHealth();

const BRIDGE_URL = "http://127.0.0.1:8765";

let currentSessionId = null;

// Load stored session ID if available
chrome.storage.local.get(["sessionId"], (res) => {
  if (res.sessionId) {
    currentSessionId = res.sessionId;
    document.getElementById("sessionLabel").textContent = "Session: " + currentSessionId.slice(0, 8);
  }
});

// Check Server Health
async function checkHealth() {
  const badge = document.getElementById("statusBadge");
  const text = document.getElementById("statusText");
  try {
    const res = await fetch(`${BRIDGE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      badge.className = "status-badge";
      text.textContent = "Online (" + data.profile + ")";
    } else {
      throw new Error("HTTP " + res.status);
    }
  } catch (err) {
    badge.className = "status-badge offline";
    text.textContent = "Offline (Start Server)";
  }
}

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
      body: jsonBody({ text: input, session_id: currentSessionId }),
    });

    const data = await res.json();
    currentSessionId = data.session_id;
    chrome.storage.local.set({ sessionId: currentSessionId });
    document.getElementById("sessionLabel").textContent = "Session: " + currentSessionId.slice(0, 8);

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
      chipsContainer.innerHTML = "<span style='font-size:11px;color:#10b981;'>✅ No PII detected</span>";
    }
  } catch (err) {
    alert("Could not reach PII Shield local bridge. Ensure 'python extension_server/server.py' is running.");
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
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  dropzone.querySelector(".dropzone-label").textContent = "⏳ Parsing & Shielding " + file.name + "...";

  try {
    const res = await fetch(`${BRIDGE_URL}/redact_file`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    currentSessionId = data.session_id;
    chrome.storage.local.set({ sessionId: currentSessionId });
    document.getElementById("sessionLabel").textContent = "Session: " + currentSessionId.slice(0, 8);

    const docBox = document.getElementById("docResult");
    docBox.textContent = data.redacted_text;
    docBox.style.display = "block";

    document.getElementById("copyDocBtn").style.display = "block";
    dropzone.querySelector(".dropzone-label").textContent = `✅ Sanitized: ${file.name} (${data.detected_pii.length} PII redacted)`;
  } catch (err) {
    alert("Error uploading/processing file: " + err.message);
    dropzone.querySelector(".dropzone-label").textContent = "Click or drop PDF / DOCX / PPTX";
  }
});

document.getElementById("copyDocBtn").addEventListener("click", () => {
  const text = document.getElementById("docResult").textContent;
  navigator.clipboard.writeText(text);
  flashBtn("copyDocBtn", "Copied!");
});

// Tab 3: Restore
document.getElementById("restoreBtn").addEventListener("click", async () => {
  const input = document.getElementById("restoreInput").value;
  if (!input.trim() || !currentSessionId) {
    alert("No session active or input empty.");
    return;
  }

  const btn = document.getElementById("restoreBtn");
  btn.textContent = "⏳ Restoring...";
  btn.disabled = true;

  try {
    const res = await fetch(`${BRIDGE_URL}/restore_text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: jsonBody({ text: input, session_id: currentSessionId }),
    });
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

function jsonBody(obj) {
  return JSON.stringify(obj);
}

function flashBtn(id, text) {
  const el = document.getElementById(id);
  const orig = el.textContent;
  el.textContent = "✓ " + text;
  setTimeout(() => (el.textContent = orig), 1500);
}

checkHealth();

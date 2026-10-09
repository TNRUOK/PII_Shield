/**
 * PII Shield Content Script
 * Injected into ChatGPT, Claude, Gemini, etc.
 * Adds in-place prompt shielding (Ctrl+Shift+S or floating button)
 * and in-place response restoration.
 */

const BRIDGE_URL = "http://127.0.0.1:8765";

let activeSessionId = null;

// Retrieve existing session
chrome.storage.local.get(["sessionId"], (res) => {
  if (res.sessionId) activeSessionId = res.sessionId;
});

function showToast(msg, isSuccess = true) {
  let toast = document.getElementById("pii-shield-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "pii-shield-toast";
    document.body.appendChild(toast);
  }
  toast.className = isSuccess ? "pii-toast success" : "pii-toast error";
  toast.textContent = msg;
  toast.style.display = "block";
  setTimeout(() => {
    toast.style.display = "none";
  }, 3500);
}

// Find the main prompt input element on the page
function findPromptInput() {
  const active = document.activeElement;
  if (
    active &&
    (active.tagName === "TEXTAREA" ||
      active.getAttribute("contenteditable") === "true")
  ) {
    return active;
  }
  // Try known selectors for ChatGPT, Claude, Gemini
  const selectors = [
    "#prompt-textarea",
    "div[contenteditable='true']",
    "textarea",
    "div.ProseMirror",
    "[data-placeholder]",
  ];
  for (const s of selectors) {
    const el = document.querySelector(s);
    if (el) return el;
  }
  return null;
}

function getElementText(el) {
  if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
    return el.value;
  }
  return el.innerText || el.textContent;
}

function setElementText(el, text) {
  if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
    el.value = text;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  } else {
    el.innerText = text;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
}

// Perform Shielding
async function shieldCurrentInput() {
  const inputEl = findPromptInput();
  if (!inputEl) {
    showToast("⚠️ Could not find prompt input box", false);
    return;
  }

  const rawText = getElementText(inputEl);
  if (!rawText || !rawText.trim()) {
    showToast("⚠️ Input box is empty", false);
    return;
  }

  showToast("🛡️ Scanning for PII...");

  try {
    const res = await fetch(`${BRIDGE_URL}/redact_text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: rawText,
        session_id: activeSessionId,
      }),
    });

    if (!res.ok) throw new Error("Bridge server returned error " + res.status);

    const data = await res.json();
    activeSessionId = data.session_id;
    chrome.storage.local.set({ sessionId: activeSessionId });

    setElementText(inputEl, data.redacted_text);

    const count = data.detected_pii ? data.detected_pii.length : 0;
    if (count > 0) {
      showToast(`🛡️ PII Shielded: ${count} sensitive items replaced!`, true);
    } else {
      showToast("✅ No PII detected. Text is safe.", true);
    }
  } catch (err) {
    showToast("❌ Local PII Bridge not running at 127.0.0.1:8765", false);
  }
}

// Floating button injector
function injectFloatingButton() {
  if (document.getElementById("pii-shield-float-btn")) return;

  const btn = document.createElement("button");
  btn.id = "pii-shield-float-btn";
  btn.title = "Click or press Ctrl+Shift+S to Shield PII";
  btn.innerHTML = `<span style="font-size:14px;">🛡️</span> <span>Shield PII</span>`;
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    shieldCurrentInput();
  });
  document.body.appendChild(btn);
}

// Global hotkey: Ctrl+Shift+S
window.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.shiftKey && e.key.toUpperCase() === "S") {
    e.preventDefault();
    shieldCurrentInput();
  }
});

// Watch DOM for response restoration targets
function scanForPlaceholdersToRestore() {
  if (!activeSessionId) return;

  // Search markdown / response containers
  const textContainers = document.querySelectorAll(
    "p, li, pre, code, .markdown"
  );
  textContainers.forEach((el) => {
    if (
      el.dataset.piiChecked !== "true" &&
      /\[[A-Z_]+_\d+\]/.test(el.textContent)
    ) {
      el.dataset.piiChecked = "true";
      // Add a restore action badge
      const badge = document.createElement("span");
      badge.className = "pii-restore-pill";
      badge.title = "Click to restore original values locally";
      badge.textContent = "🔓 Restore PII";
      badge.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          const res = await fetch(`${BRIDGE_URL}/restore_text`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text: el.innerText,
              session_id: activeSessionId,
            }),
          });
          const data = await res.json();
          el.innerText = data.restored_text;
          badge.remove();
        } catch (err) {
          showToast("Could not restore: bridge unreachable", false);
        }
      });
      el.appendChild(badge);
    }
  });
}

// Initialize
setTimeout(() => {
  injectFloatingButton();
  setInterval(scanForPlaceholdersToRestore, 2000);
}, 1000);

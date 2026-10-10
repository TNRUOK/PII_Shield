/**
 * PII Shield Content Script
 * Injected into ChatGPT, Claude, Gemini, Copilot, and other web LLMs.
 * Adds:
 *   1. In-place prompt shielding (Ctrl+Shift+S or floating widget)
 *   2. Automatic document upload interception (drag & drop or "Shield Doc" button)
 *   3. In-place response restoration without violating web CSP / Mixed Content.
 */

let activeSessionId = null;
let lastFocusedInput = null;

const SUPPORTED_DOC_EXTS = [
  ".pdf",
  ".docx",
  ".pptx",
  ".doc",
  ".ppt",
  ".txt",
  ".csv",
];

// Retrieve stored session ID if available
chrome.storage.local.get(["sessionId"], (res) => {
  if (res.sessionId) activeSessionId = res.sessionId;
});

// Listen for session updates from popup
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.sessionId) {
    activeSessionId = changes.sessionId.newValue;
  }
});

// Helper to send request through background worker (bypasses CSP & Mixed Content)
function sendBridgeMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        return reject(new Error(chrome.runtime.lastError.message));
      }
      if (!response || !response.ok) {
        return reject(new Error(response?.error || "Bridge request failed"));
      }
      resolve(response.data);
    });
  });
}

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
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toast.style.display = "none";
  }, 4500);
}

// Check if a file is a document that should be sanitized locally
function isSupportedDoc(filename) {
  if (!filename) return false;
  const lower = filename.toLowerCase();
  return SUPPORTED_DOC_EXTS.some((ext) => lower.endsWith(ext));
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = (err) => reject(err);
    reader.readAsDataURL(file);
  });
}

// Track currently active editable element
document.addEventListener(
  "focusin",
  (e) => {
    const el = e.target;
    if (isEditable(el) && !el.closest("#pii-shield-widget")) {
      lastFocusedInput = el;
    }
  },
  true
);

function isEditable(el) {
  if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
  const tag = el.tagName;
  if (tag === "TEXTAREA") return true;
  if (tag === "INPUT" && ["text", "search", ""].includes(el.type || "")) return true;
  if (el.isContentEditable || el.getAttribute("contenteditable") === "true") return true;
  return false;
}

// Find the main prompt input element on the page
function findPromptInput() {
  const active = document.activeElement;
  if (active && isEditable(active) && !active.closest("#pii-shield-widget")) {
    return active;
  }

  if (lastFocusedInput && document.body.contains(lastFocusedInput) && isEditable(lastFocusedInput)) {
    return lastFocusedInput;
  }

  // Priority selectors for ChatGPT, Claude, Gemini, Copilot, DeepSeek
  const selectors = [
    "#prompt-textarea",                           // ChatGPT
    "div#prompt-textarea",                         // ChatGPT modern ProseMirror/Lexical
    "textarea#prompt-textarea",                    // ChatGPT legacy
    "div.ProseMirror[contenteditable='true']",     // Claude
    "div.rich-textarea [contenteditable='true']",  // Gemini
    "div[contenteditable='true'][role='textbox']", // Generic LLM textbox
    "div[contenteditable='true']",
    "textarea[placeholder]",
    "textarea",
  ];

  for (const s of selectors) {
    const list = document.querySelectorAll(s);
    for (const el of list) {
      if (el.closest("#pii-shield-widget")) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        return el;
      }
    }
  }
  return null;
}

function getElementText(el) {
  if (!el) return "";
  if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
    return el.value || "";
  }
  return el.innerText || el.textContent || "";
}

function setElementText(el, text) {
  if (!el) return;
  el.focus();

  if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
    // React value-tracker override
    const proto =
      el.tagName === "TEXTAREA"
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    if (setter) {
      setter.call(el, text);
    } else {
      el.value = text;
    }
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  } else {
    // Contenteditable: select all existing content first, then replace atomically.
    // This prevents partial replacement bugs in ProseMirror / Lexical editors.
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(el);
    selection.removeAllRanges();
    selection.addRange(range);

    let executed = false;
    try {
      executed = document.execCommand("insertText", false, text);
    } catch (e) {
      executed = false;
    }

    if (!executed) {
      // execCommand is deprecated — fall back to direct assignment + synthetic event
      el.innerText = text;
      el.dispatchEvent(
        new InputEvent("input", {
          bubbles: true,
          inputType: "insertText",
          data: text,
        })
      );
    }
  }
}

// Perform In-Place Prompt Shielding
async function shieldCurrentInput() {
  const inputEl = findPromptInput();
  if (!inputEl) {
    showToast("⚠️ Could not find prompt input box on this page", false);
    return;
  }

  const rawText = getElementText(inputEl);
  if (!rawText || !rawText.trim()) {
    showToast("⚠️ Prompt input box is empty", false);
    return;
  }

  const promptBtn = document.getElementById("pii-shield-float-btn");
  if (promptBtn) {
    promptBtn.classList.add("loading");
    promptBtn.disabled = true;
  }

  showToast("🛡️ Scanning prompt for PII with local engine...");

  try {
    const data = await sendBridgeMessage({
      type: "REDACT_TEXT",
      text: rawText,
      sessionId: activeSessionId,
    });

    activeSessionId = data.session_id;
    chrome.storage.local.set({ sessionId: activeSessionId });

    setElementText(inputEl, data.redacted_text);

    const count = data.detected_pii ? data.detected_pii.length : 0;
    if (count > 0) {
      showToast(`🛡️ PII Shielded: ${count} sensitive item${count > 1 ? "s" : ""} sanitized!`, true);
    } else {
      showToast("✅ No PII detected. Prompt is clean & safe.", true);
    }
  } catch (err) {
    showToast("❌ " + err.message, false);
  } finally {
    if (promptBtn) {
      promptBtn.classList.remove("loading");
      promptBtn.disabled = false;
    }
  }
}

// Sanitize a document locally and inject sanitized text into the prompt box
async function handleLocalDocumentSanitize(file) {
  if (!file) return;

  const docBtn = document.getElementById("pii-shield-doc-btn");
  if (docBtn) {
    docBtn.classList.add("loading");
    docBtn.disabled = true;
  }

  showToast(`🛡️ Intercepted "${file.name}"! Sanitizing locally before cloud LLM...`, true);

  try {
    const base64Data = await fileToBase64(file);
    const data = await sendBridgeMessage({
      type: "REDACT_FILE",
      filename: file.name,
      fileType: file.type,
      base64Data: base64Data,
      sessionId: activeSessionId,
    });

    activeSessionId = data.session_id;
    chrome.storage.local.set({ sessionId: activeSessionId });

    const inputEl = findPromptInput();
    const formattedPrompt = `[Document: ${file.name}]\n\n${data.redacted_text}`;

    if (inputEl) {
      setElementText(inputEl, formattedPrompt);
      const count = data.detected_pii ? data.detected_pii.length : 0;
      showToast(`✅ "${file.name}" shielded! ${count} PII masked & pasted safely.`, true);
    } else {
      navigator.clipboard.writeText(formattedPrompt);
      showToast(`✅ "${file.name}" sanitized! Copied to clipboard.`, true);
    }
  } catch (err) {
    showToast("❌ Could not sanitize file: " + err.message, false);
  } finally {
    if (docBtn) {
      docBtn.classList.remove("loading");
      docBtn.disabled = false;
    }
  }
}

// Floating widget injector
function injectFloatingWidget() {
  if (document.getElementById("pii-shield-widget")) return;

  const container = document.createElement("div");
  container.id = "pii-shield-widget";

  // Hidden file input for "Shield Doc" button
  const filePicker = document.createElement("input");
  filePicker.type = "file";
  filePicker.id = "pii-shield-file-picker";
  filePicker.accept = ".pdf,.docx,.pptx,.png,.jpg,.jpeg,.txt,.csv";
  filePicker.style.display = "none";
  filePicker.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleLocalDocumentSanitize(e.target.files[0]);
    }
  });

  // Button 1: Shield Current Prompt
  const promptBtn = document.createElement("button");
  promptBtn.id = "pii-shield-float-btn";
  promptBtn.className = "pii-widget-btn";
  promptBtn.type = "button";
  promptBtn.title = "Click or press Ctrl+Shift+S to Shield Prompt text";
  promptBtn.innerHTML = `<span>🛡️</span> <span>Shield PII</span>`;
  promptBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    shieldCurrentInput();
  });

  // Button 2: Shield & Attach Document
  const docBtn = document.createElement("button");
  docBtn.id = "pii-shield-doc-btn";
  docBtn.className = "pii-widget-btn secondary";
  docBtn.type = "button";
  docBtn.title = "Upload PDF/DOCX/PPTX to sanitize locally with zero cloud leak";
  docBtn.innerHTML = `<span>📄</span> <span>Shield Doc</span>`;
  docBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    filePicker.click();
  });

  container.appendChild(promptBtn);
  container.appendChild(docBtn);
  container.appendChild(filePicker);
  document.body.appendChild(container);
}

// AUTOMATIC FILE UPLOAD INTERCEPTION
// 1. Intercept Drag-and-Drop files on the page before ChatGPT/Claude uploads raw file to cloud
window.addEventListener(
  "drop",
  async (e) => {
    const files = e.dataTransfer?.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    if (isSupportedDoc(file.name)) {
      // Intercept! Stop ChatGPT/Claude from receiving the raw document
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();

      handleLocalDocumentSanitize(file);
    }
  },
  true // CAPTURE phase to intercept before ChatGPT / Claude drop listeners
);

window.addEventListener(
  "dragover",
  (e) => {
    const items = e.dataTransfer?.items;
    if (items) {
      for (const item of items) {
        if (item.kind === "file") {
          e.preventDefault();
          break;
        }
      }
    }
  },
  true
);

// 2. Intercept web page native file inputs if a user attaches a document via paperclip
document.addEventListener(
  "change",
  async (e) => {
    const target = e.target;
    if (
      target &&
      target.tagName === "INPUT" &&
      target.type === "file" &&
      target.id !== "pii-shield-file-picker"
    ) {
      const file = target.files?.[0];
      if (file && isSupportedDoc(file.name)) {
        // Stop the web page from uploading the raw document to cloud
        e.stopPropagation();
        e.stopImmediatePropagation();
        target.value = ""; // clear input so page doesn't upload

        handleLocalDocumentSanitize(file);
      }
    }
  },
  true // CAPTURE phase
);

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

  const candidateSelector = "p, li, pre, code";
  const elements = document.querySelectorAll(candidateSelector);

  elements.forEach((el) => {
    // Skip if already checked or inside an element already checked
    if (el.dataset.piiChecked === "true" || el.closest('[data-pii-checked="true"]')) return;
    if (el.querySelector(".pii-restore-pill")) return;

    // Skip user message input containers, widgets, and file upload previews
    if (
      el.closest(
        "#pii-shield-widget, textarea, input, [contenteditable='true'], #prompt-textarea, [role='textbox'], .file-preview, [data-testid*='upload'], [class*='upload'], [class*='attachment']"
      )
    ) {
      return;
    }

    const placeholderRegex = /\[[A-Z_]+_\d+\]/;
    if (!placeholderRegex.test(el.textContent)) return;

    // Target innermost elements only: skip if any child candidate element also matches
    const matchingChild = Array.from(el.querySelectorAll(candidateSelector)).some((child) =>
      placeholderRegex.test(child.textContent)
    );
    if (matchingChild) return;

    el.dataset.piiChecked = "true";

    const badge = document.createElement("button");
    badge.className = "pii-restore-pill";
    badge.type = "button";
    badge.title = "Click to restore original values locally";
    badge.setAttribute("aria-hidden", "true");
    badge.setAttribute("tabindex", "-1");
    // Text is rendered via CSS ::after so reading el.innerText / textContent never copies the button label

    badge.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();

      badge.dataset.status = "loading";
      badge.disabled = true;

      try {
        const clone = el.cloneNode(true);
        const pill = clone.querySelector(".pii-restore-pill");
        if (pill) pill.remove();
        const targetText = clone.innerText || clone.textContent;

        const data = await sendBridgeMessage({
          type: "RESTORE_TEXT",
          text: targetText,
          sessionId: activeSessionId,
        });

        el.innerText = data.restored_text;
        showToast("✅ Restored original values locally!", true);
      } catch (err) {
        delete badge.dataset.status;
        badge.disabled = false;
        showToast("Could not restore: " + err.message, false);
      }
    });

    el.appendChild(badge);
  });
}

// Initialize widget immediately on load
injectFloatingWidget();
scanForPlaceholdersToRestore();

// Keep widget present across SPA navigation with a lightweight interval
// (only injects if widget was removed — cheap DOM id check)
setInterval(injectFloatingWidget, 2000);

// Replace polling for placeholder restoration with a MutationObserver.
// This fires only when new DOM nodes are added (zero cost on idle pages)
// rather than scanning the entire DOM every 1.5 seconds.
const _piiObserver = new MutationObserver((mutations) => {
  let hasNewNodes = false;
  for (const m of mutations) {
    if (m.addedNodes.length > 0) {
      hasNewNodes = true;
      break;
    }
  }
  if (hasNewNodes) {
    scanForPlaceholdersToRestore();
  }
});

_piiObserver.observe(document.body, {
  childList: true,
  subtree: true,
});

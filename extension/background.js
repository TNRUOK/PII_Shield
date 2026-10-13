/**
 * PII Shield - Background Service Worker (Manifest V3)
 * Proxies API requests to 127.0.0.1:8765.
 * This completely resolves browser Mixed Content (HTTPS -> HTTP) and CSP restrictions
 * on websites like ChatGPT, Claude, and Gemini.
 */

const BRIDGE_URL = "http://127.0.0.1:8765";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (!request || !request.type) return false;

  handleRequest(request)
    .then((data) => sendResponse({ ok: true, data }))
    .catch((err) => {
      console.warn("[PII Shield Background] Error:", err.message);
      sendResponse({ ok: false, error: err.message });
    });

  return true; // Keep channel open for async response
});

async function handleRequest(request) {
  switch (request.type) {
    case "CHECK_HEALTH": {
      const res = await fetchWithRetry(`${BRIDGE_URL}/health`, { method: "GET" }, 4000);
      if (!res.ok) throw new Error(`Bridge returned HTTP ${res.status}`);
      return await res.json();
    }

    case "REDACT_TEXT": {
      const res = await fetchWithRetry(
        `${BRIDGE_URL}/redact_text`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: request.text || "",
            session_id: request.sessionId || null,
            profile: request.profile || "balanced",
          }),
        },
        15000
      );
      if (!res.ok) throw new Error(`Bridge returned HTTP ${res.status}`);
      return await res.json();
    }

    case "RESTORE_TEXT": {
      const res = await fetchWithRetry(
        `${BRIDGE_URL}/restore_text`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: request.text || "",
            session_id: request.sessionId,
          }),
        },
        10000
      );
      if (!res.ok) throw new Error(`Bridge returned HTTP ${res.status}`);
      return await res.json();
    }

    case "REDACT_FILE": {
      // Decode base64 to binary blob
      const binary = atob(request.base64Data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: request.fileType || "application/octet-stream" });
      const formData = new FormData();
      formData.append("file", blob, request.filename);
      if (request.sessionId) {
        formData.append("session_id", request.sessionId);
      }
      formData.append("profile", request.profile || "balanced");

      const res = await fetchWithRetry(
        `${BRIDGE_URL}/redact_file`,
        {
          method: "POST",
          body: formData,
        },
        180000 // Allow up to 3 minutes for multi-page documents/PDFs
      );
      if (!res.ok) throw new Error(`Bridge returned HTTP ${res.status}`);
      return await res.json();
    }

    default:
      throw new Error(`Unknown message type: ${request.type}`);
  }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    return res;
  } catch (err) {
    clearTimeout(id);
    if (err.name === "AbortError") {
      throw new Error("Bridge connection timed out (server busy or offline)");
    }
    throw new Error("Cannot reach PII Shield local bridge. Ensure 'python extension_server/server.py' is running.");
  }
}

/**
 * fetchWithRetry: wraps fetchWithTimeout with up to 2 retries (3 total attempts)
 * and 500ms exponential backoff on non-network errors (server 5xx).
 */
async function fetchWithRetry(url, options = {}, timeoutMs = 8000, maxRetries = 2) {
  let lastErr;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetchWithTimeout(url, options, timeoutMs);
      // Only retry on 5xx server errors, not 4xx client errors
      if (res.status >= 500 && attempt < maxRetries) {
        lastErr = new Error(`Bridge returned HTTP ${res.status}`);
        await new Promise((r) => setTimeout(r, 500 * Math.pow(2, attempt)));
        continue;
      }
      return res;
    } catch (err) {
      lastErr = err;
      // Don't retry timeout or network errors — they'll likely fail again immediately
      if (err.message.includes("timed out") || err.message.includes("Cannot reach")) {
        throw err;
      }
      if (attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, 500 * Math.pow(2, attempt)));
      }
    }
  }
  throw lastErr;
}

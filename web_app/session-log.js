/*
 * session-log.js — history of confirmed signs for the collapsible Session Log.
 * Stored only in this browser (localStorage). Nothing leaves the device.
 * Exposes window.AnvayaSessionLog = { add(word, confidence), clear() }.
 */
(function () {
  const KEY = "anvaya_session_log";
  const MAX_ENTRIES = 100;
  const listEl = document.getElementById("session_log_list");
  const clearBtn = document.getElementById("log_clear_btn");
  let entries = [];

  try { entries = JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { entries = []; }

  function save() {
    try { localStorage.setItem(KEY, JSON.stringify(entries)); } catch (e) { /* storage blocked: keep in memory */ }
  }

  function render() {
    listEl.innerHTML = "";
    if (entries.length === 0) {
      const li = document.createElement("li");
      li.className = "log-empty";
      li.textContent = "No signs recorded yet.";
      listEl.appendChild(li);
      return;
    }
    for (const e of [...entries].reverse()) {
      const li = document.createElement("li");
      const word = document.createElement("span");
      word.textContent = e.word;
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = `${new Date(e.time).toLocaleTimeString()} · ${Math.round(e.confidence * 100)}%`;
      li.append(word, meta);
      listEl.appendChild(li);
    }
  }

  function add(word, confidence) {
    entries.push({ word, confidence, time: Date.now() });
    if (entries.length > MAX_ENTRIES) entries.shift();
    save();
    render();
  }

  function clear() {
    entries = [];
    save();
    render();
  }

  clearBtn.addEventListener("click", clear);
  window.AnvayaSessionLog = { add, clear };
  render();
})();
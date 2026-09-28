/*
 * controls.js — speech settings (speed, pitch, voice) + copy/clear buttons.
 * Uses the browser's built-in Web Speech API: no server, no network.
 * Exposes window.AnvayaControls = { speak(text, { interrupt }) }.
 * The clear button fires a "anvaya:clear" event so app.js can reset its state.
 */
(function () {
  const speedEl = document.getElementById("speed_slider");
  const pitchEl = document.getElementById("pitch_slider");
  const voiceEl = document.getElementById("voice_select");
  const speakBtn = document.getElementById("speak_btn");
  const copyBtn = document.getElementById("copy_btn");
  const clearBtn = document.getElementById("clear_btn");
  const textEl = document.getElementById("translation_text");
  let voices = [];

  function loadVoices() {
    if (!("speechSynthesis" in window)) {
      voiceEl.innerHTML = "<option>Speech not supported</option>";
      return;
    }
    voices = window.speechSynthesis.getVoices();
    if (voices.length === 0) return; // some browsers load voices late; "voiceschanged" fires again
    voiceEl.innerHTML = "";
    voices.forEach((v, i) => {
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = `${v.name} (${v.lang})`;
      voiceEl.appendChild(opt);
    });
    const preferred = voices.findIndex((v) => v.lang === "en-US");
    voiceEl.value = preferred >= 0 ? preferred : 0;
  }

  function speak(text, { interrupt = false } = {}) {
    if (!("speechSynthesis" in window) || !text) return;
    if (interrupt) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = parseFloat(speedEl.value);
    u.pitch = parseFloat(pitchEl.value);
    const voice = voices[parseInt(voiceEl.value, 10)];
    if (voice) u.voice = voice;
    window.speechSynthesis.speak(u);
  }

  speedEl.addEventListener("input", () => {
    document.getElementById("speed_value").textContent = `${parseFloat(speedEl.value).toFixed(1)}x`;
  });
  pitchEl.addEventListener("input", () => {
    document.getElementById("pitch_value").textContent = parseFloat(pitchEl.value).toFixed(1);
  });

  speakBtn.addEventListener("click", () => speak(textEl.textContent.trim(), { interrupt: true }));

  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(textEl.textContent);
      copyBtn.title = "Copied!";
      setTimeout(() => (copyBtn.title = "Copy"), 1500);
    } catch (e) {
      console.warn("Clipboard unavailable:", e);
    }
  });

  clearBtn.addEventListener("click", () => document.dispatchEvent(new Event("anvaya:clear")));

  if ("speechSynthesis" in window) window.speechSynthesis.onvoiceschanged = loadVoices;
  loadVoices();
  window.AnvayaControls = { speak };
})();
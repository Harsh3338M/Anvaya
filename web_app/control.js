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

  // Sliders are 0-100 (Stitch design). Map them to the ranges the Web Speech
  // API actually expects: rate ~0.5-1.5 (comfortable listening speed), pitch
  // 0-2 (its documented valid range). Feeding the raw 0-100 value straight
  // into `rate`/`pitch` -- as an earlier version of this file did -- produces
  // invalid or garbled speech, since those aren't the API's real units.
  const mapRate = (v) => 0.5 + (v / 100) * 1.0;   // 0 -> 0.5x, 50 -> 1.0x, 100 -> 1.5x
  const mapPitch = (v) => (v / 100) * 2.0;        // 0 -> 0.0,  50 -> 1.0,  100 -> 2.0

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
    u.rate = mapRate(parseFloat(speedEl.value));
    u.pitch = mapPitch(parseFloat(pitchEl.value));
    const voice = voices[parseInt(voiceEl.value, 10)];
    if (voice) u.voice = voice;
    window.speechSynthesis.speak(u);
  }

  const speedValueEl = document.getElementById("speed_value");
  const pitchValueEl = document.getElementById("pitch_value");

  function updateSpeedLabel() {
    speedValueEl.textContent = `${mapRate(parseFloat(speedEl.value)).toFixed(1)}x`;
  }
  function updatePitchLabel() {
    pitchValueEl.textContent = mapPitch(parseFloat(pitchEl.value)).toFixed(1);
  }

  speedEl.addEventListener("input", updateSpeedLabel);
  pitchEl.addEventListener("input", updatePitchLabel);
  updateSpeedLabel(); // sync labels to the sliders' actual starting values on load
  updatePitchLabel();

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
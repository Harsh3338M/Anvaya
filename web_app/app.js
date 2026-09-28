/*
 * app.js — runtime for studio.html. 100% client-side, inference only (no training).
 * Normalization mirrors data_collection/normalizer.py (see docs/ANVAYA_PIPELINE.md §3).
 * Depends on session-log.js and controls.js being loaded first.
 */

// ---- Config: MUST match capture_sequences.py / train_cnn.py ----
const SEQUENCE_LENGTH = 24;
const NUM_LANDMARKS = 21;
const VALUES_PER_HAND = NUM_LANDMARKS * 3 + 1; // 63 coords + presence flag
const FRAME_VECTOR_SIZE = VALUES_PER_HAND * 2; // 128

// ---- Debounce config ----
const PREDICTION_HISTORY_SIZE = 6;
const MIN_VOTES_TO_CONFIRM = 4;
const CONFIDENCE_THRESHOLD = 0.8;
const REPEAT_LOCKOUT_MS = 1200;
const INFERENCE_INTERVAL_MS = 120;

// ---- State ----
let model = null;
let indexToLabel = [];
let frameBuffer = [];
let predictionHistory = [];
let lastEmittedLabel = null, lastEmittedAt = 0, lastInferenceAt = 0;
let sentence = "";
let cameraOn = true;
let handsVisible = null;
let frameCount = 0, lastFpsCheck = performance.now();

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const videoEl = $("input_video"), canvasEl = $("overlay_canvas"), ctx = canvasEl.getContext("2d");
const badgeEl = $("live_badge"), overlayEl = $("camera_overlay"), cameraBtn = $("camera_toggle");
const modelStatusEl = $("model_status"), fpsEl = $("fps_counter");
const predictionEl = $("current_prediction"), confBarEl = $("confidence_bar");
const textEl = $("translation_text"), autoSpeakEl = $("auto_speak_toggle");

// ---- Normalization (mirror of normalizer.py) ----
function normalizeSingleHand(lm) {
  const w = lm[0];
  const pts = lm.map((p) => [p.x - w.x, p.y - w.y, p.z - w.z]);
  let dMax = 1e-8;
  for (const [x, y, z] of pts) dMax = Math.max(dMax, Math.hypot(x, y, z));
  return pts.flatMap(([x, y, z]) => [x / dMax, y / dMax, z / dMax]);
}

function handSlot(lm) {
  if (!lm) return new Array(VALUES_PER_HAND).fill(0);
  return [...normalizeSingleHand(lm), 1];
}

/*
 * IMPORTANT (fix vs. earlier version): capture_sequences.py feeds MediaPipe a
 * horizontally FLIPPED frame (selfie view). Here the video is NOT flipped, so
 * to match training data we (1) mirror x -> 1 - x and (2) swap the handedness
 * label (MediaPipe assumes mirrored input, so on an unflipped frame it reports
 * the opposite hand). Without this, left/right hands and x-direction would
 * not match what the model was trained on.
 */
function buildFrameVector(results) {
  let left = null, right = null;
  if (results.multiHandLandmarks && results.multiHandedness) {
    results.multiHandLandmarks.forEach((lm, i) => {
      const mirrored = lm.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
      const label = results.multiHandedness[i].label; // as reported for the unflipped frame
      if (label === "Left") right = mirrored; else left = mirrored;
    });
  }
  return [...handSlot(left), ...handSlot(right)];
}

// ---- Model ----
async function loadModel() {
  try {
    model = await tf.loadLayersModel("model/model.json");
    const labelMap = await (await fetch("model/label_map.json")).json();
    for (const [name, idx] of Object.entries(labelMap)) indexToLabel[idx] = name;
    modelStatusEl.textContent = `Model ready (${indexToLabel.length} signs)`;
  } catch (err) {
    console.error("Model load failed:", err);
    modelStatusEl.textContent = "No trained model found in web_app/model/ (see docs/BEGINNER_SETUP_GUIDE.md)";
  }
}

function runInference() {
  if (!model || frameBuffer.length < SEQUENCE_LENGTH) return null;
  return tf.tidy(() => {
    const input = tf.tensor3d([frameBuffer], [1, SEQUENCE_LENGTH, FRAME_VECTOR_SIZE]);
    const probs = model.predict(input).dataSync();
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: indexToLabel[best], confidence: probs[best] };
  });
}

// ---- Debounce / majority vote ----
function pushPrediction(pred) {
  predictionHistory.push(pred && pred.confidence >= CONFIDENCE_THRESHOLD ? pred.label : null);
  if (predictionHistory.length > PREDICTION_HISTORY_SIZE) predictionHistory.shift();

  const counts = {};
  for (const l of predictionHistory) if (l) counts[l] = (counts[l] || 0) + 1;
  const [winner, votes] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] || [];

  if (winner && votes >= MIN_VOTES_TO_CONFIRM) {
    const now = performance.now();
    if (winner === lastEmittedLabel && now - lastEmittedAt < REPEAT_LOCKOUT_MS) return;
    emitToken(winner, pred.confidence);
    lastEmittedLabel = winner;
    lastEmittedAt = now;
    predictionHistory = [];
  }
}

function emitToken(label, confidence) {
  const word = label.replace(/_/g, " ");
  sentence += (sentence ? " " : "") + word;
  textEl.textContent = sentence;
  window.AnvayaSessionLog.add(word, confidence);
  if (autoSpeakEl.checked) window.AnvayaControls.speak(word);
}

// ---- UI state ----
function setOverlay(text, visible) {
  overlayEl.textContent = text;
  overlayEl.classList.toggle("hidden", !visible);
}

function setLive(on) {
  cameraOn = on;
  badgeEl.className = on ? "badge badge-success" : "badge badge-off";
  badgeEl.textContent = on ? "Live Feed Active" : "Camera Off";
  if (!on) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    frameBuffer = [];
    predictionHistory = [];
    handsVisible = null;
    predictionEl.textContent = "\u2014";
    confBarEl.style.width = "0%";
    setOverlay("Camera is off", true);
  } else {
    setOverlay("Reading your signs...", true);
  }
}

// ---- Drawing ----
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];

function drawHands(results) {
  canvasEl.width = videoEl.videoWidth || 640;
  canvasEl.height = videoEl.videoHeight || 480;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  for (const lm of results.multiHandLandmarks || []) {
    ctx.strokeStyle = "#14b8a6"; ctx.lineWidth = 2;
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(lm[a].x * canvasEl.width, lm[a].y * canvasEl.height);
      ctx.lineTo(lm[b].x * canvasEl.width, lm[b].y * canvasEl.height);
      ctx.stroke();
    }
    ctx.fillStyle = "#ffffff";
    for (const p of lm) {
      ctx.beginPath();
      ctx.arc(p.x * canvasEl.width, p.y * canvasEl.height, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

// ---- MediaPipe ----
const hands = new Hands({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });

hands.onResults((results) => {
  if (!cameraOn) return;
  drawHands(results);

  const hasHands = !!(results.multiHandLandmarks && results.multiHandLandmarks.length);
  if (hasHands !== handsVisible) {
    handsVisible = hasHands;
    setOverlay("Reading your signs...", !hasHands);
  }

  frameBuffer.push(buildFrameVector(results));
  if (frameBuffer.length > SEQUENCE_LENGTH) frameBuffer.shift();

  const now = performance.now();
  if (now - lastInferenceAt >= INFERENCE_INTERVAL_MS) {
    lastInferenceAt = now;
    if (!hasHands) {
      predictionHistory.push(null); // idle frames never count as votes
      if (predictionHistory.length > PREDICTION_HISTORY_SIZE) predictionHistory.shift();
    } else {
      const pred = runInference();
      if (pred) {
        predictionEl.textContent = pred.label.replace(/_/g, " ");
        confBarEl.style.width = `${Math.round(pred.confidence * 100)}%`;
        pushPrediction(pred);
      }
    }
  }

  frameCount++;
  if (now - lastFpsCheck >= 1000) {
    fpsEl.textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastFpsCheck = now;
  }
});

const camera = new Camera(videoEl, {
  onFrame: async () => { await hands.send({ image: videoEl }); },
  width: 640, height: 480,
});

// ---- Wiring ----
cameraBtn.addEventListener("click", async () => {
  if (cameraOn) { await camera.stop(); setLive(false); }
  else { await camera.start(); setLive(true); }
});

document.addEventListener("anvaya:clear", () => {
  sentence = "";
  textEl.textContent = "";
  lastEmittedLabel = null;
});

(async function init() {
  setOverlay("Reading your signs...", true);
  await loadModel();
  camera.start();
})();
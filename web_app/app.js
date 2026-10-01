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
const badgeEl = $("live_badge"), overlayEl = $("camera_overlay"), overlayTextEl = $("camera_overlay_text"), cameraBtn = $("camera_toggle");
const cameraIconEl = cameraBtn.querySelector(".material-symbols-outlined");
const badgeDotEl = badgeEl.querySelector("span:first-child"), badgeLabelEl = badgeEl.querySelector("span:last-child");
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
  // Defensive: these come from session-log.js / controls.js, both optional
  // scripts. If either failed to load (e.g. a 404 on the server, a typo in a
  // <script> tag), calling straight into them used to throw inside this
  // function -- which runs inside MediaPipe's per-frame callback, so an
  // uncaught error here silently froze the entire camera loop with no
  // visible error. Guard + a visible warning is much easier to diagnose than
  // "the app just stopped."
  if (window.AnvayaSessionLog) {
    window.AnvayaSessionLog.add(word, confidence);
  } else {
    console.warn("AnvayaSessionLog not loaded -- check that session-log.js is present and loaded before app.js.");
  }
  if (autoSpeakEl.checked) {
    if (window.AnvayaControls) {
      window.AnvayaControls.speak(word);
    } else {
      console.warn("AnvayaControls not loaded -- check that controls.js is present and loaded before app.js.");
    }
  }
}

// ---- UI state ----
// Updates only the caption text -- the mascot <img> inside #camera_overlay is
// left untouched (a previous version used .textContent here, which silently
// deleted the mascot image; don't repeat that mistake).
function setOverlay(text, visible) {
  overlayTextEl.textContent = text;
  overlayEl.classList.toggle("hidden", !visible);
}

// Toggles the badge's Tailwind classes directly rather than replacing its
// contents, so the pulsing status dot (a nested <span>) survives the update.
function setLive(on) {
  cameraOn = on;
  if (on) {
    badgeEl.className = "flex items-center gap-2 bg-success-mint/10 text-on-primary-fixed-variant px-3 py-1.5 rounded-full border border-success-mint/30";
    badgeDotEl.className = "w-2.5 h-2.5 rounded-full bg-success-mint animate-pulse";
    badgeLabelEl.textContent = "Live Feed Active";
    cameraIconEl.textContent = "videocam_off"; // icon shows the action tapping it performs next
  } else {
    badgeEl.className = "flex items-center gap-2 bg-surface-container text-on-surface-variant px-3 py-1.5 rounded-full border border-outline-variant";
    badgeDotEl.className = "w-2.5 h-2.5 rounded-full bg-outline";
    badgeLabelEl.textContent = "Camera Off";
    cameraIconEl.textContent = "videocam";
  }
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
  // Only resize the canvas buffer when the video's actual size changes (normally
  // once, right after the camera starts). Reassigning .width/.height every frame
  // -- even to the same value -- forces the browser to clear and reallocate the
  // canvas's drawing buffer, which was a real, needless cost on every single frame.
  const w = videoEl.videoWidth || 640, h = videoEl.videoHeight || 480;
  if (canvasEl.width !== w || canvasEl.height !== h) {
    canvasEl.width = w;
    canvasEl.height = h;
  }
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
// modelComplexity: 0 = MediaPipe's "lite" hand-landmark model. This is the main
// FPS fix: complexity 1 ("full") tracking two hands on CPU/WASM is a known
// bottleneck, commonly landing well under 20 FPS on mid-range hardware -- which
// matches the slowdown happening even with no model/inference running at all.
// Trade-off: complexity 0 is somewhat less precise on tricky hand poses. If you
// have a strong laptop and want the accuracy back, set this to 1 and re-test FPS.
const HAND_MODEL_COMPLEXITY = 0;
hands.setOptions({ maxNumHands: 2, modelComplexity: HAND_MODEL_COMPLEXITY, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });

hands.onResults((results) => {
  if (!cameraOn) return;
  try {
    processResults(results);
  } catch (err) {
    // An error anywhere in here used to silently stop the whole camera loop
    // (see the guards above for the specific bug that first caused this).
    // Surfacing it both in the console and the on-page status text makes
    // "the app froze" into "here's exactly what broke."
    console.error("Error while processing a camera frame:", err);
    modelStatusEl.textContent = `Error: ${err.message} (see browser console for details)`;
  }
});

function processResults(results) {
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
}

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
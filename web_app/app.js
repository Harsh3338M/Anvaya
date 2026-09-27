/*
 * app.js
 *
 * The ENTIRE runtime for Anvaya. No backend, no server calls after page load.
 * Everything here mirrors the Python logic used during training
 * (see data_collection/normalizer.py and docs/ANVAYA_PIPELINE.md, Section 3)
 * so that what the model sees live matches what it was trained on.
 *
 * IMPORTANT: this file only ever RUNS the model (inference). It never trains
 * or updates the model's weights. See docs/GUIDE.md, "Common Questions".
 */

// ---------------------------------------------------------------------------
// Config -- MUST match data_collection/capture_sequences.py and
// model_training/train_cnn.py, or predictions will be meaningless.
// ---------------------------------------------------------------------------
const SEQUENCE_LENGTH = 24;       // frames per prediction window (~0.8s at 30fps)
const NUM_LANDMARKS = 21;
const COORDS_PER_HAND = NUM_LANDMARKS * 3; // 63
const VALUES_PER_HAND = COORDS_PER_HAND + 1; // 64 (+ presence flag)
const FRAME_VECTOR_SIZE = VALUES_PER_HAND * 2; // 128 (two hands)

// Debounce / majority-vote config
const PREDICTION_HISTORY_SIZE = 6;   // how many recent predictions to vote over
const MIN_VOTES_TO_CONFIRM = 4;      // majority threshold within that history
const CONFIDENCE_THRESHOLD = 0.80;   // per-prediction confidence floor
const REPEAT_LOCKOUT_MS = 1200;      // don't re-emit the same sign again within this window
const INFERENCE_INTERVAL_MS = 120;   // throttle how often we run the CNN forward pass

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let model = null;
let labelMap = null;      // { className: index }
let indexToLabel = null;  // [ className, ... ]
let frameBuffer = [];     // rolling window of (128,) arrays, up to SEQUENCE_LENGTH long
let predictionHistory = [];
let lastEmittedLabel = null;
let lastEmittedAt = 0;
let lastInferenceAt = 0;

const videoEl = document.getElementById("input_video");
const canvasEl = document.getElementById("overlay_canvas");
const canvasCtx = canvasEl.getContext("2d");
const modelStatusEl = document.getElementById("model_status");
const fpsCounterEl = document.getElementById("fps_counter");
const predictionBoxEl = document.getElementById("current_prediction");
const confidenceBarEl = document.getElementById("confidence_bar");
const sentenceBoxEl = document.getElementById("sentence_buffer");
const clearBtn = document.getElementById("clear_btn");
const speakBtn = document.getElementById("speak_btn");
const autoSpeakToggle = document.getElementById("auto_speak_toggle");

let sentence = "";
let frameCount = 0;
let lastFpsCheck = performance.now();

// ---------------------------------------------------------------------------
// Normalization -- mirrors data_collection/normalizer.py exactly
// ---------------------------------------------------------------------------

/** landmarks: array of 21 {x,y,z} objects from MediaPipe, for ONE hand. */
function normalizeSingleHand(landmarks) {
  const wrist = landmarks[0];
  const shifted = landmarks.map((p) => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]);

  let dMax = 1e-8;
  for (const [dx, dy, dz] of shifted) {
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > dMax) dMax = dist;
  }

  const flat = [];
  for (const [dx, dy, dz] of shifted) {
    flat.push(dx / dMax, dy / dMax, dz / dMax);
  }
  return flat; // length 63
}

/** Returns a length-64 array: 63 normalized coords + presence flag. */
function buildHandSlotVector(landmarks) {
  if (!landmarks) {
    return new Array(VALUES_PER_HAND).fill(0); // absent hand: all zero, flag included
  }
  const coords = normalizeSingleHand(landmarks);
  coords.push(1); // presence flag
  return coords; // length 64
}

/**
 * Builds the full 128-value frame vector from MediaPipe's onResults payload.
 *
 * NOTE ON HANDEDNESS: MediaPipe's handedness label is computed on the raw
 * (unmirrored) camera frame. This app displays a mirrored ("selfie") video
 * via CSS for a natural user experience, but the underlying landmark data
 * used here is NOT flipped -- so "Left"/"Right" below refers to the actual
 * physical hand, consistently, regardless of the mirrored display. This
 * matches what capture_sequences.py records in Python, so training and
 * inference stay consistent.
 */
function buildFrameVector(results) {
  let leftLandmarks = null;
  let rightLandmarks = null;

  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const label = results.multiHandedness[i].label; // "Left" or "Right"
      if (label === "Left") {
        leftLandmarks = results.multiHandLandmarks[i];
      } else {
        rightLandmarks = results.multiHandLandmarks[i];
      }
    }
  }

  const leftVec = buildHandSlotVector(leftLandmarks);
  const rightVec = buildHandSlotVector(rightLandmarks);
  return leftVec.concat(rightVec); // length 128
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------
async function loadModel() {
  try {
    model = await tf.loadLayersModel("model/model.json");
    const labelResp = await fetch("model/label_map.json");
    labelMap = await labelResp.json();

    indexToLabel = [];
    for (const [name, idx] of Object.entries(labelMap)) {
      indexToLabel[idx] = name;
    }

    modelStatusEl.textContent = `Model loaded (${indexToLabel.length} classes)`;
  } catch (err) {
    console.error("Failed to load model:", err);
    modelStatusEl.textContent =
      "Model failed to load. Have you run Phase 1 (docs/GUIDE.md) and copied files into web_app/model/?";
  }
}

// ---------------------------------------------------------------------------
// Inference -- a frozen forward pass only. No training happens here.
// ---------------------------------------------------------------------------
function runInference() {
  if (!model || frameBuffer.length < SEQUENCE_LENGTH) return null;

  return tf.tidy(() => {
    const input = tf.tensor3d([frameBuffer], [1, SEQUENCE_LENGTH, FRAME_VECTOR_SIZE]);
    const output = model.predict(input);
    const probs = output.dataSync();

    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    return { label: indexToLabel[bestIdx], confidence: probs[bestIdx] };
  });
}

// ---------------------------------------------------------------------------
// Debounce / majority-vote stabilizer
// ---------------------------------------------------------------------------
function pushPredictionAndMaybeEmit(prediction) {
  if (!prediction || prediction.confidence < CONFIDENCE_THRESHOLD) {
    predictionHistory.push(null);
  } else {
    predictionHistory.push(prediction.label);
  }
  if (predictionHistory.length > PREDICTION_HISTORY_SIZE) {
    predictionHistory.shift();
  }

  const counts = {};
  for (const label of predictionHistory) {
    if (!label) continue;
    counts[label] = (counts[label] || 0) + 1;
  }

  let winner = null;
  let winnerCount = 0;
  for (const [label, count] of Object.entries(counts)) {
    if (count > winnerCount) {
      winner = label;
      winnerCount = count;
    }
  }

  if (winner && winnerCount >= MIN_VOTES_TO_CONFIRM) {
    const now = performance.now();
    const isRepeatLockout =
      winner === lastEmittedLabel && now - lastEmittedAt < REPEAT_LOCKOUT_MS;

    if (!isRepeatLockout) {
      emitToken(winner);
      lastEmittedLabel = winner;
      lastEmittedAt = now;
      predictionHistory = []; // reset so we need a fresh majority for the next token
    }
  }
}

function emitToken(label) {
  const displayWord = label.replace(/_/g, " ");
  sentence += (sentence.length > 0 ? " " : "") + displayWord;
  sentenceBoxEl.textContent = sentence;

  if (autoSpeakToggle.checked) {
    speak(displayWord);
  }
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

// ---------------------------------------------------------------------------
// MediaPipe Hands setup
// ---------------------------------------------------------------------------
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
});

hands.onResults((results) => {
  drawOverlay(results);

  const frameVec = buildFrameVector(results);
  frameBuffer.push(frameVec);
  if (frameBuffer.length > SEQUENCE_LENGTH) {
    frameBuffer.shift();
  }

  const now = performance.now();
  if (now - lastInferenceAt >= INFERENCE_INTERVAL_MS) {
    lastInferenceAt = now;
    const prediction = runInference();
    if (prediction) {
      predictionBoxEl.textContent = prediction.label.replace(/_/g, " ");
      confidenceBarEl.style.width = `${Math.round(prediction.confidence * 100)}%`;
      pushPredictionAndMaybeEmit(prediction);
    }
  }

  updateFps();
});

function drawOverlay(results) {
  canvasEl.width = videoEl.videoWidth || 640;
  canvasEl.height = videoEl.videoHeight || 480;
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: "#4cc9f0", lineWidth: 2 });
      drawLandmarks(canvasCtx, landmarks, { color: "#f72585", lineWidth: 1, radius: 3 });
    }
  }
  canvasCtx.restore();
}

// Minimal connector/landmark drawing helpers (avoids pulling in the full
// @mediapipe/drawing_utils package just for two functions).
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

function drawConnectors(ctx, landmarks, connections, style) {
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.lineWidth;
  for (const [a, b] of connections) {
    const p1 = landmarks[a], p2 = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(p1.x * canvasEl.width, p1.y * canvasEl.height);
    ctx.lineTo(p2.x * canvasEl.width, p2.y * canvasEl.height);
    ctx.stroke();
  }
}

function drawLandmarks(ctx, landmarks, style) {
  ctx.fillStyle = style.color;
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * canvasEl.width, p.y * canvasEl.height, style.radius, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function updateFps() {
  frameCount++;
  const now = performance.now();
  if (now - lastFpsCheck >= 1000) {
    fpsCounterEl.textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastFpsCheck = now;
  }
}

// ---------------------------------------------------------------------------
// Camera setup
// ---------------------------------------------------------------------------
const camera = new Camera(videoEl, {
  onFrame: async () => {
    await hands.send({ image: videoEl });
  },
  width: 640,
  height: 480,
});

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
clearBtn.addEventListener("click", () => {
  sentence = "";
  sentenceBoxEl.textContent = "";
});

speakBtn.addEventListener("click", () => {
  if (sentence.trim().length > 0) speak(sentence);
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async function init() {
  await loadModel();
  camera.start();
})();

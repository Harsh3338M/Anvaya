# Anvaya — Beginner's Guide: From Zero to Running Demo

This guide assumes you have **never set up a development project before**. It explains every
tool you need to install and every command you'll type, in order. If you've already done some
of this (e.g. you have Python installed), just skip ahead.

This is the *detailed, first-time* version of `docs/GUIDE.md` — once you're comfortable, use
that shorter version as your quick reference.

---

## Part 0: Understand the Two Phases (read this first)

This project has two completely separate stages that happen at **different times**:

1. **Development phase (you, before your presentation):** you install some tools, record
   yourself doing signs on camera, and "train" a model — this teaches it what each sign looks
   like. This uses **Python** and takes place in a terminal.
2. **Presentation phase (the actual demo):** you just open a webpage in a browser. No terminal,
   no Python, no installing anything. The model you trained earlier is already built into the
   page and just makes predictions — it does not learn anything new live.

You must finish Phase 1 before Phase 2 will show anything useful (the webpage will load, but it
has no trained model to recognize signs with yet).

---

## Part 1: Install the Tools You Need

You only do this section once, ever, on your machine.

### 1.1 Install Python

- Go to [python.org/downloads](https://www.python.org/downloads/) and download **Python 3.10, 3.11**.
  Do NOT install 3.13 or newer: TensorFlow and MediaPipe don't support it yet.
- **Windows:** during install, check the box that says **"Add Python to PATH"** — this step is
  easy to miss and causes most beginner errors later.
- **Mac:** the installer handles this automatically.
- To check it worked, open a terminal (see 1.3 below) and type:
  ```bash
  python --version
  ```
  You should see something like `Python 3.11.4`. If you get an error, Python isn't installed
  correctly or isn't on your PATH — reinstall and make sure that checkbox was ticked.

### 1.2 Install Git

- Go to [git-scm.com/downloads](https://git-scm.com/downloads) and install it with default
  options.
- Check it worked:
  ```bash
  git --version
  ```

### 1.3 Open a Terminal

- **Windows:** search for "Command Prompt" or "PowerShell" in the Start menu.
- **Mac:** open "Terminal" from Spotlight search (Cmd+Space, type "Terminal").
- Every command in this guide gets typed into this window, followed by pressing Enter.

### 1.4 (Optional but recommended) Install VS Code

A code editor makes it easier to open and glance at project files. Download from
[code.visualstudio.com](https://code.visualstudio.com/). Not strictly required — you can follow
this whole guide using only the terminal — but it helps to see the folder structure visually.

---

## Part 2: Get the Project Onto Your Computer

If you already have the project folder (e.g. from a zip I gave you), skip to 2.2.

### 2.1 Clone from GitHub

In your terminal:
```bash
cd Desktop
git clone https://github.com/Harsh3338M/Anvaya.git
cd Anvaya
```
(`cd` means "change directory" — it moves your terminal into that folder. Everything from here
on assumes your terminal is sitting inside the project folder.)

### 2.2 Confirm you're in the right place

```bash
dir      # Windows
ls       # Mac/Linux
```
You should see folders named `data_collection`, `model_training`, `web_app`, `docs`.

---

## Part 3: Set Up Your Python Environment

A "virtual environment" keeps this project's Python packages separate from everything else on
your computer — standard practice, prevents version conflicts.

```bash
cd model_training
python -m venv venv
```

Activate it (you'll need to do this **every time** you open a new terminal to work on this
project):

```bash
# Windows:
.\venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

You'll know it worked because your terminal line now starts with `(venv)`.

Now install the required packages:
```bash
pip install -r requirements.txt
```
This downloads MediaPipe (which brings OpenCV with it), TensorFlow and a few others (MediaPipe is pinned to version 0.10.21 on purpose: newer versions removed the hand-tracking API this project uses) — it can take a few minutes and
show a lot of text scrolling by. That's normal.

---

## Part 4: Record Your Training Data

This is where you actually perform each sign on camera so the model can learn it.

```bash
cd ../data_collection
python capture_sequences.py
```

- A camera window will pop up along with a list of sign classes in your terminal.
- Type the number next to the sign you want to record and press Enter.
- Focus the camera window, press **SPACE** — a short countdown appears, then it records you for
  about 0.8 seconds.
- Do this **60–100 times per sign**, moving slightly (closer/farther, different angle) between
  takes — this variety is what actually makes the model work well, more than just doing lots of
  repeats identically.
- Press **c** to switch to a different sign class, **ESC** to quit when you're done with all
  signs.

**Don't rush this step.** It's the single biggest factor in whether your trained model actually
works. Skimpy, repetitive data is the most common reason a model performs badly later.

---

## Part 5: Train the Model

Still with `(venv)` active, move into the training folder:
```bash
cd ../model_training
```

### 5.1 Preprocess your recordings
```bash
python preprocess.py
```
This reads everything you recorded and checks it's valid. It'll warn you if any sign has too
few samples — go back to Part 4 and record more of that sign if so.

### 5.2 Train
```bash
python train_cnn.py
```
This is the actual "learning" step. It can take anywhere from a few minutes to longer,
depending on your laptop and how much data you recorded. You'll see numbers scrolling — watch
for `val_accuracy` climbing over time. At the end it prints a final validation accuracy — this
is roughly "how often the model guesses correctly on signs it wasn't trained on directly."

**If accuracy is low (below ~80%):** the fix is almost always going back to Part 4 and
recording more/varied data — not tweaking the code.

### 5.3 Convert it for the browser
```bash
python convert_tfjs.py
```
This turns the trained model into a format the web browser can use.

### 5.4 Move the trained files into the web app
```bash
cp -r model_web/* ../web_app/model/          # Mac/Linux
cp label_map.json ../web_app/model/         

# Windows equivalent:
xcopy model_web\* ..\web_app\model\ /E /Y
copy label_map.json ..\web_app\model\
```

You should now see `model.json`, one or more `.bin` files, and `label_map.json` inside
`web_app/model/`. This is the hand-off point — everything after this step needs **no Python at
all**.

---

## Part 6: Run the Actual Presentation

This is what you'll do on presentation day. No terminal window needs to stay open with Python
running — you're just serving plain webpage files.

```bash
cd ../web_app
python -m http.server 3000
```

(This uses Python only as a simple file server — not for any AI logic. If you'd rather not use
Python at all here, any static file server works, e.g. VS Code's "Live Server" extension.)

Open your browser and go to:
```
http://localhost:3000
```
You'll see the landing page. Click **Launch Gesture Studio** (or go straight to `http://localhost:3000/studio.html`).
Allow camera access when prompted. Perform a sign and hold it for about a second — the
translated word should appear on screen and be spoken aloud.

---

## Troubleshooting Cheat Sheet

| Problem | Likely Cause |
|---|---|
| `python: command not found` | Python isn't installed or isn't on PATH — redo Part 1.1 |
| `pip install` fails with permission errors | You forgot to activate the virtual environment (Part 3) — reactivate it, then retry |
| Camera window doesn't open during data collection | Another app is using your webcam — close Zoom/Teams/other camera apps first |
| Webpage loads but camera never starts | You need to allow camera permission in the browser — check the address bar for a blocked icon |
| Model doesn't recognize anything / says "model failed to load" | You skipped Part 5.4 — the trained model files aren't in `web_app/model/` yet |
| `module 'mediapipe' has no attribute 'solutions'` | Wrong MediaPipe version. Run `pip install mediapipe==0.10.21` |
| `No matching distribution found for tensorflow` | Your Python is too new. Install Python 3.10, 3.11 or 3.12 and recreate the venv |
| Predictions are wrong or random | Usually too little/too repetitive training data — go back to Part 4 |

---

## Quick Reference (once you've done this once)

```bash
# Every new terminal session, before running any Python step:
cd model_training
source venv/bin/activate      # or .\venv\Scripts\activate on Windows

# Record more data:
cd ../data_collection && python capture_sequences.py

# Retrain after new data:
cd ../model_training
python preprocess.py && python train_cnn.py && python convert_tfjs.py
cp -r model_web/* ../web_app/model/ && cp label_map.json ../web_app/model/

# Run the demo:
cd ../web_app && python -m http.server 3000 or python -m http.server 3002 --bind 127.0.0.1
```
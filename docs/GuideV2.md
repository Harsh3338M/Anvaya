# Anvaya — Guide V2: Adding the INCLUDE Video Dataset

V1 (`GUIDE.md`, `BEGINNER_SETUP_GUIDE.md`) covers recording your own signs. This guide adds a
second source of **real motion data**: video clips of Indian Sign Language *words* from the
INCLUDE dataset. Start with the **Pronouns** category, then add more categories the same way.

Read `BEGINNER_SETUP_GUIDE.md` first if you haven't set up Python and recorded anything yet —
this guide assumes your virtual environment works and you've run the pipeline once.

---

## Part 0: What's new in V2

| File | Path | Status | What it does |
|---|---|---|---|
| `video_to_sequences.py` | `data_collection/` | **new** | Converts one video clip into training windows of shape (24, 128) — the same format your webcam recordings use |
| `import_include_videos.py` | `data_collection/` | **new** | Unzips INCLUDE, maps each word folder to a label, converts every clip, saves into `data_collection/dataset/<label>/` |
| `preprocess.py` | `model_training/` | **changed** | Splits train/validation **by source clip** (see below), drops classes with no data, saves group ids |
| `train_cnn.py` | `model_training/` | **changed** | Adds a per-class accuracy report and balanced class weights |

**Why `preprocess.py` had to change.** One video becomes several overlapping windows. The old
random split could put some windows of a clip in training and their near-identical neighbours in
validation — the model would be "tested" on material it had effectively seen, and the accuracy
would look better than it is. Now every window of a clip stays on the same side.

Everything else (`capture_sequences.py`, `convert_tfjs.py`, the whole web app) is unchanged.
Imported data lands in the **same `dataset/` folders** as your own recordings, so training and
export work exactly as before.

---

## Part 1: Download the Pronouns videos

INCLUDE is hosted on Zenodo (free, no account needed). Download both Pronouns parts:

- [Pronouns_1of2.zip](https://zenodo.org/records/4010759/files/Pronouns_1of2.zip?download=1) — about 1.4 GB
- [Pronouns_2of2.zip](https://zenodo.org/records/4010759/files/Pronouns_2of2.zip?download=1) — about 0.95 GB

Keep them as `.zip` files — the importer unzips them for you. The full dataset is 56.8 GB split
into per-category zips (Greetings, Colours, Animals, …); you only ever download the categories you
want. The list of files is on [zenodo.org/records/4010759](https://zenodo.org/records/4010759).

**Credit:** INCLUDE is licensed CC-BY-4.0, which requires attribution. Cite it in your report:
*Sridhar, Ganesan, Kumar, Khapra. "INCLUDE: A Large Scale Dataset for Indian Sign Language
Recognition." ACM Multimedia 2020. doi:10.1145/3394171.3413528.*

---

## Part 2: Preview the import (dry run)

Activate your environment and go to `data_collection/`:

```bash
cd model_training
source venv/bin/activate          # Windows: venv\Scripts\activate
cd ../data_collection
```

Run the importer with `--dry-run` first. It converts nothing — it only unzips (the videos are
unpacked to `data_collection/raw_videos/INCLUDE/`, about 2.4 GB for Pronouns) and prints what it
*would* do:

```bash
python import_include_videos.py --source ~/Downloads/Pronouns_1of2.zip ~/Downloads/Pronouns_2of2.zip --categories Pronouns --dry-run
```
(Windows: use your real paths, e.g. `C:\Users\you\Downloads\Pronouns_1of2.zip`.)

You'll see a table like:

```text
label              clips  status
They                  16  NEW folder
We                    16  NEW folder
I_Word                16  NEW folder  [single character, kept apart from the 'I' letter/digit class]
```

(Numbers and words here are illustrative — yours will show what is actually in the dataset.)

**Check this table before going further.** It is the one place you can catch a wrong label cheaply.

How labels are made:
- INCLUDE folders look like `47. they`. The number prefix is dropped and the word capitalised:
  `They`. Multi-word signs become `Thank_You`-style labels.
- **If a label already exists, it is reused** — case-insensitively. `they`, `They` and `THEY` are
  one class. A label folder is **created only when a clip actually produces data**, never empty.
- **Single-character words are kept apart from letters/digits.** The pronoun *I* is a different
  sign from the fingerspelled letter *I*, so merging them would corrupt both classes. The word
  becomes `I_Word` instead. (It will show as "I Word" in the web app.)
- To rename any label, edit `LABEL_OVERRIDES` near the top of `import_include_videos.py`, e.g.
  `LABEL_OVERRIDES = {"i": "Me"}`.
- If the same label appears from two different categories, the importer warns you — they may be
  different signs that happen to share a word.

---

## Part 3: Pick `--target-fps` (don't skip this)

The model counts **frames, not seconds**: it sees the last 24 frames. If your web app runs at
about 18 frames per second, 24 frames is roughly 1.3 seconds. If the dataset clips were sampled at
a different rate, a sign that takes one second would look faster or slower to the model than the
same sign performed live, and accuracy suffers for no obvious reason.

So make the importer's frame rate match your web app's:

1. Run the web app (`python run_local_server.py` in `web_app/`), open `studio.html`, and hold your
   hands in view while signing.
2. Read the **FPS counter** under the camera. Note a typical value, e.g. `18`.
3. Pass it as `--target-fps 18` in Part 4. If you don't pass it, the default is `20`.

Your own recordings from `capture_sequences.py` run at whatever speed that Python loop manages
(often faster than the browser). That mismatch already exists and isn't new; see Part 8.

---

## Part 4: Run the import

Same command, without `--dry-run`:

```bash
python import_include_videos.py --source ~/Downloads/Pronouns_1of2.zip ~/Downloads/Pronouns_2of2.zip --categories Pronouns --target-fps 18
```

Each clip prints one line:

```text
  [We] MVI_5194.MOV: 7 windows (40 active frames)
  [They] MVI_5201.MOV: skipped (no_hands)
```

At the end you get totals. What the skip reasons mean:
- `no_hands` — hand tracking never found a hand in that clip.
- `too_short` — hands were visible for fewer than 6 frames; too little to learn from.
- `unreadable` — the video couldn't be decoded.

A few skipped clips are normal. If **most** clips are skipped, see Troubleshooting.

Each good clip becomes up to 8 overlapping windows saved as
`data_collection/dataset/<Label>/inc_<clip>_w00.npy`, `..._w01.npy`, … The `inc_` prefix tells them
apart from your own recordings (`seq_###.npy`).

**Re-running is safe.** Clips already imported are skipped, so you can run the same command again
or add more later.

Useful options: `--words we they` (only those words), `--limit-clips 3` (quick test with 3 clips
per word), `--stride`, `--max-windows`. Run `python import_include_videos.py --help` for all of them.

---

## Part 5: Preprocess, train, export

```bash
cd ../model_training
python preprocess.py
python train_cnn.py
python convert_tfjs.py
cp -r model_web/* ../web_app/model/
cp label_map.json ../web_app/model/
```
(Windows: `xcopy model_web\* ..\web_app\model\ /E /Y` and `copy label_map.json ..\web_app\model\`.)

**Read the `preprocess.py` output.** It lists, per class, how many samples are real versus
placeholder (Kaggle), says which classes were skipped for having no data, and warns about classes
built from fewer than 3 independent clips (their validation numbers can't be trusted).

**Read the `train_cnn.py` report, not just the headline number.** At the end it prints:

```text
Per-class validation accuracy (worst first):
  They            50.0%  (6 samples)  mostly mistaken for We (3/6)
  We              66.7%  (6 samples)  mostly mistaken for They (2/6)
  Hello          100.0%  (5 samples)
```

(illustrative). Two signs that keep getting mistaken for each other are exactly what this shows.

Always copy **both** `model_web/*` **and** `label_map.json`. Adding a class changes the order of
labels, so a new model with an old `label_map.json` shows the wrong words.

---

## Part 6: Test it live — the honest check

The validation accuracy printed by `train_cnn.py` is **optimistic**: the same validation set is
used to pick the best checkpoint and to stop training. And it measures how well the model handles
INCLUDE's signers, camera and lighting, not your webcam.

So test the real thing: run the web app, perform each imported word about 5 times yourself, and
note which ones fail. This is the number that matters for your presentation. If a word works on
INCLUDE clips but never live, record a batch of your own samples for it with
`capture_sequences.py` — once a class has 20+ real samples, your recordings and the INCLUDE
windows train together.

---

## Part 7: Adding more categories later

Same steps, different zips:

```bash
python import_include_videos.py --source ~/Downloads/Greetings_1of2.zip ~/Downloads/Greetings_2of2.zip --categories Greetings --dry-run
python import_include_videos.py --source ~/Downloads/Greetings_1of2.zip ~/Downloads/Greetings_2of2.zip --categories Greetings --target-fps 18
```

then repeat Part 5 (retrain on everything — a model can't be extended in place). Words that match
one of your predefined classes (e.g. INCLUDE's *Hello* → your `Hello`) are merged automatically.

To start over for imported data only, delete the `inc_*.npy` files (your own `seq_*.npy` are
untouched):

```bash
# Mac/Linux, from the project root
find data_collection/dataset -name "inc_*.npy" -delete
# Windows PowerShell, from the project root
Get-ChildItem data_collection\dataset -Recurse -Filter "inc_*.npy" | Remove-Item
```

The extracted videos in `data_collection/raw_videos/` can be deleted once imported — the training
data lives in `dataset/`. They're excluded from git.

---

## Part 8: Honest limits — read before you present

1. **Pronouns may be a hard first category.** Pronoun signs are largely pointing gestures, told
   apart by *where* and *in which direction* you point. The model's features are wrist-centred per
   hand: they keep hand shape, orientation and motion shape, but throw away **where in the frame or
   body the hand is**. Depth from a single webcam is also weak. Expect some pronouns to be confused
   with each other; the per-class report will show it. If it's bad, the fix is architectural (add
   hand position as a feature) and touches `normalizer.py`, `capture_sequences.py`, `app.js` and a
   full retrain. It hasn't been built, and shouldn't be until real results show it's needed.
   Words whose signs differ mainly by hand *shape* (many Greetings, Colours) are an easier test.
2. **Few clips per word.** INCLUDE has about 16 clips per word on average. The windows from one
   clip are near-duplicates, so the real amount of variety is the number of *clips*, not windows.
3. **Speed mismatch.** See Part 3. A cleaner long-term fix is making both the capture script and
   the web app resample to a fixed time step; not built yet.
4. **Different signers, camera and background** than your webcam. Fine-tuning with your own
   recordings is what closes this gap.
5. **Regional variation.** ISL has regional variants; INCLUDE was recorded at one school in
   Chennai. Check signs you care about against the official ISLRTC dictionary.
6. **Words only.** This recognises isolated words. It does not understand sentences (see the
   earlier discussion on continuous sign recognition).

---

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `--source is neither a folder nor a .zip` | Wrong path. On Windows, quote paths containing spaces |
| `No matching videos found` | Run with `--dry-run` and no `--categories` to see what was discovered; category names are the folder names (e.g. `Pronouns`) |
| Almost every clip is `no_hands` | Check one video plays upright in a normal player (a rotated video defeats hand tracking). Try `--limit-clips 2` and open the clip |
| `unreadable` for every clip | Your OpenCV can't decode the `.MOV` codec. Convert one clip to `.mp4` with any tool and test |
| Label looks wrong in the dry-run table | Use `LABEL_OVERRIDES` (Part 2) |
| `ModuleNotFoundError: normalizer` | Run the script from inside `data_collection/` |
| Accuracy great in training, poor live | Expected gap — Part 6 and Part 8 |
| Web app shows the wrong words | `label_map.json` wasn't copied along with the model (Part 5) |

---

## Quick reference

```bash
# one-time per category
cd data_collection
python import_include_videos.py --source <zips...> --categories <Category> --dry-run
python import_include_videos.py --source <zips...> --categories <Category> --target-fps 18

# every time data changes
cd ../model_training
python preprocess.py && python train_cnn.py && python convert_tfjs.py
cp -r model_web/* ../web_app/model/ && cp label_map.json ../web_app/model/

# run the app
cd ../web_app && python run_local_server.py
```
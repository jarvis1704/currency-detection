# CLAUDE.md — Indian Currency Recognizer (model training)

You are helping Biprangshu train the model for an offline Android app that recognises Indian banknotes from the camera and speaks the denomination, as an accessibility aid for visually impaired users. **Your job in this repo is the model side only:** prepare the data, train, evaluate, and export `currency_model.tflite` + `labels.txt`. The Android app (Kotlin, Jetpack Compose, CameraX, LiteRT, TextToSpeech) is built separately and depends on the contract in the "Android hand-off" section below. Don't change that contract.

The full training script is in the **Appendix** at the end of this file. If `train_currency.py` doesn't exist in the working directory, create it from the Appendix exactly as written. The script is the source of truth. Edit the script; don't keep parallel notebook copies.

---

## 1. Decisions that are already made (don't reopen them)

| Item | Decision |
|---|---|
| Classes (7) | `10`, `20`, `50`, `100`, `200`, `500`, `background` |
| Out of scope | ₹2000 (withdrawn in 2023), coins, counterfeit detection |
| Old vs new notes | Old and new ₹10, ₹50 and ₹100 are merged into one class each |
| Model | MobileNetV3-Small, ImageNet weights, 224×224 RGB, `include_preprocessing=True` |
| Head | GlobalAvgPool (inside the backbone, `pooling='avg'`) → Dropout 0.3 → Dense(7, softmax) |
| Training | Phase 1: head only, Adam 1e-3, ≤8 epochs. Phase 2: top 40 backbone layers, Adam 1e-5, ≤15 epochs. BatchNorm stays frozen. Label smoothing 0.1. Class weights. EarlyStopping on `val_accuracy`. |
| Export | LiteRT `.tflite`, INT8 weights and activations (representative dataset of 300 training images), **float32 input and output** |
| Data policy | **Public online datasets only.** No personal phone photos for training or testing. |
| Compute | Google Colab T4 GPU |

## 2. Execution environment — read this first

Claude Code only runs commands on the machine where it's installed. Training needs the Colab GPU.

- **Supported setup:** open Colab in the browser and run one notebook cell to mount Drive (`from google.colab import drive; drive.mount('/content/drive')`). Then open the Colab **Terminal** (same VM, so the Drive mount is visible there), install Claude Code (`curl -fsSL https://claude.ai/install.sh | bash`), `cd /content`, put this file in `/content/CLAUDE.md`, and run `claude`. All commands below then run on the GPU VM.
- **VS Code + the Colab extension:** Claude Code runs on the local laptop and can edit files, but it can't run the notebook's cells on the remote GPU. Also, `drive.mount()` has been reported as **not supported** in the extension. If you're in this setup, say so and don't pretend to run training. Either ask the user to run the commands in the Colab Terminal, or move to the supported setup.
- **Check before anything else:**
  `nvidia-smi` (expect a T4), `python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"`, and `ls /content/drive/MyDrive`. If Drive isn't mounted, stop and ask the user to run the mount cell in the notebook.
- Free Colab runtimes reset without warning. Everything that must survive (model, labels, plots, `results.json`, the background rejection list) goes to Drive under `CURRENCY_OUT`. Scratch data goes to `/content/work` and can be rebuilt with `prepare`.
- Paths are environment variables: `CURRENCY_RAW` (input zips, default `/content/drive/MyDrive/currency_raw`), `CURRENCY_OUT` (default `/content/drive/MyDrive/currency`), `CURRENCY_WORK` (default `/content/work`).

## 3. Data sources (all public)

Put these files in `CURRENCY_RAW` (`MyDrive/currency_raw/`). Only the first two must be uploaded by the user.

| Source | What we use | How it gets there |
|---|---|---|
| **IEEE DataPort "Dataset of Indian and Thai Banknotes"** (Vishwakarma University), 256×256 phone photos in class folders | `Indian Currencies/` → `10 New`, `10 Old`, `20`, `50 New`, `50 Old`, `100 New`, `100 Old`, `200`, `500`. Skip `2000` and everything Thai. | Manual: needs a free IEEE login. The user uploads the zip as downloaded (any name). |
| **Roboflow Universe "indian currency/notes"** (omkar patkar, CC BY 4.0) — an **object-detection** project, exported as **YOLOv8** | (a) Images whose label file has only class `None`, or is empty, or is missing → background. (b) Note-free square crops from images that do contain notes (the largest strip outside a rectangle drawn around every note box, +3% margin) → background. | Manual: the user exports YOLOv8 and uploads the zip (any name). There's no `None` folder, so background must be found by reading the labels. The `names` list in `data.yaml` gives the `None` class id (list or dict form). |
| **EgoHands** (Indiana University, research use) — Google Glass frames of hands playing cards, chess, Jenga and puzzles over a table | 3 frames per video × 48 videos ≈ 144 background images. These are the closest public match to "hands in front of a phone camera". | The script tries `https://vision.luddy.iu.edu/egohands_files/egohands_data.zip`, then the old `http://vision.soic.indiana.edu/...` link. **The old link returned an HTML page, not a zip** (`BadZipFile`). If both fail, the user downloads "Labeled Data" from https://vision.soic.indiana.edu/?p=1500 and uploads it as `CURRENCY_RAW/egohands_data.zip`. The script continues without EgoHands if it's missing. |
| **DTD — Describable Textures Dataset** (via `tensorflow_datasets`, `dtd`, ~600 MB, research use) | 2 images × 47 texture classes ≈ 94 surface images (wood, fabric, marble…) | Automatic. Skipped with a warning if the download fails. |

Citations for the report: Bambach et al., ICCV 2015 (EgoHands); Cimpoi et al., CVPR 2014 (DTD); the IEEE DataPort and Roboflow pages.

## 4. How to run

```bash
cd /content
python train_currency.py prepare          # unzip, build the background pool, write contact sheets
# review the contact sheets (section 5), then:
python train_currency.py reject 3 17 40   # numbers from the sheets; repeat as needed
python train_currency.py train            # splits, training, evaluation, export, verification
```

`train` writes the following to `CURRENCY_OUT`: `best.keras`, `labels.txt`, `currency_model.tflite`, `results.json`, `curves.png`, `cm_test.png`, `augmented_samples.png`.

Expect training to take about 10–20 minutes on a T4. Run long commands so that output is logged, e.g. `python train_currency.py train 2>&1 | tee /content/work/train.log`, and read the log rather than guessing.

## 5. Background review — you do this, carefully

The Roboflow export uses mosaic augmentation (4-image collages), and its labels can miss notes. So some background candidates may contain part of a banknote. Training on those teaches the model to stay silent when it sees a note, which is the worst failure for a blind user.

1. After `prepare`, open every `/content/work/contact_sheets/sheet_XX.png` (48 numbered thumbnails each; `index.csv` maps each number to a file).
2. Reject any image showing **any** part of a banknote, including edges, corners, or a note seen through a collage seam. If you're unsure, reject it. Look hardest at `rfnone-*` and `rfcrop-*` files. `egohands-*` and `dtd-*` should be clean, but skim them anyway: EgoHands contains playing cards, which are fine to keep.
3. Run `python train_currency.py reject <numbers...>`. Rejections are appended to `CURRENCY_OUT/rejected_backgrounds.txt` (on Drive) and re-applied automatically on every later `prepare`, so a runtime reset doesn't undo the review.
4. Tell the user how many images you rejected and why (a short list by category is enough). If you rejected more than about 20% of the `rfcrop-*` files, say so. The crop margin may need increasing (`CROP_MARGIN`).
5. Note that `reject` uses the numbers from the **latest** `prepare`. Don't run `prepare` again between reading a sheet and running `reject`.

## 6. Invariants — don't break these

- **Preprocessing contract with Android:** the model takes **float32 RGB in 0–255**, shape `[1,224,224,3]`. Normalisation happens **inside** the model (Keras MobileNetV3 rescaling). Training loads images with `crop_to_aspect_ratio=True` (centre square crop, then bilinear resize to 224). The app does exactly the same. Don't add a Rescaling layer, don't normalise in the data pipeline, and don't change the input size.
- **Labels:** the class order is Keras' alphabetical *string* order: `10, 100, 20, 200, 50, 500, background`. `labels.txt` is written from `CLASS_NAMES`. Never type the list by hand or "fix" the order.
- **Augmentation lives in the tf.data pipeline only** and never in the exported model. It is: rotation ±15°, an independent random 0/90/180/270° turn per image (users can't see which way they're holding the note), zoom −20%…+10%, translation ±10%, brightness ±25%, contrast ±25%, and Gaussian noise std 8 on the 0–255 scale. **No mirroring**, because mirrored digits never occur.
- **No leakage:** never train on `test/`. Exact duplicates are removed with an md5 check. All copies derived from one source (Roboflow `<photo>_jpg.rf.<hash>`, our `<source>.rf.<n>`) stay on the same side of the split via `group_key()`. Keep both mechanisms.
- **BatchNorm stays frozen** in phase 2, and the backbone is called with `training=False`.
- **Export:** INT8 with float32 I/O. After conversion, the script asserts the input/output shapes and dtypes, then measures INT8 accuracy on the test split.

## 7. Bugs already found and fixed — don't reintroduce them

1. `tf.keras.layers.GaussianNoise(8.0)` crashes on Keras 3 (it only allows stddev between 0 and 1). Noise is added by hand: `x + tf.random.normal(..., stddev=8.0)`.
2. TensorFlow ignores EXIF orientation. Every image is re-saved upright with `ImageOps.exif_transpose`.
3. Whole-batch `tf.image.rot90` turned every image in a batch the same way. Rotation is now chosen per image with `tf.gather(..., batch_dims=1)`.
4. An earlier snippet reassigned `OUT` to the background folder, which sent the model outside Drive. The background folder is `BG`; `OUT` is only for persistent outputs.
5. Copying Roboflow images by bare filename let train/valid/test files with the same name overwrite each other. Copies are prefixed with their split.
6. Looping over label files missed images with no label file. The script loops over images; a missing label means background.
7. `EarlyStopping(restore_best_weights=True)` isn't trusted on its own. The script always reloads `best.keras` after phase 2.
8. The old EgoHands link returns HTML. Downloads are validated with `zipfile.is_zipfile` and fall back to a manual upload.

## 8. What to report after `train`

Read `results.json` and report these to the user:
- Train image count per class. Flag any class under 150. If `background` is under ~150, stop and say so before trusting the results.
- `test_accuracy_keras_float32` and `test_accuracy_litert_int8`. INT8 should be within about 2 points of float32. A bigger drop means quantization trouble: check the representative dataset first.
- `background_spoken_as_note_at_0.85`. This should be ~0%. It's the "app speaks at a hand" failure.
- The threshold table. Recommend the lowest threshold where `wrong_when_speaking` is about 0%. The app defaults to 0.85. Label smoothing caps confidence a bit below 1.0, so thresholds above 0.95 may never be reached.
- `tflite_mb`.
- Look at `cm_test.png` and name the worst confusion pair.

## 9. Known risks — state them honestly in the report

- **Source shortcut.** All note images come from IEEE; background comes from Roboflow, EgoHands and DTD. The model can partly learn "which dataset does this look like" instead of "is there a banknote". Mixing three background sources weakens this, but doesn't remove it.
- **No real-camera test set** (by the user's decision). Test accuracy measures performance on held-out dataset images, not on the phone. The test split shares photographic style with training, so it's **optimistic**. The only real check is pointing the finished app at notes, hands and desks.
- The validation split inside `image_dataset_from_directory` is random, not grouped, so validation accuracy for background is slightly optimistic. This only affects early stopping and checkpoint choice; the test split is grouped.
- Don't over-claim in the report: no metric here measures counterfeit robustness or performance in low light.

## 10. Troubleshooting

- `Indian notes folder (containing "10 New") not found` → print the tree of `/content/work/raw` and set `CURRENCY_INDIAN=/path/to/Indian Currencies`.
- `Expected exactly one data.yaml` → the Roboflow zip is missing, or isn't the YOLOv8 export, or two exports were uploaded.
- `No images for class background` → `prepare` wasn't run, or everything got rejected.
- Colab reset mid-run → rerun `prepare` (re-applies rejections), then `train`. `best.keras` on Drive is from the last completed phase-2 epoch that improved.
- `tf.lite.Interpreter is deprecated` warning and long `TensorSpec` lists from `model.export` → harmless.
- Offline smoke test without a GPU or downloads: `CURRENCY_WEIGHTS= CURRENCY_EPOCHS1=1 CURRENCY_EPOCHS2=1 python train_currency.py train` (random weights; accuracy is meaningless, but it proves the pipeline runs).

## 11. Android hand-off (contract — don't change)

- Copy `currency_model.tflite` and `labels.txt` from `CURRENCY_OUT` into `app/src/main/assets/`.
- The app reads labels from `labels.txt`, checks that their count equals the model's output size, centre-crops the frame to a square, resizes to 224×224 bilinear, writes float32 R,G,B values in 0–255, and runs the LiteRT `Interpreter`.
- The app speaks only when the same non-background label wins with confidence ≥ threshold (default 0.85) for 5 frames in a row, and it doesn't repeat until background is seen again.
- Report the threshold you recommend so the user can set `Stabilizer(threshold = …)`.

---

## Appendix — `train_currency.py` (source of truth)

```python
"""
Indian Currency Recognizer — dataset prep, training and LiteRT export.
MobileNetV3-Small, 7 classes: 10, 20, 50, 100, 200, 500, background.

Stages (run in order on a Colab GPU runtime):
  python train_currency.py prepare            # unzip, build background pool, write contact sheets
  python train_currency.py reject 3 17 40     # delete background images that show any part of a note
  python train_currency.py train              # build splits, train, evaluate, export .tflite

Paths come from environment variables (defaults assume Google Drive is mounted in Colab):
  CURRENCY_RAW  folder holding the input zips       (default /content/drive/MyDrive/currency_raw)
  CURRENCY_OUT  folder for model, labels, plots      (default /content/drive/MyDrive/currency)
  CURRENCY_WORK local scratch (fast disk)            (default /content/work)
"""
import os, sys, glob, json, random, shutil, zipfile, hashlib, csv
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps, ImageDraw

RAW_ZIPS = os.environ.get('CURRENCY_RAW', '/content/drive/MyDrive/currency_raw')
OUT      = os.environ.get('CURRENCY_OUT', '/content/drive/MyDrive/currency')
WORK     = os.environ.get('CURRENCY_WORK', '/content/work')
RAW      = f'{WORK}/raw'              # unzipped inputs
BG       = f'{WORK}/background_pool'  # every background candidate, before the split
SHEETS   = f'{WORK}/contact_sheets'   # review grids for the background pool
DATA     = f'{WORK}/dataset'          # train/ and test/ class folders
REJECTED = f'{OUT}/rejected_backgrounds.txt'   # survives runtime resets (on Drive)

CLASSES = ['10', '20', '50', '100', '200', '500', 'background']
EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Background sources (all public, no own photos)
USE_EGOHANDS, EGO_PER_VIDEO = True, 3     # 48 videos -> ~144 first-person hands-over-table frames
USE_DTD, DTD_PER_CLASS      = True, 2     # 47 texture classes -> ~94 surfaces
CROP_MIN, CROP_MARGIN, MAX_CROPS = 0.30, 0.03, 300   # note-free crops from Roboflow currency photos
EGOHANDS_URLS = ['https://vision.luddy.iu.edu/egohands_files/egohands_data.zip',
                 'http://vision.soic.indiana.edu/egohands_files/egohands_data.zip']

# IEEE folder name (lowercase, no spaces/underscores/hyphens) -> class. 2000 is left out on purpose.
MERGE = {'10new': '10', '10old': '10', '20': '20', '20new': '20', '20old': '20',
         '50new': '50', '50old': '50', '100new': '100', '100old': '100',
         '200': '200', '200new': '200', '500': '500', '500new': '500'}

norm = lambda s: s.lower().replace(' ', '').replace('_', '').replace('-', '')


def images_in(folder):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(folder)
                  for f in fs if f.lower().endswith(EXT))


def save_rgb(im, dst, max_side=512):
    im = ImageOps.exif_transpose(im).convert('RGB')
    im.thumbnail((max_side, max_side))
    im.save(dst, 'JPEG', quality=95)


def group_key(path):
    """Images sharing a key stay on the same side of the train/test split.
    Roboflow augmented copies are '<photo>_jpg.rf.<hash>.jpg'; our own derived files use '<source>.rf.<n>.jpg'."""
    b = os.path.basename(path).split('__', 1)[-1]
    return b.split('.rf.')[0] if '.rf.' in b else path


# ---------------------------------------------------------------- prepare
def unzip_all():
    shutil.rmtree(RAW, ignore_errors=True)
    zips = [z for z in sorted(glob.glob(f'{RAW_ZIPS}/*.zip'))
            if os.path.basename(z) != 'egohands_data.zip']        # EgoHands is handled separately
    assert zips, f'No zips in {RAW_ZIPS}'
    for z in zips:
        name = os.path.splitext(os.path.basename(z))[0]
        print('unzipping', os.path.basename(z))
        with zipfile.ZipFile(z) as zf:
            zf.extractall(f'{RAW}/{name}')
    for z in glob.glob(f'{RAW}/**/*.zip', recursive=True):      # nested zips
        print('unzipping nested', z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(os.path.splitext(z)[0])


def find_roboflow():
    yamls = [y for y in glob.glob(f'{RAW}/**/data.yaml', recursive=True)]
    assert len(yamls) == 1, f'Expected exactly one data.yaml (Roboflow YOLOv8 export), found: {yamls}'
    import yaml
    names = yaml.safe_load(open(yamls[0]))['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return os.path.dirname(yamls[0]), names.index('None'), names


def read_label(lbl, none_id):
    """Returns (set of class ids, list of non-None boxes as normalised x0,y0,x1,y1)."""
    ids, boxes = set(), []
    if not os.path.exists(lbl):
        return ids, boxes
    for line in open(lbl):
        p = line.split()
        if not p:
            continue
        cid = int(p[0]); ids.add(cid)
        if cid == none_id or len(p) < 5:
            continue
        c = list(map(float, p[1:]))
        if len(c) == 4:
            boxes.append((c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2))
        else:
            boxes.append((min(c[0::2]), min(c[1::2]), max(c[0::2]), max(c[1::2])))
    return ids, boxes


def roboflow_backgrounds(rf, none_id):
    kept = empty = crops_ok = 0
    crops = []
    for img in glob.glob(f'{rf}/**/images/*', recursive=True):
        if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        split = os.path.basename(os.path.dirname(os.path.dirname(img)))
        lbl = os.path.splitext(img.replace('/images/', '/labels/'))[0] + '.txt'
        ids, boxes = read_label(lbl, none_id)
        if ids <= {none_id}:                       # whole image has no currency
            empty += not ids
            shutil.copy(img, f'{BG}/rfnone-{split}__{os.path.basename(img)}')
            kept += 1
            continue
        if not boxes:
            continue
        # note-free crop: largest strip outside one rectangle drawn around every note (+margin)
        x0 = max(0, min(b[0] for b in boxes) - CROP_MARGIN); y0 = max(0, min(b[1] for b in boxes) - CROP_MARGIN)
        x1 = min(1, max(b[2] for b in boxes) + CROP_MARGIN); y1 = min(1, max(b[3] for b in boxes) + CROP_MARGIN)
        W, H = Image.open(img).size
        strips = [(0, 0, x0, 1), (x1, 0, 1, 1), (0, 0, 1, y0), (0, y1, 1, 1)]
        s = max(strips, key=lambda s: min((s[2] - s[0]) * W, (s[3] - s[1]) * H))
        side = min((s[2] - s[0]) * W, (s[3] - s[1]) * H)
        if side < max(96, CROP_MIN * min(W, H)):
            continue
        cx, cy = (s[0] + s[2]) / 2 * W, (s[1] + s[3]) / 2 * H
        box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
        crops.append((img, box, f'{BG}/rfcrop-{split}__{os.path.basename(img)}'))
    random.shuffle(crops)
    for img, box, dst in crops[:MAX_CROPS]:
        Image.open(img).convert('RGB').crop(box).save(dst, 'JPEG', quality=95)
    print(f'Roboflow: {kept} "None" images ({empty} with empty/missing labels), '
          f'{min(len(crops), MAX_CROPS)} note-free crops (of {len(crops)} possible)')


def egohands_backgrounds():
    import urllib.request
    ego = f'{WORK}/egohands'
    if not glob.glob(f'{ego}/**/*.jpg', recursive=True):
        zp = f'{RAW_ZIPS}/egohands_data.zip'              # manual upload location
        if not (os.path.exists(zp) and zipfile.is_zipfile(zp)):
            zp = f'{WORK}/egohands_data.zip'
            for url in EGOHANDS_URLS:
                try:
                    print('trying', url)
                    urllib.request.urlretrieve(url, zp)
                    if zipfile.is_zipfile(zp):
                        break
                    print('  not a zip (got', os.path.getsize(zp), 'bytes)')
                except Exception as e:
                    print('  failed:', e)
        if not zipfile.is_zipfile(zp):
            print('!! EgoHands unavailable. Download "Labeled Data" from https://vision.soic.indiana.edu/?p=1500, '
                  f'put it at {RAW_ZIPS}/egohands_data.zip and rerun prepare. Continuing without it.')
            return
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(ego)
    videos = {}
    for f in glob.glob(f'{ego}/**/*.jpg', recursive=True):
        videos.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    n = 0
    for v, frames in sorted(videos.items()):
        for k, f in enumerate(random.sample(sorted(frames), min(EGO_PER_VIDEO, len(frames)))):
            save_rgb(Image.open(f), f'{BG}/egohands-{v}.rf.{k}.jpg'); n += 1
    print(f'EgoHands: {len(videos)} videos -> {n} images')


def dtd_backgrounds():
    try:
        import tensorflow_datasets as tfds
        per_class = {}
        for ex in tfds.load('dtd', split='train').shuffle(2000, seed=1):
            c = int(ex['label'])
            if per_class.get(c, 0) < DTD_PER_CLASS:
                save_rgb(Image.fromarray(ex['image'].numpy()), f'{BG}/dtd-{c}-{per_class.get(c, 0)}.jpg')
                per_class[c] = per_class.get(c, 0) + 1
        print(f'DTD: {sum(per_class.values())} images')
    except Exception as e:
        print('!! DTD unavailable, continuing without it:', e)


def contact_sheets(per_sheet=48, cols=8, cell=160):
    shutil.rmtree(SHEETS, ignore_errors=True); os.makedirs(SHEETS)
    files = sorted(glob.glob(f'{BG}/*'))
    with open(f'{SHEETS}/index.csv', 'w', newline='') as f:
        csv.writer(f).writerows(enumerate(os.path.basename(p) for p in files))
    for s in range(0, len(files), per_sheet):
        chunk = files[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * cell, rows * (cell + 18)), 'white')
        d = ImageDraw.Draw(sheet)
        for j, p in enumerate(chunk):
            im = Image.open(p).convert('RGB'); im.thumbnail((cell, cell))
            x, y = (j % cols) * cell, (j // cols) * (cell + 18)
            sheet.paste(im, (x, y + 18)); d.text((x + 4, y + 2), str(s + j), fill='red')
        sheet.save(f'{SHEETS}/sheet_{s // per_sheet:02d}.png')
    print(f'{len(files)} background candidates -> {SHEETS}/sheet_*.png (index.csv maps numbers to files)')


def prepare():
    random.seed(0)
    unzip_all()
    shutil.rmtree(BG, ignore_errors=True); os.makedirs(BG)
    rf, none_id, names = find_roboflow()
    print('Roboflow classes:', names, '| None id =', none_id)
    roboflow_backgrounds(rf, none_id)
    if USE_EGOHANDS: egohands_backgrounds()
    if USE_DTD: dtd_backgrounds()
    # re-apply earlier rejections (kept on Drive so a runtime reset doesn't undo the review)
    if os.path.exists(REJECTED):
        gone = 0
        for name in open(REJECTED).read().split():
            if os.path.exists(f'{BG}/{name}'):
                os.remove(f'{BG}/{name}'); gone += 1
        print(f'Re-applied {gone} earlier rejections from {REJECTED}')
    contact_sheets()


def reject(indices):
    rows = dict((int(i), n) for i, n in csv.reader(open(f'{SHEETS}/index.csv')))
    os.makedirs(OUT, exist_ok=True)
    with open(REJECTED, 'a') as log:
        for i in map(int, indices):
            p = f'{BG}/{rows[i]}'
            if os.path.exists(p):
                os.remove(p); log.write(rows[i] + '\n'); print('rejected', i, rows[i])
    print(len(glob.glob(f'{BG}/*')), 'background images left')


# ---------------------------------------------------------------- train
def build_splits():
    cands = [os.path.dirname(c) for c in glob.glob(f'{RAW}/**/', recursive=True)
             if norm(os.path.basename(os.path.dirname(c))) == '10new']
    cands = [os.path.dirname(c) for c in cands if 'thai' not in norm(os.path.basename(os.path.dirname(c)))]
    INDIAN = os.environ.get('CURRENCY_INDIAN') or (cands[0] if cands else None)
    assert INDIAN, 'Indian notes folder (containing "10 New") not found; set CURRENCY_INDIAN'
    print('INDIAN =', INDIAN)
    groups = {c: [] for c in CLASSES}
    for sub in sorted(os.listdir(INDIAN)):
        cls = MERGE.get(norm(sub))
        if cls is None:
            print(f'  skipping folder "{sub}"'); continue
        found = images_in(os.path.join(INDIAN, sub))
        print(f'  {sub!r:>12} -> {cls:>4}: {len(found)} images')
        groups[cls] += found
    groups['background'] = images_in(BG)
    for c in CLASSES:
        assert groups[c], f'No images for class {c}'

    random.seed(42)
    seen, split = set(), {'train': {}, 'test': {}}
    for cls in CLASSES:
        uniq = []
        for f in groups[cls]:
            h = hashlib.md5(open(f, 'rb').read()).hexdigest()
            if h not in seen:
                seen.add(h); uniq.append(f)
        by_group = {}
        for f in uniq:
            by_group.setdefault(group_key(f), []).append(f)
        keys = sorted(by_group); random.shuffle(keys)
        n_test, test, train = int(0.15 * len(uniq)), [], []
        for k in keys:
            (test if len(test) < n_test else train).extend(by_group[k])
        split['test'][cls], split['train'][cls] = test, train
        print(f'{cls:>10}: {len(uniq)} unique from {len(by_group)} sources '
              f'({len(groups[cls]) - len(uniq)} exact duplicates dropped)')

    shutil.rmtree(DATA, ignore_errors=True)
    jobs = []
    for sp, by_cls in split.items():
        for cls, fl in by_cls.items():
            d = f'{DATA}/{sp}/{cls}'; os.makedirs(d, exist_ok=True)
            jobs += [(f, f'{d}/{i:05d}.jpg') for i, f in enumerate(fl)]
    def prep(job):
        try:
            save_rgb(Image.open(job[0]), job[1]); return True
        except Exception as e:
            print('  bad image skipped:', job[0], e); return False
    with ThreadPoolExecutor(8) as ex:
        ok = sum(ex.map(prep, jobs))
    print(f'wrote {ok}/{len(jobs)} images to {DATA}')
    for cls in CLASSES:
        n = len(os.listdir(f'{DATA}/train/{cls}'))
        print(f'train/{cls:<10} {n}' + ('   <-- LOW, aim for 150+' if n < 150 else ''))


def train():
    import tensorflow as tf
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    os.makedirs(OUT, exist_ok=True)
    print('TF', tf.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))
    build_splits()

    IMG, BATCH, SEED = 224, 32, 42
    AUTOTUNE = tf.data.AUTOTUNE
    EPOCHS1 = int(os.environ.get('CURRENCY_EPOCHS1', 8))
    EPOCHS2 = int(os.environ.get('CURRENCY_EPOCHS2', 15))
    WEIGHTS = os.environ.get('CURRENCY_WEIGHTS', 'imagenet') or None   # '' only for offline tests

    def load(path, **kw):
        return tf.keras.utils.image_dataset_from_directory(
            path, image_size=(IMG, IMG), batch_size=BATCH, label_mode='categorical',
            crop_to_aspect_ratio=True, seed=SEED, **kw)

    raw_train = load(f'{DATA}/train', validation_split=0.15, subset='training')
    raw_val   = load(f'{DATA}/train', validation_split=0.15, subset='validation')
    raw_test  = load(f'{DATA}/test', shuffle=False)
    CLASS_NAMES = raw_train.class_names           # alphabetical strings: 10,100,20,200,50,500,background
    NUM_CLASSES = len(CLASS_NAMES)
    counts = np.array([len(os.listdir(f'{DATA}/train/{c}')) for c in CLASS_NAMES])
    class_weight = {i: counts.sum() / (NUM_CLASSES * n) for i, n in enumerate(counts)}
    print(CLASS_NAMES, dict(zip(CLASS_NAMES, counts.tolist())))

    # Augmentation: training pipeline only, never part of the exported model
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom((-0.2, 0.1)),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.25),
    ], name='augment')

    def rotate_per_image(x):                       # independent 0/90/180/270 turn per image
        k = tf.random.uniform([tf.shape(x)[0]], 0, 4, dtype=tf.int32)
        rots = tf.stack([tf.image.rot90(x, i) for i in range(4)], axis=1)
        return tf.gather(rots, k, batch_dims=1)

    def train_aug(x, y):
        x = augment(x, training=True)
        x = rotate_per_image(x)
        x = x + tf.random.normal(tf.shape(x), stddev=8.0)   # Keras GaussianNoise rejects stddev > 1
        return tf.clip_by_value(x, 0.0, 255.0), y

    train_ds = raw_train.map(train_aug, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds, test_ds = raw_val.prefetch(AUTOTUNE), raw_test.prefetch(AUTOTUNE)

    x, y = next(iter(train_ds))
    plt.figure(figsize=(9, 9))
    for i in range(min(16, len(x))):
        plt.subplot(4, 4, i + 1); plt.imshow(x[i].numpy().astype('uint8'))
        plt.title(CLASS_NAMES[int(np.argmax(y[i]))]); plt.axis('off')
    plt.savefig(f'{OUT}/augmented_samples.png', dpi=100, bbox_inches='tight'); plt.close()

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG, IMG, 3), include_top=False, weights=WEIGHTS,
        pooling='avg', include_preprocessing=True)            # expects raw 0-255 pixels
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG, IMG, 3))
    h = base(inputs, training=False)
    h = tf.keras.layers.Dropout(0.3)(h)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(h)
    model = tf.keras.Model(inputs, outputs)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Phase 1: head only
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS1, class_weight=class_weight,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,
                                                               restore_best_weights=True)])
    # Phase 2: top 40 backbone layers, BatchNorm frozen, 100x lower LR
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss, metrics=['accuracy'])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS2, class_weight=class_weight,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4,
                                                               restore_best_weights=True),
                              tf.keras.callbacks.ModelCheckpoint(f'{OUT}/best.keras', monitor='val_accuracy',
                                                                 save_best_only=True)])
    model = tf.keras.models.load_model(f'{OUT}/best.keras')
    with open(f'{OUT}/labels.txt', 'w') as f:
        f.write('\n'.join(CLASS_NAMES))

    acc = h1.history['accuracy'] + h2.history['accuracy']
    val = h1.history['val_accuracy'] + h2.history['val_accuracy']
    plt.figure(figsize=(7, 4)); plt.plot(acc, label='train'); plt.plot(val, label='validation')
    plt.axvline(len(h1.history['accuracy']) - 0.5, ls='--', c='gray', label='start of fine-tuning')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
    plt.savefig(f'{OUT}/curves.png', dpi=150, bbox_inches='tight'); plt.close()

    # Evaluation on the held-out test split
    y_true = np.concatenate([np.argmax(yb, axis=1) for _, yb in test_ds])
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    test_acc = float((y_true == y_pred).mean())
    print('test accuracy:', round(test_acc, 4))
    print(classification_report(y_true, y_pred, labels=range(NUM_CLASSES), target_names=CLASS_NAMES,
                                digits=3, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES)),
                           display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, xticks_rotation=45)
    fig.savefig(f'{OUT}/cm_test.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # Speaking threshold trade-off
    conf, bg = y_prob.max(axis=1), CLASS_NAMES.index('background')
    notes, thresholds = y_true != bg, {}
    for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        speaks = (conf >= t) & (y_pred != bg)
        cov = float((speaks & notes).sum() / max(notes.sum(), 1))
        wrong = float((speaks & (y_pred != y_true)).sum() / max(speaks.sum(), 1))
        thresholds[t] = {'speaks_on_notes': cov, 'wrong_when_speaking': wrong}
        print(f'threshold {t:.2f}: speaks on {cov:6.1%} of note images, wrong {wrong:5.1%}')
    bg_as_note = float(((y_true == bg) & (y_pred != bg) & (conf >= 0.85)).sum() / max((y_true == bg).sum(), 1))
    print(f'background images that would be SPOKEN as a note at 0.85: {bg_as_note:.1%}')

    # LiteRT export: INT8 weights/activations, float32 input/output
    model.export(f'{WORK}/saved_model')
    def representative_data():
        for xb, _ in raw_train.unbatch().take(300):
            yield [tf.expand_dims(tf.cast(xb, tf.float32), 0)]
    conv = tf.lite.TFLiteConverter.from_saved_model(f'{WORK}/saved_model')
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_data
    tfl = conv.convert()
    tflite_path = f'{OUT}/currency_model.tflite'
    open(tflite_path, 'wb').write(tfl)

    interp = tf.lite.Interpreter(model_path=tflite_path); interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    assert list(inp['shape']) == [1, IMG, IMG, 3] and inp['dtype'] == np.float32, inp
    assert list(out['shape']) == [1, NUM_CLASSES] and out['dtype'] == np.float32, out
    correct = total = 0
    for xb, yb in raw_test.unbatch():
        interp.set_tensor(inp['index'], tf.expand_dims(tf.cast(xb, tf.float32), 0).numpy())
        interp.invoke()
        correct += int(np.argmax(interp.get_tensor(out['index'])) == np.argmax(yb)); total += 1
    tfl_acc = correct / total

    results = {'class_names': CLASS_NAMES, 'train_counts': dict(zip(CLASS_NAMES, counts.tolist())),
               'test_accuracy_keras_float32': round(test_acc, 4),
               'test_accuracy_litert_int8': round(tfl_acc, 4),
               'background_spoken_as_note_at_0.85': round(bg_as_note, 4),
               'thresholds': thresholds, 'tflite_mb': round(len(tfl) / 1e6, 2)}
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=2)
    print(json.dumps(results, indent=2))
    print('Done. Copy', tflite_path, 'and', f'{OUT}/labels.txt', 'into app/src/main/assets/')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'prepare':
        prepare()
    elif cmd == 'reject':
        reject(sys.argv[2:])
    elif cmd == 'train':
        train()
    else:
        print(__doc__)
```

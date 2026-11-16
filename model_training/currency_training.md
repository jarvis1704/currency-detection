# CLAUDE.md — Indian Currency Recognizer (model training)

You are helping Biprangshu train the model for an offline Android app that recognises Indian banknotes from the camera and speaks the denomination, as an accessibility aid for visually impaired users. **Your job in this repo is the model side only:** prepare the data, train, evaluate, and export `currency_model.tflite` + `labels.txt`. The Android app (Kotlin, Jetpack Compose, CameraX, LiteRT, TextToSpeech) is built separately and depends on the contract in the "Android hand-off" section below. Don't change that contract.

The full training script is in the **Appendix** at the end of this file. If `train_currency.py` doesn't exist in the working directory, create it from the Appendix exactly as written. The training code in step 3 of `currency_training.ipynb` is the source of truth. The `train_currency.py` file in the repo is the older version; don't edit it.

---

## 1. Decisions that are already made (don't reopen them)

| Item | Decision |
|---|---|
| Classes (7) | `10`, `20`, `50`, `100`, `200`, `500`, `background` |
| Out of scope | ₹2000 (withdrawn in 2023), coins, counterfeit detection |
| Old vs new notes | ₹10, ₹50 and ₹100 are learned as separate old/new classes (10 fine classes) and summed at export, so the app sees one class each |
| Model | MobileNetV3-Small, ImageNet weights, 224×224 RGB, `include_preprocessing=True` |
| Head | GlobalAvgPool (inside the backbone, `pooling='avg'`) → Dropout 0.3 → Dense(10, softmax) → fixed Dense(7) that sums series probabilities |
| Training | Phase 1: head only, Adam 1e-3, 8 epochs. Phase 2: whole backbone, Adam 2e-4 with cosine decay, ≤60 epochs, patience 12 on `val_coarse_accuracy`. BatchNorm stays frozen. Label smoothing 0.05. Inverse-frequency sample weights. |
| Export | LiteRT `.tflite`, plain **float32** (no quantization), float32 input and output. INT8 scored 70.9% against 85.8% for the same weights on MobileNetV3-Small |
| Data policy | **Public online datasets only.** No personal phone photos for training or testing. |
| Compute | Google Colab T4 GPU |

## 2. Execution environment — read this first

Claude Code only runs commands on the machine where it's installed. Training needs the Colab GPU.

- **Supported setup:** open Colab in the browser and run one notebook cell to mount Drive (`from google.colab import drive; drive.mount('/content/drive')`). Then open the Colab **Terminal** (same VM, so the Drive mount is visible there), install Claude Code (`curl -fsSL https://claude.ai/install.sh | bash`), `cd /content`, put this file in `/content/CLAUDE.md`, and run `claude`. All commands below then run on the GPU VM.
- **VS Code + the Colab extension (the user's setup):** the notebook runs on the remote GPU from VS Code. Mount Drive with `drive.mount('/content/drive')` in the first notebook cell, as in Colab on the web; it works in the extension. `files.download()` isn't relied on: finished files are fetched from Drive. Claude Code on the laptop can edit files but can't run cells on the remote GPU, so don't pretend to run training, and don't download datasets or install TensorFlow on the laptop.
- **Check before anything else:**
  `nvidia-smi` (expect a T4), `python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"`, and `ls /content/drive/MyDrive`. If Drive isn't mounted, stop and ask the user to run the mount cell in the notebook.
- Free Colab runtimes reset without warning. Everything that must survive (model, labels, plots, `results.json`, the background rejection list) goes to Drive under `CURRENCY_OUT`. Scratch data goes to `/content/work` and can be rebuilt with `prepare`.
- Paths are environment variables: `CURRENCY_RAW` (input zips, default `/content/drive/MyDrive/currency_raw`), `CURRENCY_OUT` (default `/content/drive/MyDrive/currency`), `CURRENCY_WORK` (default `/content/work`).

## 3. Data sources (all public)

Put these files in `CURRENCY_RAW` (`MyDrive/currency_raw/`). Only the first two must be uploaded by the user.

| Source | What we use | How it gets there |
|---|---|---|
| **IEEE DataPort "Dataset of Indian and Thai Banknotes"** (Vishwakarma University), 256×256 phone photos in class folders | `Indian Currencies/` → `10 New`, `10 Old`, `20`, `50 New`, `50 Old`, `100 New`, `100 Old`, `200`, `500`, keeping the series as fine labels. Skip `2000` and everything Thai. | Manual: needs a free IEEE login. The user uploads the zip as downloaded (any name). |
| **Roboflow Universe "indian currency/notes"** (omkar patkar, CC BY 4.0) — an **object-detection** project, exported as **YOLOv8** | (a) One square crop around the largest note of each single-denomination photo → note image, series unknown. (b) Images whose label file has only class `None`, or is empty, or is missing → background. (c) Note-free square crops from images with notes → background. | Manual: the user exports YOLOv8 and uploads the zip (any name). The `names` list in `data.yaml` gives the `None` class id (list or dict form). |
| **EgoHands** (Indiana University, research use) — Google Glass frames of hands playing cards, chess, Jenga and puzzles over a table | 3 frames per video × 48 videos ≈ 144 background images. These are the closest public match to "hands in front of a phone camera". | The script tries `https://vision.luddy.iu.edu/egohands_files/egohands_data.zip`, then the old `http://vision.soic.indiana.edu/...` link. **The old link returned an HTML page, not a zip** (`BadZipFile`). If both fail, the user downloads "Labeled Data" from https://vision.soic.indiana.edu/?p=1500 and uploads it as `CURRENCY_RAW/egohands_data.zip`. The script continues without EgoHands if it's missing. |
| **DTD — Describable Textures Dataset** (direct download, ~625 MB, research use) | 6 images × 47 texture classes ≈ 280 surface images | Automatic. Skipped with a warning if the download fails. |
| **Imagenette** (fast.ai, 320px, ~340 MB) | 30 images × 10 classes ≈ 300 everyday-object images → background | Automatic. Skipped with a warning if the download fails. |
| **Mendeley 48ympv8jjf** (CC BY 4.0) | Augmented copies only, 1000 per denomination. Up to 3 copies per original photo, 500 per denomination. Series unknown. Some copies are flipped, and they can't be told apart by name. | Automatic, cached in `CURRENCY_OUT/cache`. |
| **Mendeley jh6979fg2t** (CC BY 4.0) | Spoilt ₹10/20/50/100 with series labels, capped at 250 per label | Automatic, cached. |
| **Mendeley 8ckhkssyn3** (CC BY 4.0) | Phone photos, no augmentation. **External test set: never trained on.** | Automatic, cached. |

Citations for the report: Bambach et al., ICCV 2015 (EgoHands); Cimpoi et al., CVPR 2014 (DTD); the IEEE DataPort and Roboflow pages.

## 4. How to run

```bash
cd /content
python train_currency.py prepare          # unzip, build the background pool, write contact sheets
# review the contact sheets (section 5), then:
python train_currency.py reject 3 17 40   # numbers from the sheets; repeat as needed
python train_currency.py train            # splits, training, evaluation, export, verification
```

`train` writes the following to `CURRENCY_OUT`: `best.weights.h5`, `best.keras`, `labels.txt`, `currency_model.tflite`, `previous_currency_model.tflite` (the model from before the first new run, kept as baseline), `results.json`, `curves.png`, `cm_test.png`, `cm_external.png`, `augmented_samples.png`.

Expect about an hour of training on a T4. Run long commands so that output is logged, e.g. `python train_currency.py train 2>&1 | tee /content/work/train.log`, and read the log rather than guessing.

## 5. Background review — you do this, carefully

The Roboflow export uses mosaic augmentation (4-image collages), and its labels can miss notes. So some background candidates may contain part of a banknote. Training on those teaches the model to stay silent when it sees a note, which is the worst failure for a blind user.

1. After `prepare`, open every `/content/work/contact_sheets/sheet_XX.png` (48 numbered thumbnails each; `index.csv` maps each number to a file). Only Roboflow-derived backgrounds are on the sheets, since the other sources can't contain notes.
2. Reject any image showing **any** part of a banknote, including edges, corners, or a note seen through a collage seam. If you're unsure, reject it.
3. Run `python train_currency.py reject <numbers...>`. Rejections are appended to `CURRENCY_OUT/rejected_backgrounds.txt` (on Drive) and re-applied automatically on every later `prepare`, so a runtime reset doesn't undo the review.
4. Tell the user how many images you rejected and why (a short list by category is enough). If you rejected more than about 20% of the `rfcrop-*` files, say so. The crop margin may need increasing (`CROP_MARGIN`).
5. Note that `reject` uses the numbers from the **latest** `prepare`. Don't run `prepare` again between reading a sheet and running `reject`.

## 6. Invariants — don't break these

- **Preprocessing contract with Android:** the model takes **float32 RGB in 0–255**, shape `[1,224,224,3]`. Normalisation happens **inside** the model (Keras MobileNetV3 rescaling). Evaluation uses a centre square crop, then bilinear resize to 224, and the app does exactly the same. Training picks the square at a random position, so wide photos don't always lose both ends. Don't add a Rescaling layer, don't normalise in the data pipeline, and don't change the input size.
- **Labels:** the class order is Keras' alphabetical *string* order: `10, 100, 20, 200, 50, 500, background` (`COARSE` in the script). `labels.txt` is written from it. Never type the list by hand or "fix" the order. A label like `10` means series unknown; the loss gives credit to either series.
- **Augmentation lives in the tf.data pipeline only** and never in the exported model. It is: rotation ±15°, an independent random 0/90/180/270° turn per image, zoom −20%…+35%, translation ±10%, perspective where Keras has it, brightness ±25%, contrast ±25%, blur, white balance ±8%, a random occluding block, and Gaussian noise std 8 on the 0–255 scale. **No mirroring and no hue shift**: mirrored digits never occur, and colour separates denominations.
- **No leakage:** never train on `test` or `external`. Exact duplicates are removed with md5. Copies of one photo (`group_key()`) and images with equal perceptual hashes share a group, and each group gets one split. Training images within Hamming distance 4 of an external image are dropped.
- **BatchNorm stays frozen** in phase 2, and the backbone is called with `training=False`.
- **Export:** float32 with float32 I/O. After conversion, the script asserts the input/output shapes and dtypes, and measures accuracy on the exported file, not the Keras model.

## 7. Bugs already found and fixed — don't reintroduce them

1. `tf.keras.layers.GaussianNoise(8.0)` crashes on Keras 3 (it only allows stddev between 0 and 1). Noise is added by hand: `x + tf.random.normal(..., stddev=8.0)`.
2. TensorFlow ignores EXIF orientation. Every image is re-saved upright with `ImageOps.exif_transpose`.
3. Whole-batch `tf.image.rot90` turned every image in a batch the same way. Rotation is now chosen per image with `tf.gather(..., batch_dims=1)`.
4. An earlier snippet reassigned `OUT` to the background folder, which sent the model outside Drive. The background folder is `BG`; `OUT` is only for persistent outputs.
5. Copying Roboflow images by bare filename let train/valid/test files with the same name overwrite each other. Copies are prefixed with their split.
6. Looping over label files missed images with no label file. The script loops over images; a missing label means background.
7. `EarlyStopping(restore_best_weights=True)` isn't trusted on its own. The script always reloads `best.keras` after phase 2.
8. The old EgoHands link returns HTML. Downloads are validated with `zipfile.is_zipfile` and fall back to a manual upload.
9. The previous run lost EgoHands and DTD silently, so every background came from Roboflow and every note from IEEE. DTD now downloads directly, with no `tensorflow_datasets` dependency.
10. INT8 export failed under the default XNNPACK delegate ("failed to prepare") and cost 15 accuracy points. Float32 avoids both.
11. Fine-tuning at 1e-5 for 15 epochs was still improving on its last epoch. Don't go back to it.

## 8. What to report after `train`

Read `results.json` and report these to the user:
- Train image count per class. Flag any class under 150.
- `external_accuracy` against `external_accuracy_previous_model`: the only before/after that counts. `test_accuracy` shares sources with training, so it's optimistic. Don't ship a model that doesn't beat the baseline.
- `background_spoken_as_note`, at the recommended threshold. This should be ~0%. It's the "app speaks at a hand" failure.
- `recommended_threshold`: the lowest threshold with ≤1% wrong speech on validation. `thresholds_external` shows how it holds up on unseen photos, single image at a time.
- `tflite_mb`.
- Look at `cm_external.png` and name the worst confusion pair.

## 9. Known risks — state them honestly in the report

- **Source shortcut.** Notes now come from several sources, and backgrounds partly from the same Roboflow photos, which weakens "which dataset does this look like". It isn't eliminated.
- **No real-camera test set** (by the user's decision). The external test set is a different dataset, so it's much fairer than the internal split, but it isn't the phone. The only real check is pointing the finished app at notes, hands and desks. Series labels are unknown on some sources, so old/new accuracy can't be reported separately.
- The external set has no background images, so the background figure comes from the internal test split.
- Don't over-claim in the report: no metric here measures counterfeit robustness or performance in low light.

## 10. Troubleshooting

- `Indian notes folder (containing "10 New") not found` → print the tree of `/content/work/raw` and set `CURRENCY_INDIAN=/path/to/Indian Currencies`.
- `Expected exactly one data.yaml` → the Roboflow zip is missing, or isn't the YOLOv8 export, or two exports were uploaded.
- `no folder name gave a denomination and series` → the Mendeley spoilt-notes zip layout changed; print its `namelist()` and adjust `fetch_spoilt`.
- Too many Mendeley downloads failed → rerun `prepare`; finished datasets come from `CURRENCY_OUT/cache`.
- `No images for class background` → `prepare` wasn't run, or everything got rejected.
- Colab reset mid-run → rerun `prepare` (re-applies rejections, restores downloads from cache), then `train`.
- `tf.lite.Interpreter is deprecated` warning and long `TensorSpec` lists from `model.export` → harmless.
- Offline smoke test without a GPU or downloads: run `prepare` with `CURRENCY_OFFLINE=1` and images placed under `CURRENCY_WORK/notes/mendeley-8ckhkssyn3/<label>/`, then `CURRENCY_WEIGHTS= CURRENCY_EPOCHS1=1 CURRENCY_EPOCHS2=1 python train_currency.py train` (random weights; accuracy is meaningless, but it proves the pipeline runs).

## 11. Android hand-off (contract — don't change)

- Copy `currency_model.tflite` and `labels.txt` from `CURRENCY_OUT` into `app/src/main/assets/`.
- The app reads labels from `labels.txt`, checks that their count equals the model's output size, centre-crops the frame to a square, resizes to 224×224 bilinear, writes float32 R,G,B values in 0–255, and runs the LiteRT `Interpreter`.
- The app speaks only when the same non-background label wins with confidence ≥ threshold (default 0.70) in 4 of the last 6 frames, and it doesn't repeat until background is seen again.
- Report `recommended_threshold` so the user can set `Stabilizer(threshold = …)`.

---

## Appendix — copy of the training code from the notebook

```python
"""
Indian Currency Recognizer — dataset prep, training and LiteRT export.
MobileNetV3-Small. The network learns 10 fine classes (old and new series of 10, 50 and 100 apart);
the exported model merges them, so the app still sees 7: 10, 20, 50, 100, 200, 500, background.

Stages (run in order on a Colab GPU runtime):
  python train_currency.py prepare            # unzip, download public datasets, build pools, write contact sheets
  python train_currency.py reject 3 17 40     # delete background images that show any part of a note
  python train_currency.py train              # split, train, evaluate, export .tflite

Paths come from environment variables (defaults assume Google Drive is mounted in Colab):
  CURRENCY_RAW  folder holding the input zips       (default /content/drive/MyDrive/currency_raw)
  CURRENCY_OUT  folder for model, labels, plots      (default /content/drive/MyDrive/currency)
  CURRENCY_WORK local scratch (fast disk)            (default /content/work)
"""
import os, sys, glob, json, random, re, shutil, tarfile, zipfile, hashlib, csv, io
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps, ImageDraw

RAW_ZIPS = os.environ.get('CURRENCY_RAW', '/content/drive/MyDrive/currency_raw')
OUT      = os.environ.get('CURRENCY_OUT', '/content/drive/MyDrive/currency')
WORK     = os.environ.get('CURRENCY_WORK', '/content/work')
OFFLINE  = bool(os.environ.get('CURRENCY_OFFLINE'))   # skip all downloads (pipeline tests only)
RAW      = f'{WORK}/raw'              # unzipped inputs
BG       = f'{WORK}/background_pool'  # every background candidate, before the split
NOTES    = f'{WORK}/notes'            # <source>/<label>/<image>.jpg, every note image
SHEETS   = f'{WORK}/contact_sheets'   # review grids for the background pool
CACHE    = f'{OUT}/cache'             # resized downloads on Drive, so a runtime reset needs no re-download
REJECTED = f'{OUT}/rejected_backgrounds.txt'   # survives runtime resets (on Drive)

# What the app sees. Alphabetical *string* order: this is the labels.txt order.
COARSE = sorted(['10', '20', '50', '100', '200', '500', 'background'])
# What the network learns. The old and new series of these denominations look nothing alike.
FINE = ['10_new', '10_old', '100_new', '100_old', '20', '200', '50_new', '50_old', '500', 'background']
DENOMINATIONS = {'10', '20', '50', '100', '200', '500'}
EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
UA = {'User-Agent': 'Mozilla/5.0'}

# Sources that never enter training. Its accuracy is the honest "new photographer, new notes" number.
EXTERNAL_TEST = 'mendeley-8ckhkssyn3'
# Mendeley id -> version. All CC BY 4.0.
FOLDER_DATASETS = {'8ckhkssyn3': 1, '48ympv8jjf': 2}     # one folder per denomination, images fetched one by one
SPOILT = ('jh6979fg2t', 4)                               # worn and torn 10/20/50/100, one big zip
AUGMENTED_DATASET = '48ympv8jjf'                         # holds only augmented copies: 1000 per denomination
AUGMENTED_CAP, AUGMENTED_PER_PHOTO = 500, 3              # keep copies of one photo from dominating
SPOILT_CAP = 250                                         # per label, or low denominations swamp 200 and 500

# Background sources (all public, no own photos)
USE_EGOHANDS, EGO_PER_VIDEO = True, 3     # optional manual upload, see README section in the notebook
DTD_PER_CLASS = 6                         # 47 texture classes -> ~280 surfaces
IMAGENETTE_PER_CLASS = 30                 # 10 everyday object classes -> ~300 images
CROP_MIN, CROP_MARGIN, MAX_CROPS = 0.30, 0.03, 300   # note-free crops from Roboflow currency photos
EGOHANDS_URLS = ['https://vision.luddy.iu.edu/egohands_files/egohands_data.zip',
                 'http://vision.soic.indiana.edu/egohands_files/egohands_data.zip']
DTD_URL = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'

# IEEE folder name (lowercase, no spaces/underscores/hyphens) -> fine label. 2000 is left out on purpose.
MERGE = {'10new': '10_new', '10old': '10_old', '20': '20', '20new': '20', '20old': '20',
         '50new': '50_new', '50old': '50_old', '100new': '100_new', '100old': '100_old',
         '200': '200', '200new': '200', '500': '500', '500new': '500'}

norm = lambda s: s.lower().replace(' ', '').replace('_', '').replace('-', '')


def coarse_of(label):
    """'10_new' -> '10'. Also accepts labels that are already coarse."""
    return label.split('_')[0]


def mask_of(label):
    """Which fine classes a label allows. '10_new' allows one; '10' (series unknown) allows both 10s."""
    return [float(f == label or coarse_of(f) == label) for f in FINE]


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
def fetch(url, dst, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 20)


def fetch_bytes(url, tries=4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if attempt == tries - 1:
                raise e


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


def restore_cache(source):
    """Unpack a finished download from Drive. True if there was one."""
    z = f'{CACHE}/{source}.zip'
    if not (os.path.exists(z) and zipfile.is_zipfile(z)):
        return False
    with zipfile.ZipFile(z) as zf:
        zf.extractall(f'{NOTES}/{source}')
    print(f'{source}: restored {len(images_in(f"{NOTES}/{source}"))} images from Drive cache')
    return True


def store_cache(source):
    print(f'{source}: saving to Drive cache')
    os.makedirs(CACHE, exist_ok=True)
    tmp = shutil.make_archive(f'{WORK}/{source}', 'zip', f'{NOTES}/{source}')     # JPEGs, nothing to compress
    shutil.copy(tmp, f'{CACHE}/{source}.zip.tmp')
    os.replace(f'{CACHE}/{source}.zip.tmp', f'{CACHE}/{source}.zip')
    os.remove(tmp)


def mendeley_api(path):
    url = f'https://data.mendeley.com/public-api/datasets/{path}'
    return json.loads(fetch_bytes(url))


def mendeley_files(dataset, version):
    """[(folder name, file name, download url)] for one Mendeley dataset, skipping non-denomination folders."""
    folders = mendeley_api(f'{dataset}/folders/{version}')
    def list_folder(folder):
        files = mendeley_api(f'{dataset}/files?folder_id={folder["id"]}&version={version}')
        return [(folder['name'], f['filename'], f['content_details']['download_url']) for f in files]
    with ThreadPoolExecutor(8) as ex:
        return [item for chunk in ex.map(list_folder, folders) for item in chunk]


def denomination_of(folder):
    m = re.search(r'\d+', folder)
    return m.group() if m and m.group() in DENOMINATIONS else None


def select_augmented(items):
    """The augmented Mendeley set: few copies per original photo, capped per denomination.
    Files are named '<photo>.jpg_<uuid>.jpg', so the part before '.jpg_' identifies the photo."""
    by_photo = {}
    for item in items:
        by_photo.setdefault((item[0], item[1].split('.jpg_')[0]), []).append(item)
    rng, chosen, per_class = random.Random(7), [], {}
    keys = sorted(by_photo)
    rng.shuffle(keys)
    for key in keys:
        if per_class.get(key[0], 0) >= AUGMENTED_CAP:
            continue
        copies = sorted(by_photo[key], key=lambda i: i[1])
        rng.shuffle(copies)
        for k, (folder, _, url) in enumerate(copies[:AUGMENTED_PER_PHOTO]):
            chosen.append((folder, f'{re.sub(r"[^A-Za-z0-9]", "_", key[1])}.rf.{k}.jpg', url))
        per_class[key[0]] = per_class.get(key[0], 0) + min(len(copies), AUGMENTED_PER_PHOTO)
    print(f'  {AUGMENTED_DATASET}: {len(by_photo)} original photos -> {len(chosen)} copies selected')
    return chosen


def fetch_folder_dataset(dataset, version):
    source = f'mendeley-{dataset}'
    if restore_cache(source):
        return
    print(f'{source}: downloading')
    items = [i for i in mendeley_files(dataset, version) if denomination_of(i[0])]
    assert items, f'{source}: the Mendeley listing is empty'
    if dataset == AUGMENTED_DATASET:
        items = select_augmented(items)

    def job(item):
        folder, name, url = item
        label = denomination_of(folder)
        os.makedirs(f'{NOTES}/{source}/{label}', exist_ok=True)
        stem = os.path.splitext(name)[0] if dataset != AUGMENTED_DATASET else name[:-4]
        dst = f'{NOTES}/{source}/{label}/{stem}.jpg'
        if os.path.exists(dst):                   # already fetched by an earlier, interrupted run
            return True
        try:
            with Image.open(io.BytesIO(fetch_bytes(url))) as im:
                im.draft('RGB', (1024, 1024))     # decode 12 MP photos at reduced size: far less memory
                save_rgb(im, dst + '.part')
            os.replace(dst + '.part', dst)        # a killed run can't leave half an image behind
            return True
        except Exception as e:
            print('  failed:', folder, name, e)
            return False

    print(f'  {len(items)} images to fetch; progress is printed every 100')
    ok = done = 0
    with ThreadPoolExecutor(12) as ex:
        for success in ex.map(job, items):
            ok += success; done += 1
            if done % 100 == 0:
                print(f'  {done}/{len(items)}', flush=True)
    print(f'{source}: {ok}/{len(items)} images')
    assert ok >= 0.9 * len(items), f'{source}: too many downloads failed; rerun prepare (a cached dataset is not fetched again)'
    store_cache(source)


def fetch_spoilt():
    dataset, version = SPOILT
    source = f'mendeley-{dataset}'
    if restore_cache(source):
        return
    files = mendeley_api(f'{dataset}/files?folder_id=root&version={version}')
    url = next(f['content_details']['download_url'] for f in files if f['filename'].lower().endswith('.zip'))
    path = f'{WORK}/{source}.download.zip'
    print(f'{source}: downloading the zip (about 760 MB)')
    fetch(url, path, timeout=600)

    by_label, skipped = {}, {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(EXT):
                continue
            folder = os.path.dirname(info.filename).lower()
            numbers = re.findall(r'(?<!\d)(10|20|50|100)(?!\d)', folder)
            series = 'new' if 'new' in folder else 'old' if 'old' in folder else None
            if not numbers or not series or 'unspoilt' in folder:
                skipped[folder] = skipped.get(folder, 0) + 1
                continue
            d = numbers[-1]
            by_label.setdefault(d if d == '20' else f'{d}_{series}', []).append(info)
        if skipped:
            print('  folders with no usable label, skipped:', skipped)
        assert by_label, f'{source}: no folder name gave a denomination and series; look at the zip layout'
        rng = random.Random(11)
        for label, infos in sorted(by_label.items()):
            rng.shuffle(infos)
            os.makedirs(f'{NOTES}/{source}/{label}', exist_ok=True)
            n = 0
            for k, info in enumerate(infos[:SPOILT_CAP]):
                try:
                    with Image.open(io.BytesIO(zf.read(info))) as im:
                        save_rgb(im, f'{NOTES}/{source}/{label}/{k:05d}.jpg')
                    n += 1
                except Exception as e:
                    print('  bad image skipped:', info.filename, e)
            print(f'  {label:>8}: {n} of {len(infos)}')
    os.remove(path)
    store_cache(source)


def public_note_datasets():
    if OFFLINE:
        print('CURRENCY_OFFLINE set: no Mendeley downloads')
        return
    for dataset, version in FOLDER_DATASETS.items():
        fetch_folder_dataset(dataset, version)
    fetch_spoilt()


def ieee_notes():
    cands = [os.path.dirname(c) for c in glob.glob(f'{RAW}/**/', recursive=True)
             if norm(os.path.basename(os.path.dirname(c))) == '10new']
    cands = [os.path.dirname(c) for c in cands if 'thai' not in norm(os.path.basename(os.path.dirname(c)))]
    indian = os.environ.get('CURRENCY_INDIAN') or (cands[0] if cands else None)
    assert indian, 'Indian notes folder (containing "10 New") not found; set CURRENCY_INDIAN'
    print('INDIAN =', indian)
    jobs = []
    for sub in sorted(os.listdir(indian)):
        label = MERGE.get(norm(sub))
        if label is None:
            print(f'  skipping folder "{sub}"'); continue
        d = f'{NOTES}/ieee/{label}'; os.makedirs(d, exist_ok=True)
        found = images_in(os.path.join(indian, sub))
        print(f'  {sub!r:>12} -> {label:>8}: {len(found)} images')
        jobs += [(f, f'{d}/{norm(sub)}_{i:05d}.jpg') for i, f in enumerate(found)]
    def prep(job):
        try:
            save_rgb(Image.open(job[0]), job[1]); return True
        except Exception as e:
            print('  bad image skipped:', job[0], e); return False
    with ThreadPoolExecutor(8) as ex:
        ok = sum(ex.map(prep, jobs))
    print(f'IEEE: {ok}/{len(jobs)} images')


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


def roboflow_images(rf):
    for img in glob.glob(f'{rf}/**/images/*', recursive=True):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            split = os.path.basename(os.path.dirname(os.path.dirname(img)))
            yield img, split, os.path.splitext(img.replace('/images/', '/labels/'))[0] + '.txt'


def roboflow_notes(rf, none_id, names):
    """One square crop per photo, around its largest note. Photos mixing denominations are skipped.
    The series is not labelled, so these use the coarse label ('10', not '10_new')."""
    counts = {}
    for img, split, lbl in roboflow_images(rf):
        ids, boxes = read_label(lbl, none_id)
        present = ids - {none_id}
        if len(present) != 1 or not boxes:
            continue
        label = names[next(iter(present))]
        if label not in DENOMINATIONS:
            continue
        im = Image.open(img).convert('RGB')
        W, H = im.size
        x0, y0, x1, y1 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        bw, bh = (x1 - x0) * W, (y1 - y0) * H
        if min(bw, bh) < 64 or max(bw, bh) / min(bw, bh) > 3.2:       # tiny, or a sliver cut by a mosaic seam
            continue
        side = int(min(W, H, max(bw, bh) * 1.15))
        left = int(min(max((x0 + x1) / 2 * W - side / 2, 0), W - side))
        top = int(min(max((y0 + y1) / 2 * H - side / 2, 0), H - side))
        d = f'{NOTES}/roboflow/{label}'; os.makedirs(d, exist_ok=True)
        save_rgb(im.crop((left, top, left + side, top + side)), f'{d}/{split}__{os.path.basename(img)}')
        counts[label] = counts.get(label, 0) + 1
    print('Roboflow note crops:', dict(sorted(counts.items())))


def roboflow_backgrounds(rf, none_id):
    kept = empty = 0
    crops = []
    for img, split, lbl in roboflow_images(rf):
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
    crops.sort()
    random.shuffle(crops)
    for img, box, dst in crops[:MAX_CROPS]:
        Image.open(img).convert('RGB').crop(box).save(dst, 'JPEG', quality=95)
    print(f'Roboflow: {kept} "None" images ({empty} with empty/missing labels), '
          f'{min(len(crops), MAX_CROPS)} note-free crops (of {len(crops)} possible)')


def egohands_backgrounds():
    ego = f'{WORK}/egohands'
    if not glob.glob(f'{ego}/**/*.jpg', recursive=True):
        zp = f'{RAW_ZIPS}/egohands_data.zip'              # manual upload location
        if not (os.path.exists(zp) and zipfile.is_zipfile(zp)):
            zp = f'{WORK}/egohands_data.zip'
            for url in EGOHANDS_URLS:
                try:
                    print('trying', url)
                    fetch(url, zp)
                    if zipfile.is_zipfile(zp):
                        break
                    print('  not a zip (got', os.path.getsize(zp), 'bytes)')
                except Exception as e:
                    print('  failed:', e)
        if not (os.path.exists(zp) and zipfile.is_zipfile(zp)):
            print('EgoHands unavailable (optional). To add it: download "Labeled Data" from '
                  f'https://vision.soic.indiana.edu/?p=1500 and put it at {RAW_ZIPS}/egohands_data.zip.')
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


def tar_backgrounds(name, url, class_of, per_class):
    """Sample per_class images per class from a public .tar.gz. A failed download is not fatal."""
    try:
        path, root = f'{WORK}/{name}.tgz', f'{WORK}/{name}'
        if not os.path.isdir(root):
            print(f'{name}: downloading')
            fetch(url, path, timeout=900)
            with tarfile.open(path) as t:
                t.extractall(root, filter='data')
            os.remove(path)
        by_class = {}
        for f in images_in(root):
            c = class_of(f)
            if c:
                by_class.setdefault(c, []).append(f)
        n = 0
        for c, files in sorted(by_class.items()):
            for k, f in enumerate(random.sample(files, min(per_class, len(files)))):
                save_rgb(Image.open(f), f'{BG}/{name}-{c}-{k}.jpg'); n += 1
        print(f'{name}: {len(by_class)} classes -> {n} images')
    except Exception as e:
        print(f'!! {name} unavailable, continuing without it:', e)


def contact_sheets(per_sheet=48, cols=8, cell=160):
    """Only Roboflow-derived backgrounds can contain notes, so only those are reviewed."""
    shutil.rmtree(SHEETS, ignore_errors=True); os.makedirs(SHEETS)
    files = sorted(glob.glob(f'{BG}/rf*'))
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
    print(f'{len(files)} Roboflow background candidates -> {SHEETS}/sheet_*.png (index.csv maps numbers to files)')


def prepare():
    random.seed(0)
    os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    unzip_all()
    shutil.rmtree(BG, ignore_errors=True); os.makedirs(BG)
    os.makedirs(NOTES, exist_ok=True)
    for rebuilt in ('ieee', 'roboflow'):          # finished or partial Mendeley downloads are kept
        shutil.rmtree(f'{NOTES}/{rebuilt}', ignore_errors=True)
    ieee_notes()
    rf, none_id, names = find_roboflow()
    print('Roboflow classes:', names, '| None id =', none_id)
    roboflow_notes(rf, none_id, names)
    public_note_datasets()
    roboflow_backgrounds(rf, none_id)
    if USE_EGOHANDS and not OFFLINE:
        egohands_backgrounds()
    if not OFFLINE:
        tar_backgrounds('dtd', DTD_URL, lambda f: os.path.basename(os.path.dirname(f)), DTD_PER_CLASS)
        tar_backgrounds('imagenette', IMAGENETTE_URL,
                        lambda f: os.path.basename(os.path.dirname(f)) if '/train/' in f else None,
                        IMAGENETTE_PER_CLASS)
    # re-apply earlier rejections (kept on Drive so a runtime reset doesn't undo the review)
    if os.path.exists(REJECTED):
        gone = 0
        for name in open(REJECTED).read().split():
            if os.path.exists(f'{BG}/{name}'):
                os.remove(f'{BG}/{name}'); gone += 1
        print(f'Re-applied {gone} earlier rejections from {REJECTED}')
    contact_sheets()
    print('Note images per source:')
    for source in sorted(os.listdir(NOTES)):
        print(f'  {source:<24}', {l: len(images_in(f'{NOTES}/{source}/{l}')) for l in sorted(os.listdir(f'{NOTES}/{source}'))})
    print('Background images:', len(glob.glob(f'{BG}/*')))


def reject(indices):
    rows = dict((int(i), n) for i, n in csv.reader(open(f'{SHEETS}/index.csv')))
    os.makedirs(OUT, exist_ok=True)
    with open(REJECTED, 'a') as log:
        for i in map(int, indices):
            p = f'{BG}/{rows[i]}'
            if os.path.exists(p):
                os.remove(p); log.write(rows[i] + '\n'); print('rejected', i, rows[i])
    print(len(glob.glob(f'{BG}/*')), 'background images left')


# ---------------------------------------------------------------- splits
def dhash(path):
    """64-bit perceptual hash: survives resizing and recompression, unlike md5."""
    with Image.open(path) as im:
        a = np.asarray(im.convert('L').resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), 'big')


def build_manifest():
    """Rows of {path, label, source, split}. Splits are train, val, test and external."""
    candidates = []
    for source in sorted(os.listdir(NOTES)):
        for label in sorted(os.listdir(f'{NOTES}/{source}')):
            candidates += [(f, label, source) for f in images_in(f'{NOTES}/{source}/{label}')]
    candidates += [(f, 'background', 'background') for f in images_in(BG)]

    def identify(c):
        try:
            return hashlib.md5(open(c[0], 'rb').read()).hexdigest(), dhash(c[0])
        except Exception as e:
            print('  bad image skipped:', c[0], e); return None
    with ThreadPoolExecutor(8) as ex:
        ids = list(ex.map(identify, candidates))

    # Exact copies go. Equal perceptual hashes share a group, so look-alikes land in the same split.
    seen, group_of_hash, rows, exact = set(), {}, [], 0
    for (path, label, source), i in zip(candidates, ids):
        if i is None:
            continue
        if i[0] in seen:
            exact += 1; continue
        seen.add(i[0])
        group = group_of_hash.setdefault(i[1], f'{source}/{group_key(path)}')
        rows.append({'path': path, 'label': label, 'source': source, 'group': group, 'hash': i[1]})
    print(f'{len(rows)} unique images ({exact} exact duplicates dropped)')

    # A training image that nearly matches an external test image would make the test meaningless.
    external = np.array([r['hash'] for r in rows if r['source'] == EXTERNAL_TEST], dtype=np.uint64)
    popcount = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    kept, near = [], 0
    for r in rows:
        if r['source'] != EXTERNAL_TEST and len(external):
            xor = np.bitwise_xor(external, np.uint64(r['hash']))
            if popcount[np.frombuffer(xor.tobytes(), dtype=np.uint8)].reshape(-1, 8).sum(1).min() <= 4:
                near += 1; continue
        kept.append(r)
    print(f'{near} images dropped for nearly matching the external test set')

    for r in kept:
        if r['source'] == EXTERNAL_TEST:
            r['split'] = 'external'
        else:                                 # by group, so every copy of a photo gets the same split
            bucket = int(hashlib.md5(r['group'].encode()).hexdigest(), 16) % 100
            r['split'] = 'test' if bucket < 12 else 'val' if bucket < 24 else 'train'
    summary = {}
    for r in kept:
        summary.setdefault(r['split'], {}).setdefault(coarse_of(r['label']), 0)
        summary[r['split']][coarse_of(r['label'])] += 1
    for split, counts in sorted(summary.items()):
        print(f'{split:>9}: {sum(counts.values()):5d}', dict(sorted(counts.items())))
    with open(f'{WORK}/manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['path', 'label', 'source', 'split'], extrasaction='ignore')
        w.writeheader(); w.writerows(kept)
    return kept, summary


# ---------------------------------------------------------------- train
def threshold_table(y_true, probs, bg):
    """What the app would do at each confidence threshold, per single image."""
    pred, conf, notes, table = probs.argmax(1), probs.max(1), y_true != bg, {}
    for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        speaks = (conf >= t) & (pred != bg)
        table[t] = {'speaks_on_notes': float((speaks & notes).sum() / max(notes.sum(), 1)),
                    'wrong_when_speaking': float((speaks & (pred != y_true)).sum() / max(speaks.sum(), 1)),
                    'spoken': int(speaks.sum())}
    return table


def train():
    import tensorflow as tf
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    os.makedirs(OUT, exist_ok=True)
    print('TF', tf.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))
    rows, summary = build_manifest()
    by_split = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test', 'external')}
    assert by_split['external'], f'No {EXTERNAL_TEST} images: did prepare finish?'
    for label in COARSE:
        assert summary['train'].get(label), f'No training images for class {label}'
        if summary['train'][label] < 150:
            print(f'  <-- LOW: only {summary["train"][label]} training images for {label}')

    IMG, BATCH, SEED = 224, 32, 42
    AUTOTUNE = tf.data.AUTOTUNE
    EPOCHS1 = int(os.environ.get('CURRENCY_EPOCHS1', 8))
    EPOCHS2 = int(os.environ.get('CURRENCY_EPOCHS2', 60))
    FINE_TUNE_LR = float(os.environ.get('CURRENCY_LR', 2e-4))
    SMOOTHING = 0.05
    WEIGHTS = os.environ.get('CURRENCY_WEIGHTS', 'imagenet') or None   # '' only for offline tests
    tf.keras.utils.set_random_seed(SEED)

    # fine -> coarse, shared by the loss, the metric and the export
    merge_matrix = np.array([[float(coarse_of(f) == c) for c in COARSE] for f in FINE], dtype=np.float32)
    bg = COARSE.index('background')
    M = tf.constant(merge_matrix)

    def decode(path):
        x = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        x.set_shape([None, None, 3])
        return tf.cast(x, tf.float32)

    def to_model_size(x):
        return tf.image.resize(x, [IMG, IMG], method='bilinear')

    def center_square(x):                          # the same crop the app makes
        s = tf.shape(x); side = tf.minimum(s[0], s[1])
        return tf.image.crop_to_bounding_box(x, (s[0] - side) // 2, (s[1] - side) // 2, side, side)

    def random_square(x):                          # a wide photo shouldn't always lose both ends
        s = tf.shape(x); side = tf.minimum(s[0], s[1])
        return tf.image.random_crop(x, [side, side, 3])

    counts = np.array([sum(coarse_of(r['label']) == c for r in by_split['train']) for c in COARSE], dtype=np.float32)
    class_weight = counts.sum() / (len(COARSE) * counts)

    def training_dataset():
        paths = [r['path'] for r in by_split['train']]
        masks = np.array([mask_of(r['label']) for r in by_split['train']], dtype=np.float32)
        weights = np.array([class_weight[COARSE.index(coarse_of(r['label']))] for r in by_split['train']],
                           dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, masks, weights)).shuffle(len(paths), seed=SEED)
        return ds.map(lambda p, m, w: (to_model_size(random_square(decode(p))), m, w), num_parallel_calls=AUTOTUNE) \
                 .batch(BATCH, drop_remainder=True)

    def validation_dataset():
        paths = [r['path'] for r in by_split['val']]
        masks = np.array([mask_of(r['label']) for r in by_split['val']], dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, masks))
        return ds.map(lambda p, m: (to_model_size(center_square(decode(p))), m), num_parallel_calls=AUTOTUNE) \
                 .batch(BATCH).prefetch(AUTOTUNE)

    def eval_images(split_rows):
        ds = tf.data.Dataset.from_tensor_slices([r['path'] for r in split_rows])
        return ds.map(lambda p: to_model_size(center_square(decode(p))), num_parallel_calls=AUTOTUNE).batch(1)

    # Augmentation: training pipeline only, never part of the exported model. No mirroring, no hue
    # shift: mirrored digits never occur, and colour is what separates denominations.
    geometry = [tf.keras.layers.RandomRotation(0.04),
                tf.keras.layers.RandomZoom((-0.2, 0.35)),         # positive = further away
                tf.keras.layers.RandomTranslation(0.1, 0.1)]
    if hasattr(tf.keras.layers, 'RandomPerspective'):             # not in every Keras 3 release
        geometry.append(tf.keras.layers.RandomPerspective(factor=0.5, scale=0.15))
    augment = tf.keras.Sequential(geometry + [
        tf.keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.25),
    ], name='augment')

    def rotate_per_image(x):                       # independent 0/90/180/270 turn per image
        k = tf.random.uniform([tf.shape(x)[0]], 0, 4, dtype=tf.int32)
        rots = tf.stack([tf.image.rot90(x, i) for i in range(4)], axis=1)
        return tf.gather(rots, k, batch_dims=1)

    def blur_some(x):                              # hand shake and missed focus
        n = tf.shape(x)[0]
        light, heavy = tf.nn.avg_pool2d(x, 3, 1, 'SAME'), tf.nn.avg_pool2d(x, 5, 1, 'SAME')
        blurred = tf.where(tf.random.uniform([n, 1, 1, 1]) < 0.5, light, heavy)
        return tf.where(tf.random.uniform([n, 1, 1, 1]) < 0.3, blurred, x)

    def white_balance(x):                          # phone cameras disagree on colour temperature
        return x * tf.random.uniform([tf.shape(x)[0], 1, 1, 3], 0.92, 1.08)

    def occlude_some(x):                           # fingers holding the note
        n = tf.shape(x)[0]
        size = tf.cast(IMG, tf.float32)
        w = tf.random.uniform([n, 1, 1], 0.12, 0.40) * size
        h = tf.random.uniform([n, 1, 1], 0.12, 0.40) * size
        left = tf.random.uniform([n, 1, 1]) * (size - w)
        top = tf.random.uniform([n, 1, 1]) * (size - h)
        grid = tf.range(IMG, dtype=tf.float32)
        xs = tf.reshape(grid, [1, 1, IMG]); ys = tf.reshape(grid, [1, IMG, 1])
        inside = (xs >= left) & (xs < left + w) & (ys >= top) & (ys < top + h)
        inside &= tf.random.uniform([n, 1, 1]) < 0.4
        colour = tf.random.uniform([n, 1, 1, 3], 40.0, 230.0)
        return tf.where(inside[..., tf.newaxis], colour, x)

    def train_aug(x, mask, weight):
        x = augment(x, training=True)
        x = rotate_per_image(x)
        x = blur_some(x)
        x = white_balance(x)
        x = occlude_some(x)
        x = x + tf.random.normal(tf.shape(x), stddev=8.0)   # Keras GaussianNoise rejects stddev > 1
        return tf.clip_by_value(x, 0.0, 255.0), mask, weight

    train_ds = training_dataset().map(train_aug, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = validation_dataset()
    steps = max(len(by_split['train']) // BATCH, 1)

    x, mask, _ = next(iter(train_ds))
    plt.figure(figsize=(9, 9))
    for i in range(min(16, len(x))):
        allowed = [f for f, m in zip(FINE, mask[i].numpy()) if m > 0]
        plt.subplot(4, 4, i + 1); plt.imshow(x[i].numpy().astype('uint8'))
        plt.title(allowed[0] if len(allowed) == 1 else coarse_of(allowed[0])); plt.axis('off')
    plt.savefig(f'{OUT}/augmented_samples.png', dpi=100, bbox_inches='tight'); plt.close()

    def partial_label_loss(mask, probs):
        """Cross-entropy over the *allowed* fine classes: a photo labelled just '100' may be
        either series, so the two 100s share the credit. One-hot masks give ordinary cross-entropy."""
        probs = tf.clip_by_value(probs, 1e-7, 1.0)
        allowed = tf.reduce_sum(tf.cast(mask, probs.dtype) * probs, axis=1)
        uniform = -tf.reduce_mean(tf.math.log(probs), axis=1)
        return -(1 - SMOOTHING) * tf.math.log(allowed) + SMOOTHING * uniform

    def coarse_accuracy(mask, probs):
        truth = tf.argmax(tf.matmul(tf.cast(mask, tf.float32), M), axis=1)
        guess = tf.argmax(tf.matmul(tf.cast(probs, tf.float32), M), axis=1)
        return tf.cast(tf.equal(truth, guess), tf.float32)

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG, IMG, 3), include_top=False, weights=WEIGHTS,
        pooling='avg', include_preprocessing=True)            # expects raw 0-255 pixels
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG, IMG, 3))
    h = base(inputs, training=False)
    h = tf.keras.layers.Dropout(0.3)(h)
    fine_output = tf.keras.layers.Dense(len(FINE), activation='softmax', name='fine')(h)
    model = tf.keras.Model(inputs, fine_output)

    # Phase 1: head only
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=partial_label_loss, metrics=[coarse_accuracy])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS1)

    # Phase 2: the whole backbone, BatchNorm frozen, learning rate decaying to zero
    base.trainable = True
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    schedule = tf.keras.optimizers.schedules.CosineDecay(FINE_TUNE_LR, EPOCHS2 * steps)
    model.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss=partial_label_loss, metrics=[coarse_accuracy])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS2,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_coarse_accuracy', mode='max', patience=12),
                              tf.keras.callbacks.ModelCheckpoint(f'{OUT}/best.weights.h5', save_weights_only=True,
                                                                 monitor='val_coarse_accuracy', mode='max',
                                                                 save_best_only=True)])
    model.load_weights(f'{OUT}/best.weights.h5')

    # What ships: the same network with a fixed layer that adds series probabilities together.
    merge = tf.keras.layers.Dense(len(COARSE), use_bias=False, trainable=False, name='to_denomination')
    export_model = tf.keras.Model(inputs, merge(model.output))
    merge.set_weights([merge_matrix])
    export_model.save(f'{OUT}/best.keras')
    with open(f'{OUT}/labels.txt', 'w') as f:
        f.write('\n'.join(COARSE))

    # Per-epoch history is kept on disk: the training log does not survive a runtime reset.
    history = {k: [float(v) for v in h1.history[k] + h2.history.get(k, [])] for k in h1.history}
    history['phase1_epochs'] = len(h1.history['loss'])
    json.dump(history, open(f'{OUT}/history.json', 'w'), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, key, name in zip(axes, ('coarse_accuracy', 'loss'), ('accuracy', 'loss')):
        ax.plot(history[key], label='train'); ax.plot(history['val_' + key], label='validation')
        ax.axvline(history['phase1_epochs'] - 0.5, ls='--', c='gray', label='start of fine-tuning')
        ax.set_xlabel('epoch'); ax.set_ylabel(name); ax.legend()
    fig.savefig(f'{OUT}/curves.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # LiteRT export. Float32: INT8 lost 15 accuracy points on this architecture, and the float
    # graph is the one the default XNNPACK CPU delegate accepts. 3.8 MB is nothing in an APK.
    tflite_path = f'{OUT}/currency_model.tflite'
    previous_path = f'{OUT}/previous_currency_model.tflite'
    if os.path.exists(tflite_path) and not os.path.exists(previous_path):
        shutil.copy(tflite_path, previous_path)                 # the baseline to beat; never overwritten
    export_model.export(f'{WORK}/saved_model')
    tfl = tf.lite.TFLiteConverter.from_saved_model(f'{WORK}/saved_model').convert()
    open(tflite_path, 'wb').write(tfl)

    resolver = tf.lite.experimental.OpResolverType
    def predict_tflite(split_rows, model_path=tflite_path, no_delegates=False):
        kw = {'experimental_op_resolver_type': resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES} if no_delegates else {}
        interp = tf.lite.Interpreter(model_path=model_path, **kw)
        interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        assert list(inp['shape']) == [1, IMG, IMG, 3] and inp['dtype'] == np.float32, inp
        assert list(out['shape']) == [1, len(COARSE)] and out['dtype'] == np.float32, out
        probs = []
        for x in eval_images(split_rows):
            interp.set_tensor(inp['index'], x.numpy()); interp.invoke()
            probs.append(interp.get_tensor(out['index'])[0].copy())
        return np.array(probs)

    truth = lambda split_rows: np.array([COARSE.index(coarse_of(r['label'])) for r in split_rows])

    def report(name, split_rows, probs):
        y_true, y_pred = truth(split_rows), probs.argmax(1)
        accuracy = float((y_true == y_pred).mean())
        present = sorted(set(y_true.tolist()))
        names = [COARSE[i] for i in present]
        print(f'\n=== {name}: accuracy {accuracy:.4f} on {len(y_true)} images')
        print(classification_report(y_true, y_pred, labels=present, target_names=names, digits=3, zero_division=0))
        fig, ax = plt.subplots(figsize=(7, 7))
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=range(len(COARSE))),
                               display_labels=COARSE).plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(name)
        fig.savefig(f'{OUT}/cm_{name}.png', dpi=150, bbox_inches='tight'); plt.close(fig)
        recall = {COARSE[i]: round(float((y_pred[y_true == i] == i).mean()), 4) for i in present}
        return accuracy, recall

    probs = {s: predict_tflite(by_split[s]) for s in ('val', 'test', 'external')}
    assert abs(probs['test'].sum(1) - 1).max() < 1e-3, 'exported probabilities do not sum to 1'
    test_acc, test_recall = report('test', by_split['test'], probs['test'])
    external_acc, external_recall = report('external', by_split['external'], probs['external'])

    # The threshold is chosen on validation data, then judged on data it never saw.
    val_table = threshold_table(truth(by_split['val']), probs['val'], bg)
    usable = [t for t, v in val_table.items() if v['spoken'] and v['wrong_when_speaking'] <= 0.01]
    threshold = min(usable) if usable else 0.85
    external_table = threshold_table(truth(by_split['external']), probs['external'], bg)
    test_truth = truth(by_split['test'])
    is_bg, conf, pred = test_truth == bg, probs['test'].max(1), probs['test'].argmax(1)
    bg_as_note = float(((pred != bg) & (conf >= threshold) & is_bg).sum() / max(is_bg.sum(), 1))

    baseline = None
    if os.path.exists(previous_path):
        try:
            previous = predict_tflite(by_split['external'], previous_path, no_delegates=True)
            baseline = round(float((truth(by_split['external']) == previous.argmax(1)).mean()), 4)
        except Exception as e:
            print('Previous model could not be evaluated:', e)

    results = {'class_names': COARSE, 'fine_classes': FINE,
               'train_counts': summary['train'],
               'test_accuracy': round(test_acc, 4), 'test_recall': test_recall,
               'external_accuracy': round(external_acc, 4), 'external_recall': external_recall,
               'external_accuracy_previous_model': baseline,
               'recommended_threshold': threshold,
               'background_spoken_as_note': round(bg_as_note, 4),
               'thresholds_validation': val_table, 'thresholds_external': external_table,
               'tflite_mb': round(len(tfl) / 1e6, 2)}
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

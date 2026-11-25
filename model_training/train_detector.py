"""
Indian Currency Detector — multi-note detection, dataset prep, training and LiteRT export.

YOLO11n (Ultralytics) that finds every banknote in a frame and labels its denomination, so the app
can say "two 100 rupee notes, one 500 rupee note, total 700 rupees". It is trained next to the
existing classifier and never touches the classifier's files: the classifier keeps working.

Classes (6, numeric order, this is the detector_labels.txt order): 10, 20, 50, 100, 200, 500.
There is no background class: "no box" means no note. Old and new series share one class, because
the Roboflow and augmented Mendeley sources don't say which series a note is.

Stages (run in order on a Kaggle GPU session; several can be chained in one call):
  python train_detector.py prepare         # unzip, download public datasets, background pool, contact sheets
  python train_detector.py reject bg 3 17  # drop background images that show any part of a note
  python train_detector.py autobox         # find the one note in every single-note photo (OWLv2), review sheets
  python train_detector.py reject box 5 9  # drop photos whose box is wrong
  python train_detector.py synth           # splits, multi-note composite scenes, YOLO dataset
  python train_detector.py train           # YOLO11n training (resumes after a session reset)
  python train_detector.py export          # float32 .tflite + labels, checked on the exported file
  python train_detector.py evaluate        # counts, totals and false speech, measured on the .tflite

Paths come from environment variables (defaults are for Kaggle):
  CURRENCY_RAW   input zips (IEEE + Roboflow YOLOv8 export)   (default /kaggle/working/currency_raw)
  DETECTOR_OUT   persistent outputs of this script            (default /kaggle/working/currency_detector)
  DETECTOR_WORK  scratch, rebuilt by the stages               (default /kaggle/temp/detector)
  CURRENCY_OUT   the classifier's outputs, read only          (default /kaggle/working/currency)
                 (its Mendeley download cache and background rejections are reused, nothing is written)
Tuning: DETECTOR_IMGSZ (416), DETECTOR_EPOCHS (150), DETECTOR_BATCH (32), DETECTOR_DEVICE (auto),
DETECTOR_FRACTION (1.0, share of the train split to use), DETECTOR_SYNTH_TRAIN/VAL/TEST (8000/800/800),
DETECTOR_EVAL_LIMIT (0 = no limit), DETECTOR_OFFLINE (skip downloads), DETECTOR_FRESH (restart training).
"""
import os, sys, glob, json, math, random, re, shutil, tarfile, zipfile, hashlib, csv, io, time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageFilter

RAW_ZIPS = os.environ.get('CURRENCY_RAW', '/kaggle/working/currency_raw')
OUT      = os.environ.get('DETECTOR_OUT', '/kaggle/working/currency_detector')
WORK     = os.environ.get('DETECTOR_WORK', '/kaggle/temp/detector')
CLS_OUT  = os.environ.get('CURRENCY_OUT', '/kaggle/working/currency')   # classifier outputs: read only
OFFLINE  = bool(os.environ.get('DETECTOR_OFFLINE'))
RAW      = f'{WORK}/raw'              # unzipped inputs
BG       = f'{WORK}/background_pool'  # every background candidate, before the split
NOTES    = f'{WORK}/notes'            # <source>/<label>/<image>.jpg, single-note photos
SCENES   = f'{WORK}/roboflow_scenes'  # real Roboflow photos with their own boxes
SHEETS   = f'{WORK}/contact_sheets'   # bg/ and box/ review grids
YOLO     = f'{WORK}/yolo'             # images/{train,val,test} + labels/{train,val,test}
EVAL     = f'{WORK}/eval'             # extra evaluation sets that never enter training
CACHE    = f'{OUT}/cache'             # resized downloads, so a session reset needs no re-download
BOXES    = f'{OUT}/boxes.json'        # autobox results, survives session resets
REJECTED_BG  = f'{OUT}/rejected_backgrounds.txt'
REJECTED_BOX = f'{OUT}/rejected_boxes.txt'
RUNS     = f'{OUT}/runs'
TFLITE   = f'{OUT}/currency_detector.tflite'
LABELS   = f'{OUT}/detector_labels.txt'

CLASSES = ['10', '20', '50', '100', '200', '500']      # class id = index; never reorder
DENOMINATIONS = set(CLASSES)
EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
UA = {'User-Agent': 'Mozilla/5.0'}

IMGSZ  = int(os.environ.get('DETECTOR_IMGSZ', 416))
EPOCHS = int(os.environ.get('DETECTOR_EPOCHS', 150))
BATCH  = int(os.environ.get('DETECTOR_BATCH', 32))
FRACTION = float(os.environ.get('DETECTOR_FRACTION', 1.0))
EVAL_LIMIT = int(os.environ.get('DETECTOR_EVAL_LIMIT', 0))
SYNTH_COUNTS = {'train': int(os.environ.get('DETECTOR_SYNTH_TRAIN', 8000)),
                'val': int(os.environ.get('DETECTOR_SYNTH_VAL', 800)),
                'test': int(os.environ.get('DETECTOR_SYNTH_TEST', 800))}
EXTERNAL_SCENES = 800                  # composites built only from external notes, for the count test
REAL_NOTES_PER_CLASS = {'train': 600, 'val': 100, 'test': 100}   # single-note photos used as they are
NEGATIVES = {'train': 600, 'val': 100, 'test': 100}              # empty backgrounds, no boxes

# Sources that never enter training. Its accuracy is the honest "new photographer, new notes" number.
EXTERNAL_TEST = 'mendeley-8ckhkssyn3'
FOLDER_DATASETS = {'8ckhkssyn3': 1, '48ympv8jjf': 2}     # Mendeley id -> version, all CC BY 4.0
SPOILT = ('jh6979fg2t', 4)
AUGMENTED_DATASET = '48ympv8jjf'
AUGMENTED_CAP, AUGMENTED_PER_PHOTO = 500, 3
SPOILT_CAP = 250

# Background sources (all public)
USE_EGOHANDS, EGO_PER_VIDEO = True, 3
DTD_PER_CLASS = 6
IMAGENETTE_PER_CLASS = 30
CROP_MIN, CROP_MARGIN, MAX_CROPS = 0.30, 0.03, 300
EGOHANDS_URLS = ['https://vision.luddy.iu.edu/egohands_files/egohands_data.zip',
                 'http://vision.soic.indiana.edu/egohands_files/egohands_data.zip']
DTD_URL = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'

# Autobox (OWLv2 open-vocabulary detector, only used to label the single-note photos)
OWL_MODEL = 'google/owlv2-base-patch16-ensemble'
OWL_QUERIES = ['a photo of a banknote', 'a currency note', 'paper money']
OWL_PAD = 0.12          # grey border added before detection, so a note that fills the photo still has edges
BOX_MIN_SCORE = 0.20    # below this the photo is dropped
BOX_MIN_AREA = 0.06     # box smaller than this share of the photo is suspicious
BOX_RIVAL = 0.70        # a separate box scoring this share of the best one makes the photo ambiguous

# Composite scenes
SCENE_SIZES = [(640, 480), (480, 640), (640, 640)]
NOTES_PER_SCENE = {1: 0.15, 2: 0.25, 3: 0.22, 4: 0.18, 5: 0.12, 6: 0.08}
SAME_DENOMINATION = 0.40   # share of multi-note scenes where every note has one denomination
MIN_VISIBLE = 0.45         # a note may be covered by later notes until this share of it is left

# Inference, shared with the app: agnostic NMS, so one note never counts as two denominations
NMS_IOU = 0.55
THRESHOLDS = [round(t, 2) for t in np.arange(0.25, 0.81, 0.05)]
MAX_FALSE_SPEECH = 0.01

IEEE_MERGE = {'10new': '10_new', '10old': '10_old', '20': '20', '20new': '20', '20old': '20',
              '50new': '50_new', '50old': '50_old', '100new': '100_new', '100old': '100_old',
              '200': '200', '200new': '200', '500': '500', '500new': '500'}

norm = lambda s: s.lower().replace(' ', '').replace('_', '').replace('-', '')


def coarse_of(label):
    """'10_new' -> '10'. Also accepts labels that are already coarse."""
    return label.split('_')[0]


def images_in(folder):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(folder)
                  for f in fs if f.lower().endswith(EXT))


def save_rgb(im, dst, max_side=512):
    im = ImageOps.exif_transpose(im).convert('RGB')
    im.thumbnail((max_side, max_side))
    im.save(dst, 'JPEG', quality=95)


def group_key(path):
    """Images sharing a key stay on the same side of the split.
    Roboflow augmented copies are '<photo>_jpg.rf.<hash>.jpg'; our own derived files use '<source>.rf.<n>.jpg'."""
    b = os.path.basename(path).split('__', 1)[-1]
    return b.split('.rf.')[0] if '.rf.' in b else path


def rel(path):
    return os.path.relpath(path, WORK)


def write_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ================================================================ prepare
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
            if os.path.basename(z) != 'egohands_data.zip']
    assert zips, f'No zips in {RAW_ZIPS}'
    for z in zips:
        name = os.path.splitext(os.path.basename(z))[0]
        print('unzipping', os.path.basename(z))
        with zipfile.ZipFile(z) as zf:
            zf.extractall(f'{RAW}/{name}')
    for z in glob.glob(f'{RAW}/**/*.zip', recursive=True):
        print('unzipping nested', z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(os.path.splitext(z)[0])


def restore_cache(source):
    """Unpack a finished download, from this script's cache or the classifier's. True if there was one."""
    for cache in (CACHE, f'{CLS_OUT}/cache'):
        z = f'{cache}/{source}.zip'
        if os.path.exists(z) and zipfile.is_zipfile(z):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(f'{NOTES}/{source}')
            print(f'{source}: restored {len(images_in(f"{NOTES}/{source}"))} images from {cache}')
            return True
    return False


def store_cache(source):
    print(f'{source}: saving to cache')
    os.makedirs(CACHE, exist_ok=True)
    tmp = shutil.make_archive(f'{WORK}/{source}', 'zip', f'{NOTES}/{source}')
    shutil.copy(tmp, f'{CACHE}/{source}.zip.tmp')
    os.replace(f'{CACHE}/{source}.zip.tmp', f'{CACHE}/{source}.zip')
    os.remove(tmp)


def mendeley_api(path):
    return json.loads(fetch_bytes(f'https://data.mendeley.com/public-api/datasets/{path}'))


def mendeley_files(dataset, version):
    folders = mendeley_api(f'{dataset}/folders/{version}')
    def list_folder(folder):
        files = mendeley_api(f'{dataset}/files?folder_id={folder["id"]}&version={version}')
        return [(folder['name'], f['filename'], f['content_details']['download_url']) for f in files]
    with ThreadPoolExecutor(8) as ex:
        return [item for chunk in ex.map(list_folder, folders) for item in chunk]


def denomination_of(name):
    m = re.search(r'\d+', str(name))
    return m.group() if m and m.group() in DENOMINATIONS else None


def select_augmented(items):
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
        if os.path.exists(dst):
            return True
        try:
            with Image.open(io.BytesIO(fetch_bytes(url))) as im:
                im.draft('RGB', (1024, 1024))
                save_rgb(im, dst + '.part')
            os.replace(dst + '.part', dst)
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
    assert ok >= 0.9 * len(items), f'{source}: too many downloads failed; rerun prepare'
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
        print('DETECTOR_OFFLINE set: no Mendeley downloads')
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
        label = IEEE_MERGE.get(norm(sub))
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
    yamls = glob.glob(f'{RAW}/**/data.yaml', recursive=True)
    assert len(yamls) == 1, f'Expected exactly one data.yaml (Roboflow YOLOv8 export), found: {yamls}'
    import yaml
    names = yaml.safe_load(open(yamls[0]))['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return os.path.dirname(yamls[0]), names


def roboflow_class_map(names):
    """Roboflow class id -> our class id, 'none' for the no-currency class, or None for anything else."""
    mapping = {}
    for i, n in enumerate(names):
        if norm(n) in ('none', 'background'):
            mapping[i] = 'none'
        else:
            d = denomination_of(n)
            mapping[i] = CLASSES.index(d) if d else None
    print('Roboflow classes:', {n: mapping[i] for i, n in enumerate(names)}, '(None = image skipped)')
    return mapping


def read_boxes(lbl):
    """[(roboflow class id, x0, y0, x1, y1)], normalised. Polygons become their bounding box."""
    out = []
    if not os.path.exists(lbl):
        return out
    for line in open(lbl):
        p = line.split()
        if len(p) < 5:
            continue
        cid, c = int(p[0]), list(map(float, p[1:]))
        if len(c) == 4:
            box = (c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2)
        else:
            box = (min(c[0::2]), min(c[1::2]), max(c[0::2]), max(c[1::2]))
        out.append((cid, *[min(max(v, 0.0), 1.0) for v in box]))
    return out


def roboflow_images(rf):
    for img in sorted(glob.glob(f'{rf}/**/images/*', recursive=True)):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            split = os.path.basename(os.path.dirname(os.path.dirname(img)))
            yield img, split, os.path.splitext(img.replace('/images/', '/labels/'))[0] + '.txt'


def roboflow_scenes(rf, mapping):
    """Real photos with boxes, kept whole. A photo with any box we can't map (₹2000, coins, ...) is skipped,
    because training on it would teach the model that such a note is background."""
    shutil.rmtree(SCENES, ignore_errors=True); os.makedirs(SCENES)
    scenes, skipped, counts = {}, 0, Counter()
    for img, split, lbl in roboflow_images(rf):
        boxes = read_boxes(lbl)
        notes = [b for b in boxes if mapping.get(b[0]) != 'none']
        if not notes:
            continue
        if any(mapping.get(b[0]) is None for b in notes):
            skipped += 1; continue
        notes = [(mapping[b[0]], *b[1:]) for b in notes if b[3] > b[1] and b[4] > b[2]]
        dst = f'{SCENES}/{split}__{os.path.basename(os.path.splitext(img)[0])}.jpg'
        save_rgb(Image.open(img), dst, max_side=640)
        scenes[rel(dst)] = notes
        counts.update(CLASSES[b[0]] for b in notes)
    write_json(scenes, f'{WORK}/roboflow_scenes.json')
    print(f'Roboflow scenes: {len(scenes)} photos, notes per class {dict(sorted(counts.items(), key=lambda kv: int(kv[0])))}, '
          f'{skipped} skipped for an unmapped class')


def roboflow_backgrounds(rf, mapping):
    kept = 0
    crops = []
    for img, split, lbl in roboflow_images(rf):
        boxes = read_boxes(lbl)
        notes = [b for b in boxes if mapping.get(b[0]) != 'none']
        if not notes:                       # whole image has no currency (or no label file)
            shutil.copy(img, f'{BG}/rfnone-{split}__{os.path.basename(img)}')
            kept += 1
            continue
        x0 = max(0, min(b[1] for b in notes) - CROP_MARGIN); y0 = max(0, min(b[2] for b in notes) - CROP_MARGIN)
        x1 = min(1, max(b[3] for b in notes) + CROP_MARGIN); y1 = min(1, max(b[4] for b in notes) + CROP_MARGIN)
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
    print(f'Roboflow backgrounds: {kept} no-note images, {min(len(crops), MAX_CROPS)} note-free crops')


def egohands_backgrounds():
    ego = f'{WORK}/egohands'
    if not glob.glob(f'{ego}/**/*.jpg', recursive=True):
        zp = f'{RAW_ZIPS}/egohands_data.zip'
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
            print('EgoHands unavailable (optional). Put "Labeled Data" from '
                  f'https://vision.soic.indiana.edu/?p=1500 at {RAW_ZIPS}/egohands_data.zip to add it.')
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


def contact_sheet(kind, entries, per_sheet=48, cols=8, cell=160):
    """entries: [(name, image path, box or None)]. Writes SHEETS/<kind>/sheet_XX.png and index.csv."""
    d = f'{SHEETS}/{kind}'
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    with open(f'{d}/index.csv', 'w', newline='') as f:
        csv.writer(f).writerows((i, e[0]) for i, e in enumerate(entries))
    for s in range(0, len(entries), per_sheet):
        chunk = entries[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * cell, rows * (cell + 18)), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, (_, path, box) in enumerate(chunk):
            im = Image.open(path).convert('RGB')
            if box is not None:
                W, H = im.size
                ImageDraw.Draw(im).rectangle([box[0] * W, box[1] * H, box[2] * W, box[3] * H],
                                             outline=(255, 0, 255), width=max(2, W // 100))
            im.thumbnail((cell, cell))
            x, y = (j % cols) * cell, (j // cols) * (cell + 18)
            sheet.paste(im, (x, y + 18)); draw.text((x + 4, y + 2), str(s + j), fill='red')
        sheet.save(f'{d}/sheet_{s // per_sheet:02d}.png')
    print(f'{len(entries)} images -> {d}/sheet_*.png (index.csv maps numbers to files)')


def apply_bg_rejections():
    gone = 0
    for log in (REJECTED_BG, f'{CLS_OUT}/rejected_backgrounds.txt'):   # the classifier's review still holds
        if os.path.exists(log):
            for name in open(log).read().split():
                if os.path.exists(f'{BG}/{name}'):
                    os.remove(f'{BG}/{name}'); gone += 1
    print(f'Re-applied {gone} earlier background rejections')


def prepare():
    random.seed(0)
    os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    unzip_all()
    shutil.rmtree(BG, ignore_errors=True); os.makedirs(BG)
    os.makedirs(NOTES, exist_ok=True)
    shutil.rmtree(f'{NOTES}/ieee', ignore_errors=True)
    ieee_notes()
    rf, names = find_roboflow()
    mapping = roboflow_class_map(names)
    roboflow_scenes(rf, mapping)
    public_note_datasets()
    roboflow_backgrounds(rf, mapping)
    if USE_EGOHANDS and not OFFLINE:
        egohands_backgrounds()
    if not OFFLINE:
        tar_backgrounds('dtd', DTD_URL, lambda f: os.path.basename(os.path.dirname(f)), DTD_PER_CLASS)
        tar_backgrounds('imagenette', IMAGENETTE_URL,
                        lambda f: os.path.basename(os.path.dirname(f)) if '/train/' in f else None,
                        IMAGENETTE_PER_CLASS)
    apply_bg_rejections()
    contact_sheet('bg', [(os.path.basename(p), p, None) for p in sorted(glob.glob(f'{BG}/rf*'))])
    print('Single-note photos per source:')
    for source in sorted(os.listdir(NOTES)):
        print(f'  {source:<24}', {l: len(images_in(f'{NOTES}/{source}/{l}')) for l in sorted(os.listdir(f'{NOTES}/{source}'))})
    print('Background images:', len(glob.glob(f'{BG}/*')))


def reject(kind, indices):
    assert kind in ('bg', 'box'), 'usage: reject bg|box <numbers...>'
    rows = dict((int(i), n) for i, n in csv.reader(open(f'{SHEETS}/{kind}/index.csv')))
    os.makedirs(OUT, exist_ok=True)
    with open(REJECTED_BG if kind == 'bg' else REJECTED_BOX, 'a') as log:
        for i in map(int, indices):
            name = rows[i]
            if kind == 'bg' and os.path.exists(f'{BG}/{name}'):
                os.remove(f'{BG}/{name}')
            log.write(name + '\n'); print('rejected', i, name)
    if kind == 'bg':
        print(len(glob.glob(f'{BG}/*')), 'background images left')


# ================================================================ autobox
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(small, big):
    """Share of `small` that lies inside `big`."""
    ix = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    area = (small[2] - small[0]) * (small[3] - small[1])
    return ix * iy / area if area > 0 else 0.0


def judge_box(candidates):
    """candidates: [(score, box)] sorted best first. Returns (status, box, score)."""
    if not candidates or candidates[0][0] < BOX_MIN_SCORE:
        return 'none', None, candidates[0][0] if candidates else 0.0
    score, box = candidates[0]
    for s, b in candidates[1:]:
        if s < BOX_RIVAL * score:
            break
        # a part of the note (numeral panel, portrait) or the same note again is fine; a separate box is not
        if inside(b, box) < 0.8 and inside(box, b) < 0.8 and iou(b, box) < 0.5:
            return 'ambiguous', box, score
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ('small' if area < BOX_MIN_AREA else 'ok'), box, score


def autobox():
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    os.makedirs(OUT, exist_ok=True)
    paths = [p for p in images_in(NOTES)]
    assert paths, 'No single-note photos: run prepare first'
    results = json.load(open(BOXES)) if os.path.exists(BOXES) else {}
    todo = [p for p in paths if rel(p) not in results]
    print(f'{len(paths)} single-note photos, {len(todo)} still to box')

    if todo:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        model = Owlv2ForObjectDetection.from_pretrained(
            OWL_MODEL, torch_dtype=torch.float16 if device == 'cuda' else torch.float32).to(device).eval()
        post = getattr(processor, 'post_process_object_detection', None) or processor.post_process_grounded_object_detection
        batch, started = 8, time.time()
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            images, geometry = [], []
            for p in chunk:
                im = Image.open(p).convert('RGB')
                W, H = im.size
                pad = int(OWL_PAD * max(W, H))
                images.append(ImageOps.expand(im, border=pad, fill=(128, 128, 128)))
                geometry.append((W, H, pad))
            inputs = processor(text=[OWL_QUERIES] * len(chunk), images=images, return_tensors='pt').to(device)
            if device == 'cuda':
                inputs['pixel_values'] = inputs['pixel_values'].half()
            with torch.no_grad():
                outputs = model(**inputs)
            # OWLv2 pads to a square at the bottom/right, so boxes are in a (side x side) frame
            sizes = torch.tensor([[max(im.size)] * 2 for im in images], device=device)
            detections = post(outputs=outputs, threshold=0.05, target_sizes=sizes)
            for p, det, (W, H, pad) in zip(chunk, detections, geometry):
                cands = []
                for s, box in zip(det['scores'].float().tolist(), det['boxes'].float().tolist()):
                    x0, y0, x1, y1 = [(v - pad) for v in box]
                    x0, x1 = min(max(x0 / W, 0), 1), min(max(x1 / W, 0), 1)
                    y0, y1 = min(max(y0 / H, 0), 1), min(max(y1 / H, 0), 1)
                    if x1 - x0 > 0.02 and y1 - y0 > 0.02:
                        cands.append((s, (x0, y0, x1, y1)))
                cands.sort(key=lambda c: -c[0])
                status, box, score = judge_box(cands)
                results[rel(p)] = {'status': status, 'box': box, 'score': round(score, 3)}
            if (b // batch) % 50 == 0 or b + batch >= len(todo):
                write_json(results, BOXES)
                done = b + len(chunk)
                print(f'  {done}/{len(todo)}  ({done / (time.time() - started):.1f} images/s)', flush=True)

    by_status = Counter(r['status'] for r in results.values())
    print('Autobox results:', dict(by_status))
    per_label = {}
    for k, r in results.items():
        label = coarse_of(k.split('/')[2])
        per_label.setdefault(label, Counter())[r['status']] += 1
    for label, c in sorted(per_label.items(), key=lambda kv: int(kv[0])):
        kept = c['ok'] / max(sum(c.values()), 1)
        print(f'  {label:>4}: {dict(c)}' + ('   <-- over 30% lost, look at the "small"/"none" photos' if kept < 0.7 else ''))
    # Spot check: a random sample of accepted boxes. Only 'ok' boxes are ever used.
    ok = sorted(k for k, r in results.items() if r['status'] == 'ok')
    sample = random.Random(3).sample(ok, min(240, len(ok)))
    contact_sheet('box', [(k, f'{WORK}/{k}', results[k]['box']) for k in sample])
    print('Review the box sheets: reject any photo whose magenta box misses the note, cuts off a big part '
          'of it, or holds more than one note.')


def accepted_boxes():
    results = json.load(open(BOXES)) if os.path.exists(BOXES) else {}
    rejected = set(open(REJECTED_BOX).read().split()) if os.path.exists(REJECTED_BOX) else set()
    return {k: r['box'] for k, r in results.items() if r['status'] == 'ok' and k not in rejected}


# ================================================================ splits
def dhash(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert('L').resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), 'big')


def build_manifest():
    """Rows of {path, kind, label, source, split}. kind is note, scene or background."""
    candidates = []
    for source in sorted(os.listdir(NOTES)):
        for label in sorted(os.listdir(f'{NOTES}/{source}')):
            candidates += [(f, 'note', label, source) for f in images_in(f'{NOTES}/{source}/{label}')]
    scenes = json.load(open(f'{WORK}/roboflow_scenes.json'))
    candidates += [(f'{WORK}/{k}', 'scene', '', 'roboflow') for k in sorted(scenes)]
    candidates += [(f, 'background', '', 'background') for f in images_in(BG)]

    def identify(c):
        try:
            return hashlib.md5(open(c[0], 'rb').read()).hexdigest(), dhash(c[0])
        except Exception as e:
            print('  bad image skipped:', c[0], e); return None
    with ThreadPoolExecutor(8) as ex:
        ids = list(ex.map(identify, candidates))

    seen, group_of_hash, rows, exact = set(), {}, [], 0
    for (path, kind, label, source), i in zip(candidates, ids):
        if i is None:
            continue
        if i[0] in seen:
            exact += 1; continue
        seen.add(i[0])
        group = group_of_hash.setdefault(i[1], f'{source}/{group_key(path)}')
        rows.append({'path': path, 'kind': kind, 'label': label, 'source': source, 'group': group, 'hash': i[1]})
    print(f'{len(rows)} unique images ({exact} exact duplicates dropped)')

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
        else:
            bucket = int(hashlib.md5(r['group'].encode()).hexdigest(), 16) % 100
            r['split'] = 'test' if bucket < 12 else 'val' if bucket < 24 else 'train'
    with open(f'{WORK}/manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['path', 'kind', 'label', 'source', 'split'], extrasaction='ignore')
        w.writeheader(); w.writerows(kept)
    return kept


# ================================================================ synth
def load_cutout(path, box):
    with Image.open(path) as im:
        im = im.convert('RGB')
        W, H = im.size
        return im.crop((int(box[0] * W), int(box[1] * H), math.ceil(box[2] * W), math.ceil(box[3] * H)))


def cover(im, W, H, rng):
    """Scale `im` to fill W x H, then take a random W x H window."""
    s = max(W / im.width, H / im.height) * rng.uniform(1.0, 1.3)
    im = im.resize((max(W, round(im.width * s)), max(H, round(im.height * s))), Image.BILINEAR)
    x, y = rng.randint(0, im.width - W), rng.randint(0, im.height - H)
    return im.crop((x, y, x + W, y + H))


def jitter(im, rng, strength=1.0):
    """Brightness, contrast, blur. No hue changes: colour is what tells denominations apart."""
    im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-0.3, 0.3) * strength)
    im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-0.2, 0.2) * strength)
    if rng.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return im


def make_scene(seed, bg_path, pool):
    """One composite: notes pasted on a background, later notes covering earlier ones.
    pool: {class id: [(path, box)]}. Returns (image, [(class id, x0, y0, x1, y1) normalised])."""
    rng = random.Random(seed)
    W, H = rng.choice(SCENE_SIZES)
    canvas = cover(Image.open(bg_path).convert('RGB'), W, H, rng)
    if rng.random() < 0.5:
        canvas = jitter(canvas, rng, 0.7)
    available = sorted(pool)
    n = rng.choices(list(NOTES_PER_SCENE), weights=list(NOTES_PER_SCENE.values()))[0]
    if n > 1 and rng.random() < SAME_DENOMINATION:
        classes = [rng.choice(available)] * n
    else:
        classes = [rng.choice(available) for _ in range(n)]
    shrink = 1.0 if n <= 2 else 0.8 if n <= 4 else 0.65

    owner = np.full((H, W), -1, np.int16)     # which note each pixel shows
    areas, labels = [], []
    for c in classes:
        note = jitter(load_cutout(*rng.choice(pool[c])), rng)
        s = rng.uniform(0.25, 0.55) * shrink * max(W, H) / max(note.size)
        note = note.resize((max(8, round(note.width * s)), max(8, round(note.height * s))), Image.BILINEAR)
        rgba = note.convert('RGBA').rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
        fit = min(1.0, 0.95 * W / rgba.width, 0.95 * H / rgba.height)
        if fit < 1.0:
            rgba = rgba.resize((max(8, int(rgba.width * fit)), max(8, int(rgba.height * fit))), Image.BILINEAR)
        alpha = np.asarray(rgba.getchannel('A')) > 127
        if alpha.sum() < 200:
            continue
        k, placed = len(areas), None
        for _ in range(20):
            x, y = rng.randint(0, W - rgba.width), rng.randint(0, H - rgba.height)
            trial = owner.copy()
            trial[y:y + rgba.height, x:x + rgba.width][alpha] = k
            if all((trial == i).sum() >= MIN_VISIBLE * areas[i] for i in range(k)):
                placed = (x, y, trial); break
        if placed is None:
            continue
        x, y, owner = placed
        soft = rgba.copy()
        soft.putalpha(rgba.getchannel('A').filter(ImageFilter.GaussianBlur(0.8)))
        canvas.paste(soft, (x, y), soft)
        areas.append(int(alpha.sum())); labels.append(c)

    boxes = []
    for i, c in enumerate(labels):
        ys, xs = np.nonzero(owner == i)
        boxes.append((c, xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H))
    if rng.random() < 0.3:
        a = np.asarray(canvas, dtype=np.float32) + np.random.default_rng(seed).normal(0, rng.uniform(2, 8), (H, W, 3))
        canvas = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.15:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.0)))
    return canvas, boxes


def write_yolo(split, name, im, boxes, quality=92):
    im.save(f'{YOLO}/images/{split}/{name}.jpg', 'JPEG', quality=quality)
    with open(f'{YOLO}/labels/{split}/{name}.txt', 'w') as f:
        for c, x0, y0, x1, y1 in boxes:
            f.write(f'{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n')


_POOLS = {}


def _init_worker(pools):
    _POOLS.update(pools)


def _scene_job(args):
    split, idx, seed, bg, out_dir = args
    im, boxes = make_scene(seed, bg, _POOLS[split])
    q = random.Random(seed + 1).randint(60, 95)
    if out_dir is None:
        write_yolo(split, f'synth_{idx:05d}', im, boxes, quality=q)
    else:
        im.save(f'{out_dir}/images/{idx:05d}.jpg', 'JPEG', quality=q)
    return idx, boxes


def synth():
    rows = build_manifest()
    boxes = accepted_boxes()
    assert boxes, 'No accepted boxes: run autobox first'
    scenes = json.load(open(f'{WORK}/roboflow_scenes.json'))
    shutil.rmtree(YOLO, ignore_errors=True); shutil.rmtree(EVAL, ignore_errors=True)
    for split in ('train', 'val', 'test'):
        os.makedirs(f'{YOLO}/images/{split}'); os.makedirs(f'{YOLO}/labels/{split}')
    rng = random.Random(42)

    by = lambda split, kind: [r for r in rows if r['split'] == split and r['kind'] == kind]
    pools, stats = {}, {}
    for split in ('train', 'val', 'test', 'external'):
        pool = {}
        for r in by(split, 'note'):
            box = boxes.get(rel(r['path']))
            if box:
                pool.setdefault(CLASSES.index(coarse_of(r['label'])), []).append((r['path'], box))
        for r in by(split, 'scene'):         # Roboflow notes too, if big enough to be a clean cutout
            with Image.open(r['path']) as im:
                W, H = im.size
            for c, x0, y0, x1, y1 in scenes[rel(r['path'])]:
                w, h = (x1 - x0) * W, (y1 - y0) * H
                if min(w, h) >= 48 and max(w, h) / min(w, h) <= 3.5:
                    pool.setdefault(c, []).append((r['path'], (x0, y0, x1, y1)))
        pools[split] = pool
        print(f'{split:>9} cutouts per class:', {CLASSES[c]: len(v) for c, v in sorted(pool.items())})
    for split in ('train', 'val', 'test', 'external'):
        assert len(pools[split]) == len(CLASSES), f'{split}: some class has no cutouts'

    jobs = []
    for split in ('train', 'val', 'test'):
        counts = Counter()
        # 1. real Roboflow scenes, as they are
        for r in by(split, 'scene'):
            name = 'rf_' + re.sub(r'[^A-Za-z0-9]', '_', os.path.splitext(os.path.basename(r['path']))[0])
            write_yolo(split, name, Image.open(r['path']).convert('RGB'), scenes[rel(r['path'])], quality=95)
            counts['real_scenes'] += 1
        # 2. real single-note photos with their autobox, capped per class
        per_class = {}
        notes = by(split, 'note'); rng.shuffle(notes)
        for r in notes:
            box = boxes.get(rel(r['path']))
            c = CLASSES.index(coarse_of(r['label']))
            if box and per_class.get(c, 0) < REAL_NOTES_PER_CLASS[split]:
                per_class[c] = per_class.get(c, 0) + 1
                write_yolo(split, f'note_{counts["real_notes"]:05d}', Image.open(r['path']).convert('RGB'),
                           [(c, *box)], quality=95)
                counts['real_notes'] += 1
        # 3. empty backgrounds
        bgs = [r['path'] for r in by(split, 'background')]
        assert bgs, f'No background images in {split}'
        for k, p in enumerate(rng.sample(bgs, min(NEGATIVES[split], len(bgs)))):
            write_yolo(split, f'empty_{k:05d}', Image.open(p).convert('RGB'), [], quality=95)
            counts['empty'] += 1
        # 4. composites
        for i in range(SYNTH_COUNTS[split]):
            jobs.append((split, i, rng.randrange(1 << 31), rng.choice(bgs), None))
        counts['synthetic'] = SYNTH_COUNTS[split]
        stats[split] = dict(counts)

    # External count test: composites of external notes on test backgrounds. Never trained on.
    ext_dir = f'{EVAL}/external_scenes'; os.makedirs(f'{ext_dir}/images')
    test_bgs = [r['path'] for r in by('test', 'background')]
    for i in range(EXTERNAL_SCENES):
        jobs.append(('external', i, rng.randrange(1 << 31), rng.choice(test_bgs), ext_dir))

    print(f'building {len(jobs)} composite scenes')
    ext_labels, done = {}, 0
    with ProcessPoolExecutor(os.cpu_count() or 4, initializer=_init_worker, initargs=(pools,)) as ex:
        for job, (idx, bxs) in zip(jobs, ex.map(_scene_job, jobs, chunksize=32)):
            if job[0] == 'external':
                ext_labels[f'{idx:05d}.jpg'] = bxs
            done += 1
            if done % 1000 == 0:
                print(f'  {done}/{len(jobs)}', flush=True)
    write_json(ext_labels, f'{ext_dir}/labels.json')
    stats['external_scenes'] = len(ext_labels)
    stats['external_photos'] = len(by('external', 'note'))

    with open(f'{YOLO}/data.yaml', 'w') as f:
        f.write(f'path: {YOLO}\ntrain: images/train\nval: images/val\ntest: images/test\n')
        f.write('names:\n' + ''.join(f'  {i}: "{c}"\n' for i, c in enumerate(CLASSES)))
    write_json(stats, f'{OUT}/dataset_stats.json')
    print('YOLO dataset:', json.dumps(stats, indent=1))
    # a look at what the model will be trained on
    sample = sorted(glob.glob(f'{YOLO}/images/train/synth_*.jpg'))[:24]
    sheet = Image.new('RGB', (6 * 213, 4 * 213), 'white')
    for j, p in enumerate(sample):
        im = Image.open(p).convert('RGB'); W, H = im.size
        d = ImageDraw.Draw(im)
        for line in open(p.replace('/images/', '/labels/')[:-4] + '.txt'):
            c, cx, cy, w, h = line.split(); cx, cy, w, h = map(float, (cx, cy, w, h))
            d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], outline=(255, 0, 255), width=3)
            d.text(((cx - w / 2) * W + 4, (cy - h / 2) * H + 2), CLASSES[int(c)], fill=(255, 0, 255))
        im.thumbnail((213, 213)); sheet.paste(im, ((j % 6) * 213, (j // 6) * 213))
    sheet.save(f'{OUT}/synthetic_samples.jpg', quality=90)


# ================================================================ train / export
def run_dir():
    return f'{RUNS}/detector'


def train():
    import torch
    from ultralytics import YOLO as UltralyticsYOLO, settings
    for key in ('wandb', 'mlflow', 'comet', 'clearml', 'tensorboard'):     # no logging services on Kaggle
        try:
            settings.update({key: False})
        except Exception:
            pass
    assert os.path.exists(f'{YOLO}/data.yaml'), 'No YOLO dataset: run synth first'
    os.makedirs(OUT, exist_ok=True)
    last, done_flag = f'{run_dir()}/weights/last.pt', f'{run_dir()}/DONE'
    if os.environ.get('DETECTOR_FRESH'):
        shutil.rmtree(run_dir(), ignore_errors=True)
    if os.path.exists(done_flag):
        print('Training already finished (delete', done_flag, 'or set DETECTOR_FRESH=1 to train again)')
        return
    gpus = torch.cuda.device_count()
    device = os.environ.get('DETECTOR_DEVICE') or (','.join(map(str, range(gpus))) if gpus else 'cpu')
    print('torch', torch.__version__, '| GPUs:', gpus, '| device:', device)
    if os.path.exists(last):
        print('Resuming from', last)
        UltralyticsYOLO(last).train(resume=True)
    else:
        UltralyticsYOLO('yolo11n.pt').train(
            data=f'{YOLO}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, patience=30, batch=BATCH, device=device,
            workers=4, project=RUNS, name='detector', exist_ok=True, seed=42, deterministic=False,
            fraction=FRACTION, plots=True,
            # no flips (mirrored digits never occur), no hue shift (colour separates denominations),
            # no extra rotation (the composites already turn notes through 360°)
            fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.4, hsv_v=0.4, degrees=0.0,
            translate=0.1, scale=0.5, mosaic=1.0, close_mosaic=15, mixup=0.0)
    open(done_flag, 'w').write(time.strftime('%Y-%m-%d %H:%M'))
    shutil.copy(f'{run_dir()}/weights/best.pt', f'{OUT}/best.pt')
    print('best weights ->', f'{OUT}/best.pt')


def load_interpreter(path, threads=4):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    return it


def export():
    from ultralytics import YOLO as UltralyticsYOLO
    best = f'{OUT}/best.pt'
    assert os.path.exists(best), 'No best.pt: run train first'
    path = UltralyticsYOLO(best).export(format='tflite', imgsz=IMGSZ, half=False, int8=False, nms=False)
    if not str(path).endswith('_float32.tflite'):
        found = glob.glob(f'{os.path.dirname(str(path))}/**/*_float32.tflite', recursive=True) or [path]
        path = found[0]
    shutil.copy(path, TFLITE)
    with open(LABELS, 'w') as f:
        f.write('\n'.join(CLASSES) + '\n')
    it = load_interpreter(TFLITE)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    assert list(inp['shape']) == [1, IMGSZ, IMGSZ, 3] and inp['dtype'] == np.float32, inp
    assert len(out['shape']) == 3 and out['shape'][1] == 4 + len(CLASSES) and out['dtype'] == np.float32, out
    # Are box coordinates 0-1 or pixels? The app must know.
    probe = sorted(glob.glob(f'{YOLO}/images/val/note_*.jpg'))[:1] or sorted(glob.glob(f'{YOLO}/images/val/*.jpg'))[:1]
    raw = run_tflite(it, Image.open(probe[0]).convert('RGB'))[0]
    normalised = bool(np.abs(raw[:4]).max() <= 2.0)
    info = {'input': [int(v) for v in inp['shape']], 'input_dtype': 'float32',
            'input_values': 'RGB 0-1, letterboxed (aspect kept, centred, padding value 114/255)',
            'output': [int(v) for v in out['shape']], 'output_layout': '[1, 4 + classes, anchors]: cx, cy, w, h, then one score per class',
            'box_coordinates': 'normalised 0-1 of the input square' if normalised else f'pixels of the {IMGSZ} input square',
            'labels': CLASSES, 'nms': f'class-agnostic, IoU {NMS_IOU}, done by the app',
            'tflite_mb': round(os.path.getsize(TFLITE) / 1e6, 2)}
    write_json(info, f'{OUT}/export.json')
    print(json.dumps(info, indent=1))


# ================================================================ evaluate
def letterbox(im, size=IMGSZ):
    """Exactly what the app must do: keep the aspect, scale to fit, centre on grey 114."""
    W, H = im.size
    s = min(size / W, size / H)
    w, h = round(W * s), round(H * s)
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    left, top = (size - w) // 2, (size - h) // 2
    canvas.paste(im.resize((w, h), Image.BILINEAR), (left, top))
    return np.asarray(canvas, dtype=np.float32)[None] / 255.0, (s, left, top)


def run_tflite(it, im):
    x, _ = letterbox(im)
    it.set_tensor(it.get_input_details()[0]['index'], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]['index'])[0]


def decode(raw, geom, W, H, conf_min=0.05):
    """[4 + nc, N] -> [(class, score, x0, y0, x1, y1)] in 0-1 of the original image, after agnostic NMS."""
    s, left, top = geom
    p = raw.T
    scores = p[:, 4:]
    cls, conf = scores.argmax(1), scores.max(1)
    keep = conf >= conf_min
    p, cls, conf = p[keep], cls[keep], conf[keep]
    xywh = p[:, :4] * (IMGSZ if np.abs(p[:, :4]).max(initial=0) <= 2.0 else 1.0)
    x0 = (xywh[:, 0] - xywh[:, 2] / 2 - left) / s / W; x1 = (xywh[:, 0] + xywh[:, 2] / 2 - left) / s / W
    y0 = (xywh[:, 1] - xywh[:, 3] / 2 - top) / s / H; y1 = (xywh[:, 1] + xywh[:, 3] / 2 - top) / s / H
    order, kept = np.argsort(-conf), []
    for i in order:
        b = (x0[i], y0[i], x1[i], y1[i])
        if all(iou(b, k[2:]) < NMS_IOU for k in kept):
            kept.append((int(cls[i]), float(conf[i]), *map(float, b)))
    return kept


def predict_all(it, items):
    """items: [(image path, truth Counter)] -> [(truth, detections)]."""
    if EVAL_LIMIT:
        items = items[:EVAL_LIMIT]
    out = []
    for path, truth in items:
        im = Image.open(path).convert('RGB')
        x, geom = letterbox(im)
        it.set_tensor(it.get_input_details()[0]['index'], x)
        it.invoke()
        raw = it.get_tensor(it.get_output_details()[0]['index'])[0]
        out.append((truth, decode(raw, geom, *im.size)))
    return out


def count_table(preds):
    """What the app would say, per image, at each threshold."""
    table = {}
    notes = [p for p in preds if sum(p[0].values())]
    empty = [p for p in preds if not sum(p[0].values())]
    for t in THRESHOLDS:
        said = lambda dets: Counter(CLASSES[d[0]] for d in dets if d[1] >= t)
        exact = sum(said(d) == truth for truth, d in notes)
        total = sum(sum(int(k) * v for k, v in said(d).items()) == sum(int(k) * v for k, v in truth.items())
                    for truth, d in notes)
        silent = sum(not said(d) for _, d in notes)
        table[t] = {'exact_counts': round(exact / max(len(notes), 1), 4),
                    'right_total': round(total / max(len(notes), 1), 4),
                    'silent_on_notes': round(silent / max(len(notes), 1), 4),
                    'speaks_on_background': round(sum(bool(said(d)) for _, d in empty) / len(empty), 4) if empty else None}
    return table


def yolo_items(split):
    items = []
    for img in sorted(glob.glob(f'{YOLO}/images/{split}/*.jpg')):
        lines = open(img.replace('/images/', '/labels/')[:-4] + '.txt').read().split('\n')
        items.append((img, Counter(CLASSES[int(l.split()[0])] for l in lines if l.strip())))
    return items


def evaluate():
    import matplotlib; matplotlib.use('Agg')
    assert os.path.exists(TFLITE), 'No exported model: run export first'
    it = load_interpreter(TFLITE)
    results = {'imgsz': IMGSZ, 'tflite_mb': round(os.path.getsize(TFLITE) / 1e6, 2)}

    # mAP from Ultralytics on the test split (boxes, not just counts)
    try:
        from ultralytics import YOLO as UltralyticsYOLO
        m = UltralyticsYOLO(f'{OUT}/best.pt').val(data=f'{YOLO}/data.yaml', split='test', imgsz=IMGSZ,
                                                   project=RUNS, name='test', exist_ok=True, plots=True)
        results['test_map50'] = round(float(m.box.map50), 4)
        results['test_map50_95'] = round(float(m.box.map), 4)
        results['test_map50_per_class'] = {CLASSES[int(c)]: round(float(v), 4) for c, v in zip(m.box.ap_class_index, m.box.ap50)}
    except Exception as e:
        print('!! Ultralytics validation failed, continuing:', e)

    ext_dir = f'{EVAL}/external_scenes'
    ext_labels = json.load(open(f'{ext_dir}/labels.json'))
    ext_photos = [(r['path'], Counter({coarse_of(r['label']): 1}))
                  for r in csv.DictReader(open(f'{WORK}/manifest.csv')) if r['split'] == 'external' and r['kind'] == 'note']
    sets = {
        'val': yolo_items('val'),
        'test': yolo_items('test'),
        'external_scenes': [(f'{ext_dir}/images/{k}', Counter(CLASSES[b[0]] for b in v)) for k, v in sorted(ext_labels.items())],
        'external_photos': ext_photos,
    }
    tables = {}
    for name, items in sets.items():
        print(f'running the .tflite on {name}: {len(items) if not EVAL_LIMIT else min(len(items), EVAL_LIMIT)} images', flush=True)
        preds = predict_all(it, items)
        tables[name] = count_table(preds)
        if name == 'external_scenes':
            draw_samples(preds[:24], [i[0] for i in items[:24]], f'{OUT}/samples_external.jpg')

    # The lowest threshold whose false speech on empty validation images is acceptable, best exact count
    allowed = [t for t in THRESHOLDS if (tables['val'][t]['speaks_on_background'] or 0) <= MAX_FALSE_SPEECH]
    rec = max(allowed, key=lambda t: tables['val'][t]['exact_counts']) if allowed else THRESHOLDS[-1]
    results['recommended_threshold'] = rec
    for name in sets:
        results[f'{name}_at_threshold'] = tables[name][rec]
    results['tables'] = tables

    baseline = None
    if os.path.exists(f'{CLS_OUT}/results.json'):
        baseline = json.load(open(f'{CLS_OUT}/results.json')).get('external_accuracy')
    results['classifier_external_accuracy'] = baseline
    results['note_on_baseline'] = ('external_photos exact_counts (one note, right denomination, nothing else) '
                                   'is the closest match to the classifier\'s external_accuracy; it is stricter.')

    # Speed of the exported model on this machine's CPU (a phone is slower; the app will log its own)
    im = Image.open(sets['val'][0][0]).convert('RGB')
    run_tflite(it, im)
    started = time.time()
    for _ in range(20):
        run_tflite(it, im)
    results['cpu_ms_per_frame_kaggle'] = round((time.time() - started) / 20 * 1000, 1)
    write_json(results, f'{OUT}/results.json')
    summary = {k: v for k, v in results.items() if k != 'tables'}
    print(json.dumps(summary, indent=1))


def draw_samples(preds, paths, dst):
    sheet = Image.new('RGB', (6 * 213, 4 * 213), 'white')
    for j, ((truth, dets), p) in enumerate(zip(preds, paths)):
        im = Image.open(p).convert('RGB'); W, H = im.size
        d = ImageDraw.Draw(im)
        for c, s, x0, y0, x1, y1 in dets:
            if s >= 0.5:
                d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], outline=(0, 255, 0), width=3)
                d.text((x0 * W + 4, y0 * H + 2), f'{CLASSES[c]} {s:.2f}', fill=(0, 255, 0))
        d.text((4, H - 14), 'truth: ' + ', '.join(f'{v}x{k}' for k, v in sorted(truth.items())), fill=(255, 0, 0))
        im.thumbnail((213, 213)); sheet.paste(im, ((j % 6) * 213, (j // 6) * 213))
    sheet.save(dst, quality=90)


# ================================================================ main
STAGES = {'prepare': prepare, 'autobox': autobox, 'synth': synth, 'train': train,
          'export': export, 'evaluate': evaluate}

if __name__ == '__main__':
    args = sys.argv[1:]
    assert args, __doc__
    if args[0] == 'reject':
        reject(args[1], args[2:])
    else:
        for stage in args:
            assert stage in STAGES, f'unknown stage {stage!r}; stages: {", ".join(STAGES)} or reject'
            print(f'==== {stage}', flush=True)
            STAGES[stage]()
    print('Done.')

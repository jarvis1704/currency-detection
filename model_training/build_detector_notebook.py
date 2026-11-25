"""
Builds detector_training.ipynb for Kaggle from train_detector.py.
train_detector.py is the source of truth: edit it, then run `python build_detector_notebook.py`
and upload the new notebook. The notebook writes the script to /kaggle/working itself.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = open(os.path.join(HERE, 'train_detector.py')).read()


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


cells = [
    md("""
# Indian Currency Detector (multi-note) — YOLO11n

Trains a detector that finds **every** banknote in the camera frame and labels its denomination, so the app can say
*"two 100 rupee notes, one 500 rupee note, total 700 rupees"*. It is trained next to the existing classifier and
writes only to `/kaggle/working/currency_detector`, so the deployed classifier and its files stay untouched.

### Running on Kaggle
1. **Settings:** Accelerator → **GPU T4 x2**, **Internet → on**, **Persistence → Files only**
   (so `/kaggle/working` survives a restart and finished stages are not redone).
2. **Inputs:** paste the share link of your Google Drive `currency_raw` folder (shared as *Anyone with the link*)
   into `DRIVE_FOLDER_URL` in step 1. It must hold the IEEE zip and the Roboflow **YOLOv8** export zip.
   The zips are downloaded once and kept in `/kaggle/working/currency_raw`.
3. Run the cells **one at a time** (the two review steps need your eyes). If the session stops, run the cells again
   from the top: downloads, autobox results, review decisions and training checkpoints are kept.
4. At the end download the results from the Output panel and use **Save Version → Quick Save**.

| Step | Time on T4 x2 (rough) |
|---|---|
| prepare | 5–15 min (less when the classifier's Mendeley cache exists) |
| autobox | 20–40 min |
| synth | 5–10 min |
| train (150 epochs max, early stop) | 2–4 h |
| export + evaluate | 10 min |
"""),
    md("""
## 1. Environment and inputs
Paste your Drive folder link below, then run. The first run downloads the zips from Drive (a few minutes);
later runs reuse them.

**Expected output:** two Tesla T4 GPUs, then the zips with their sizes.
"""),
    code("""
# ▼ Paste the share link of your Google Drive folder that holds the input zips (currency_raw).
#   It must be shared as "Anyone with the link". Example: https://drive.google.com/drive/folders/1AbC...xyz
DRIVE_FOLDER_URL = ''

!nvidia-smi --query-gpu=name,memory.total --format=csv
import os, glob, shutil, subprocess, zipfile

RAW = '/kaggle/working/currency_raw'
os.makedirs(RAW, exist_ok=True)
good_zips = lambda: [z for z in glob.glob(f'{RAW}/*.zip') if zipfile.is_zipfile(z)]

if not good_zips():
    # Fallback: zips attached as a Kaggle dataset (+ Add Input). Download caches are not inputs.
    for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
        if '/cache/' not in z and not os.path.basename(z).startswith('mendeley-'):
            shutil.copy(z, RAW); print('copied', z)

if not good_zips():
    assert DRIVE_FOLDER_URL.startswith('https://drive.google.com/'), \
        'Paste your Google Drive folder link into DRIVE_FOLDER_URL at the top of this cell, then run it again'
    subprocess.run(['pip', 'install', '-q', '-U', 'gdown'], check=True)
    download = f'{RAW}/_drive'
    shutil.rmtree(download, ignore_errors=True)
    result = subprocess.run(['gdown', '--folder', DRIVE_FOLDER_URL, '-O', download, '--remaining-ok'])
    for z in glob.glob(f'{download}/**/*.zip', recursive=True):     # gdown may nest a subfolder
        shutil.move(z, f'{RAW}/{os.path.basename(z)}')
    shutil.rmtree(download, ignore_errors=True)
    broken = [z for z in glob.glob(f'{RAW}/*.zip') if not zipfile.is_zipfile(z)]
    for z in broken:
        os.remove(z)
    assert good_zips() and not broken and result.returncode == 0, (
        'Download from Drive failed or is incomplete. Check that the folder is shared as "Anyone with the link". '
        'If gdown printed "Too many users have viewed or downloaded this file recently", wait an hour and run '
        'this cell again (finished zips are kept).')

os.environ['CURRENCY_RAW']  = RAW
os.environ['DETECTOR_OUT']  = '/kaggle/working/currency_detector'
os.environ['DETECTOR_WORK'] = '/kaggle/temp/detector'
os.environ['CURRENCY_OUT']  = '/kaggle/working/currency'          # classifier outputs: read only, may not exist
for d in (os.environ['DETECTOR_OUT'], os.environ['DETECTOR_WORK']):
    os.makedirs(d, exist_ok=True)
for z in sorted(good_zips()):
    print(f'{os.path.getsize(z) / 1e6:8.1f} MB  {os.path.basename(z)}')
"""),
    md("## 2. Dependencies"),
    code("""
!pip install -q "ultralytics>=8.3.0" "transformers>=4.40" pyyaml
import ultralytics, transformers
print('ultralytics', ultralytics.__version__, '| transformers', transformers.__version__)
"""),
    md("""
## 3. Training pipeline
The next cell writes `train_detector.py` (generated from the repo copy, don't edit it here). Stages:
`prepare` → `reject bg` → `autobox` → `reject box` → `synth` → `train` → `export` → `evaluate`.

**Why composites:** only the Roboflow photos have boxes, and few show several notes. Every other source is one note per
photo. The script finds the box in those photos (OWLv2), cuts the notes out and pastes 1–6 of them, rotated, scaled and
overlapping, onto background photos. That gives exact boxes for multi-note scenes, including stacks of the same note.
"""),
    code('%%writefile /kaggle/working/train_detector.py\n' + SCRIPT),
    code("""
import os, subprocess, time

def run_stage(stages, log, env=None):
    \"\"\"Runs stages as their own process, so a dropped browser connection doesn't stop them.
    Running the cell again while it is alive just shows the progress.\"\"\"
    pid_file = log + '.pid'
    def alive(pid):
        try:
            return open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[0] != 'Z'
        except OSError:
            return False
    pid = int(open(pid_file).read()) if os.path.exists(pid_file) else None
    if pid and alive(pid):
        print('Already running: showing its progress.\\n')
    else:
        command = f'cd /kaggle/working && python -u train_detector.py {stages}; echo "__EXIT__ $?"'
        with open(log, 'w') as out:
            pid = subprocess.Popen(command, shell=True, stdout=out, stderr=subprocess.STDOUT,
                                   env={**os.environ, **(env or {})}, start_new_session=True).pid
        open(pid_file, 'w').write(str(pid))
    position = 0
    while True:
        with open(log, errors='ignore') as f:
            f.seek(position); text = f.read(); position = f.tell()
        if '__EXIT__' in text:
            before, _, code = text.partition('__EXIT__')
            print(before, end='', flush=True)
            code = int(code.split()[0])
            if code != 0:
                raise RuntimeError(f'{stages} failed with exit code {code}'
                                   + (' (137 means out of memory)' if code == 137 else ''))
            print('\\nFinished OK.'); return
        if text:
            print(text, end='', flush=True)
        if not alive(pid):
            raise RuntimeError(f'{stages} was killed before it reported a result. Run this cell again.')
        time.sleep(5)

LOGS = '/kaggle/working/currency_detector/logs'
os.makedirs(LOGS, exist_ok=True)
"""),
    md("""
## 4. Data preparation
Unzips the inputs, downloads the Mendeley sets (or restores them from the classifier's cache), keeps the Roboflow
photos whole with their boxes, and builds the background pool. Prints the Roboflow class mapping: check that every
denomination maps to the right number and that `₹2000`-style classes map to `None` (those photos are skipped).
"""),
    code("run_stage('prepare', f'{LOGS}/prepare.log')"),
    md("""
### 4.1 Background review
Background crops from Roboflow photos must show **no part of a banknote**. Rejections made in the classifier notebook
are re-applied automatically, so only check new ones. Page through with `SHEET`, then list numbers in `REJECT`.
"""),
    code("""
import glob
from IPython.display import display, Image as IPImage
sheets = sorted(glob.glob('/kaggle/temp/detector/contact_sheets/bg/sheet_*.png'))
print(len(sheets), 'sheets')
SHEET = 0
display(IPImage(sheets[SHEET]))
"""),
    code("""
REJECT = []   # e.g. [3, 17, 40]
if REJECT:
    subprocess.run(['python', 'train_detector.py', 'reject', 'bg', *map(str, REJECT)], cwd='/kaggle/working', check=True)
else:
    print('nothing to reject')
"""),
    md("""
## 5. Autobox
Runs OWLv2 on every single-note photo to find its one note. Only confident, single, reasonably large boxes are kept;
ambiguous ones are dropped automatically (there is plenty of data). Resumable: results go to `boxes.json`.
**Watch the per-class lines:** a class that loses more than 30 % of its photos is flagged.
"""),
    code("run_stage('autobox', f'{LOGS}/autobox.log')"),
    md("""
### 5.1 Box spot-check
A random sample of 240 accepted boxes. Reject any photo whose magenta box misses the note, cuts off a big part of it,
or covers more than one note. If more than ~5 % are wrong, tell Claude: the thresholds need tightening.
"""),
    code("""
sheets = sorted(glob.glob('/kaggle/temp/detector/contact_sheets/box/sheet_*.png'))
print(len(sheets), 'sheets')
SHEET = 0
display(IPImage(sheets[SHEET]))
"""),
    code("""
REJECT = []
if REJECT:
    subprocess.run(['python', 'train_detector.py', 'reject', 'box', *map(str, REJECT)], cwd='/kaggle/working', check=True)
else:
    print('nothing to reject')
"""),
    md("""
## 6. Splits and composite scenes
Builds the YOLO dataset: real Roboflow scenes, real single-note photos, empty backgrounds and composites, split by
source photo so copies never cross splits. The Mendeley `8ckhkssyn3` notes are kept out entirely and become the
external test (single photos plus composites). **Look at the sample sheet:** boxes must sit on visible notes.
"""),
    code("""
run_stage('synth', f'{LOGS}/synth.log')
display(IPImage('/kaggle/working/currency_detector/synthetic_samples.jpg'))
"""),
    md("""
## 7. Smoke test
Trains 1 epoch on 5 % of the data, exports and evaluates a few images into a throwaway folder. Catches problems
in minutes instead of hours. The numbers mean nothing.
"""),
    code("""
run_stage('train export evaluate', f'{LOGS}/dryrun.log',
          {'DETECTOR_OUT': '/kaggle/temp/detector_dryrun', 'DETECTOR_EPOCHS': '1', 'DETECTOR_FRACTION': '0.05',
           'DETECTOR_EVAL_LIMIT': '30', 'DETECTOR_FRESH': '1'})
"""),
    md("""
## 8. Full training
YOLO11n, 416 × 416, up to 150 epochs with early stopping (patience 30), both GPUs. If the session ends, run this cell
again: it resumes from `last.pt`.
"""),
    code("run_stage('train', f'{LOGS}/train.log')"),
    md("## 9. Export to LiteRT (.tflite) and evaluate"),
    code("run_stage('export evaluate', f'{LOGS}/evaluate.log')"),
    code("""
import json
r = json.load(open('/kaggle/working/currency_detector/results.json'))
t = r['recommended_threshold']
print(f"Recommended threshold          : {t}")
print(f"Test mAP50 / mAP50-95          : {r.get('test_map50')} / {r.get('test_map50_95')}")
for name in ('test', 'external_scenes', 'external_photos'):
    m = r[f'{name}_at_threshold']
    print(f"{name:<16} exact counts {m['exact_counts']:.1%} | right total {m['right_total']:.1%} | "
          f"silent on notes {m['silent_on_notes']:.1%}")
print(f"Speaks on empty test images    : {r['test_at_threshold']['speaks_on_background']:.1%}")
if r.get('classifier_external_accuracy') is not None:
    print(f"Classifier external accuracy   : {r['classifier_external_accuracy']:.1%}  (compare with external_photos exact counts)")
print(f"Model size                     : {r['tflite_mb']} MB, {r['cpu_ms_per_frame_kaggle']} ms/frame on Kaggle CPU")
for f in ('samples_external.jpg', 'runs/detector/results.png', 'runs/test/confusion_matrix_normalized.png'):
    p = f'/kaggle/working/currency_detector/{f}'
    if os.path.exists(p):
        print(f); display(IPImage(p, width=900))
"""),
    md("""
## 10. Hand-off (not integrated yet)
Save the version, then download from `currency_detector/`: `currency_detector.tflite`, `detector_labels.txt`,
`export.json` (input/output format the app must follow) and `results.json` (including the recommended threshold).
The app keeps using the classifier until the integration work on the `feat/multi-detect` branch is done.
"""),
]

nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                   'language_info': {'name': 'python'},
                   'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': True, 'isGpuEnabled': True}}}
with open(os.path.join(HERE, 'detector_training.ipynb'), 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote detector_training.ipynb')

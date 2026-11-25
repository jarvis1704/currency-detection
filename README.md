<div align="center">

# 💵 Currency Reader

**Offline Indian banknote recognition for visually impaired users — built with Jetpack Compose, CameraX, LiteRT, and On-Device ML.**

[![Platform](https://img.shields.io/badge/Platform-Android%208.0%2B%20(API%2026)-3DDC84?style=flat-square&logo=android&logoColor=white)](android/app/build.gradle.kts)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.2.10-7F52FF?style=flat-square&logo=kotlin&logoColor=white)](android/gradle/libs.versions.toml)
[![LiteRT](https://img.shields.io/badge/LiteRT-1.4.2-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](android/gradle/libs.versions.toml)
[![Jetpack Compose](https://img.shields.io/badge/Jetpack%20Compose-Material%203-4285F4?style=flat-square&logo=jetpackcompose&logoColor=white)](android/gradle/libs.versions.toml)

*Point the camera at an Indian banknote and hear the denomination spoken aloud — completely offline, with zero data leaving the device.*

</div>

---

## ✨ Overview

**Currency Reader** (`com.biprangshu.currency`) is an accessibility-first Android application that identifies Indian rupee banknotes (**₹10, ₹20, ₹50, ₹100, ₹200, ₹500**) in real time and announces them via speech and haptic feedback.

### Key Features

- **100% On-Device & Offline**: All classification, object localization, and speech synthesis run locally on the phone. No internet connection or cloud API is used.
- **Supported Denominations**: Recognizes **₹10**, **₹20**, **₹50**, **₹100**, **₹200**, and **₹500** notes (covering both old and new note series where applicable).
- **Accessibility-First UX**:
  - **Voice + Haptics**: Automatically speaks detected denominations (preferring `en-IN`, falling back to English) alongside a 150 ms vibration pulse.
  - **Tap Anywhere to Repeat**: Tapping anywhere on the scanner screen repeats the last detected note (or prompts if none has been detected yet).
  - **High-Contrast Dark UI**: Fixed `#000000` background with `#FFD600` signal-yellow accents and `96 sp` denomination display for low-vision readability.
  - **TalkBack Support**: Uses live regions for errors, unified scanner accessibility descriptions, and keeps the screen awake (`FLAG_KEEP_SCREEN_ON`) with volume keys bound to `STREAM_MUSIC`.
  - **Low-Light Torch Toggle**: Full-width `72 dp` button to toggle the camera flash when a flash unit is available.
- **Flicker-Free Stabilization & Live Highlighting**:
  - Temporal smoothing (`Stabilizer`) requires **4 out of 6** recent frames to agree at **≥ 0.70** confidence before announcing, and suppresses repeated announcements until background is seen for 4 consecutive frames.
  - Optional ML Kit on-device object detection (`NoteLocator`) draws animated corner markers and a denomination pill around centered notes in the viewfinder.

---

## 🏗️ Architecture

### Real-Time Android Pipeline

```
CameraX ImageAnalysis (640×480, 4:3, RGBA_8888, Keep-Only-Latest)
        │
        ▼
FramePreprocessor ──► Center square crop ─► 224×224 bilinear resize ─► Float32 RGB [0..255]
        │
        ▼
CurrencyClassifier (LiteRT Interpreter, 2 threads, currency_model.tflite)
        │
        ├──► Stabilizer (≥ 0.70 conf in 4/6 frames) ──► Announcer (TTS + 150ms Vibration)
        │
        └──► NoteHighlighter + NoteLocator (ML Kit ObjectDetection) ──► NoteHighlightOverlay
```

### Repository Layout

```text
currency-detection/
├── currency_model.tflite                    # Earlier quantized model artifact
├── android/                                 # Native Android application (Kotlin + Compose)
│   ├── app/
│   │   ├── build.gradle.kts                 # App config (minSdk 26, targetSdk 37, R8 release)
│   │   ├── release/                         # Built release APK & mapping outputs
│   │   └── src/
│   │       ├── main/
│   │       │   ├── assets/
│   │       │   │   ├── currency_model.tflite  # Bundled MobileNetV3-Small LiteRT model (3.7 MB)
│   │       │   │   └── labels.txt             # Class order: 10, 100, 20, 200, 50, 500, background
│   │       │   ├── java/com/biprangshu/currency/
│   │       │   │   ├── MainActivity.kt        # Edge-to-edge host, keep-screen-on, media volume
│   │       │   │   ├── recognition/
│   │       │   │   │   ├── CurrencyClassifier.kt # LiteRT interpreter & asset contract validation
│   │       │   │   │   ├── FramePreprocessor.kt  # Center-crop & bilinear float32 resize (224×224)
│   │       │   │   │   ├── NormalizedBox.kt      # Sensor-space box math & crop-bounds check
│   │       │   │   │   ├── NoteHighlighter.kt    # Hold-frame smoothing for visual box stability
│   │       │   │   │   ├── NoteLocator.kt        # ML Kit bundled object detector wrapper
│   │       │   │   │   └── Stabilizer.kt         # Sliding-window prediction debouncer
│   │       │   │   ├── scanner/
│   │       │   │   │   ├── Announcer.kt          # TextToSpeech (en-IN / en) + Vibrator helper
│   │       │   │   │   ├── NoteHighlightOverlay.kt # Animated Canvas corner markers & pill label
│   │       │   │   │   ├── ScannerScreen.kt      # CameraX viewfinder, permissions & high-contrast UI
│   │       │   │   │   ├── ScannerState.kt       # Unidirectional UI state, errors & actions
│   │       │   │   │   └── ScannerViewModel.kt   # Single-thread analysis executor & lifecycle
│   │       │   │   └── ui/theme/               # High-contrast dark Material 3 palette & typography
│   │       │   └── res/                        # Strings, icons, adaptive launcher, XML rules
│   │       └── test/java/com/biprangshu/currency/recognition/
│   │           ├── FramePreprocessorTest.kt
│   │           ├── NormalizedBoxTest.kt
│   │           ├── NoteHighlighterTest.kt
│   │           └── StabilizerTest.kt
│   └── gradle/libs.versions.toml            # Version catalog
└── model_training/                          # Dataset prep, training, evaluation & export
    ├── currency_training.ipynb              # Classifier training notebook (source of truth)
    ├── train_currency.py                    # Standalone classifier prep/train/export CLI
    ├── train_detector.py                    # Multi-note YOLO11n detector pipeline (in development)
    ├── build_detector_notebook.py           # Generates detector_training.ipynb from train_detector.py
    └── detector_training.ipynb              # Generated Kaggle GPU notebook for YOLO11n detector
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language & Build** | Kotlin `2.2.10`, Android Gradle Plugin `9.5.1`, Java 11 bytecode |
| **UI** | Jetpack Compose (`2026.02.01` BOM), Material 3, Custom High-Contrast Theme |
| **Camera** | CameraX `1.6.2` (`camera2`, `camera-compose`, `camera-core`, `camera-lifecycle`) |
| **ML Inference** | Google AI Edge LiteRT `1.4.2` (`com.google.ai.edge.litert:litert`) |
| **Note Localization** | ML Kit On-Device Object Detection `17.0.2` (`STREAM_MODE`) |
| **Speech & Haptics** | Android `TextToSpeech` & `Vibrator` |
| **Model Training** | TensorFlow / Keras (MobileNetV3-Small), Ultralytics YOLO11n, OWLv2 (`transformers`) |

---

## 🚀 Getting Started (Android App)

### Prerequisites

- **Android Studio** (supporting AGP `9.5.1` / JDK 11+ / Android SDK 37)
- An Android device running **Android 8.0 (API 26)** or higher with a rear camera (recommended for testing live camera + TTS)

### Build & Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/jarvis1704/currency-detection.git
   cd currency-detection/android
   ```

2. **Run unit tests**
   ```bash
   ./gradlew testDebugUnitTest
   ```

3. **Build a debug APK**
   ```bash
   ./gradlew assembleDebug
   ```
   Or open the `android/` directory directly in **Android Studio** and click **Run**.

4. **Release Signing (Optional)**

   The release build enables R8 code minification and resource shrinking (`app/build.gradle.kts`). To sign a release build, add your keystore properties to `android/local.properties` (gitignored):
   ```properties
   release.storeFile=/path/to/keystore.jks
   release.storePassword=your_store_password
   release.keyAlias=your_key_alias
   release.keyPassword=your_key_password
   ```
   Then run:
   ```bash
   ./gradlew assembleRelease
   ```

---

## 🧠 Model Training Pipeline

All training code lives in [`model_training/`](model_training/) and uses **publicly available datasets only** (no personal phone photos), with strict deduplication (MD5 + 64-bit perceptual `dhash`) and external test-set isolation (`mendeley-8ckhkssyn3`).

### 1. Deployed Classifier (`MobileNetV3-Small`)

- **Notebook**: [`currency_training.ipynb`](model_training/currency_training.ipynb)
- **Model**: `MobileNetV3Small` (`include_preprocessing=True`, ImageNet weights), trained on 10 fine-grained classes (`10_new`, `10_old`, `100_new`, `100_old`, `20`, `200`, `50_new`, `50_old`, `500`, `background`) and summed into 7 coarse classes at export.
- **Export**: Plain **float32** LiteRT `.tflite` (selected over INT8 quantization to preserve accuracy and XNNPACK compatibility).
- **Android Hand-off Contract**:
  - Input: `[1, 224, 224, 3]` `float32` RGB in `0–255` (center square crop + bilinear resize, unrotated).
  - Output: `[1, 7]` `float32` probabilities matching `labels.txt` (`10`, `100`, `20`, `200`, `50`, `500`, `background`).
  - Place exported `currency_model.tflite` and `labels.txt` in `android/app/src/main/assets/`.

### 2. Multi-Note Detector *(In Progress — `feat/multi-detect`)*

- **Script / Notebook**: [`train_detector.py`](model_training/train_detector.py) & [`detector_training.ipynb`](model_training/detector_training.ipynb)
- Trains a **YOLO11n** (`416×416` float32 `.tflite`) detector across 6 banknote classes (`10, 20, 50, 100, 200, 500`) using OWLv2 (`google/owlv2-base-patch16-ensemble`) auto-boxing and synthetic multi-note scenes so a future update can count multiple overlapping banknotes and sum their total value.

#### Running the Detector Pipeline (Kaggle GPU)

```bash
python train_detector.py prepare          # Unpack inputs, download datasets, generate bg contact sheets
python train_detector.py reject bg 3 17   # Reject background candidates containing any part of a note
python train_detector.py autobox          # Auto-box single-note photos via OWLv2 + spot-check sheets
python train_detector.py reject box 5 9   # Reject inaccurate boxes
python train_detector.py synth            # Build deduplicated splits & multi-note composite scenes
python train_detector.py train            # Train YOLO11n (resumable)
python train_detector.py export           # Export float32 .tflite + detector_labels.txt
python train_detector.py evaluate         # Measure mAP, exact counts, totals & false-speech rates
```

---

## 📊 Datasets Used

Training and evaluation rely exclusively on public datasets:
- **IEEE DataPort**: *Dataset of Indian and Thai Banknotes* (Indian currency folders only; ₹2000 excluded as withdrawn)
- **Roboflow Universe**: *Indian currency/notes* (YOLOv8 export, CC BY 4.0)
- **Mendeley Data** (CC BY 4.0): `48ympv8jjf`, `jh6979fg2t` (spoilt notes), and `8ckhkssyn3` (held-out external test set)
- **Background Pools**: Roboflow note-free crops, **EgoHands** (Indiana University), **DTD** (Describable Textures Dataset), and **Imagenette**

---

## 🔒 Privacy & Permissions

| Permission | Purpose |
|---|---|
| `android.permission.CAMERA` | Live camera feed for on-device banknote recognition |
| `android.permission.VIBRATE` | 150 ms haptic confirmation when a denomination is recognized |
| `TTS_SERVICE` (`<queries>`) | Query local Android Text-to-Speech engines on Android 11+ |

No network permission is declared in [`AndroidManifest.xml`](android/app/src/main/AndroidManifest.xml). All processing stays strictly on the device.

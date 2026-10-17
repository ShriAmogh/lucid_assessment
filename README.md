# OCR Engine — Offline CPU-Optimized ONNX Document Extraction Pipeline

An offline-capable, CPU-optimized Optical Character Recognition (OCR) and structured Key-Value extraction pipeline built with **ONNX Runtime**, **FastAPI**, and **React**. Designed to extract identity fields from international documents (Greek, German, Romanian, and English) entirely on CPU without cloud egress or GPU hardware.

---

## 🌟 Key Highlights

- **100% Offline & Private:** Zero cloud APIs, zero external telemetry, zero GPU dependency.
- **Sub-Second CPU Inference:** End-to-end forward pass + parsing in **~300–650 ms** on standard multi-core CPUs.
- **Multilingual Support:**
  - **German:** Full umlauts (`ä, ö, ü, ß`), *Personalausweis*, and *Führerschein* (EU driving licences).
  - **Greek:** Full Greek script (`Α–Ω, α–ω, ά, έ, ή...`) + Latin transliteration (*Δελτίο Ταυτότητας*).
  - **Romanian:** Romanian diacritics (`â, î, ş, Ş`) and standard Latin fields (*Carte de Identitate*).
  - **English / UK:** UK Driving Licences and Passports.
- **2D Spatial Layout & Semantic Extraction:** Spatial column alignment, bilingual anchor parsing, and strict ICAO Doc 9303 MRZ verification.
- **Interactive Modern UI:** React dashboard with live bounding box SVG overlay, confidence badges, latency profiler, and 1-click test presets.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React (Vite) Frontend                           │
│  - 1-Click Evaluation Presets (German, Greek, Romanian, UK)            │
│  - Interactive Bounding Box Canvas Overlay                             │
│  - Key-Value Cards with Confidence Ratings & Copy                      │
│  - Micro-Latency Profiler Gauges & JSON Export                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP POST /api/extract
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend (CPU)                           │
│                                                                        │
│  1. Preprocessing Engine                                               │
│     - RGBA Alpha Flattening on White Background                        │
│     - Aspect-Ratio Preserved Scaling (H, W mod 32 == 0)                │
│     - Channel Normalization (ImageNet mean/std)                        │
│                                                                        │
│  2. DBNet Text Detection (ONNX Runtime CPU)                            │
│     - Probability map binarization & contour extraction                │
│     - Polygon expansion via ClipperLib (Vatti unclip)                  │
│     - Oriented Bounding Boxes (OBB) & Reading-Order Sorting            │
│                                                                        │
│  3. SVTR Multilingual Text Recognition (ONNX Runtime CPU)              │
│     - 4-point perspective warp for rotated/sheared text crops          │
│     - Batch execution across crops to minimize session overhead        │
│     - Argmax greedy CTC decoding with character dictionary mapping     │
│                                                                        │
│  4. Semantic Key-Value & MRZ Engine                                    │
│     - 2D Spatial Proximity & Column Alignment                          │
│     - Bilingual Label Token Matching (handles "SURNAME/NOM", etc.)     │
│     - Date normalization (ISO 8601: YYYY-MM-DD)                        │
│     - ICAO Doc 9303 MRZ Checksum Parser (TD1, TD2, TD3)                │
│                                                                        │
│  5. High-Resolution Benchmarker                                        │
│     - Nanosecond stage profiling: Preprocess, Det, Rec, Parse, Total   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Quantized / Optimized ONNX Models                      │
│  - models/det.onnx       (DBNet++, ~2.4 MB)                            │
│  - models/rec_latin.onnx (SVTR Latin, ~7.8 MB - DE, RO, EN)            │
│  - models/rec_greek.onnx (SVTR Greek, ~7.7 MB - GR, EN)                │
│  - models/dict_latin.txt & dict_greek.txt                              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ ONNX CPU Optimizations

To achieve sub-second inference on commodity CPUs:
1. **Lightweight Backbones:** Utilizes MobileNetV3/LCNet-based DBNet detection (~2.4 MB) and SVTR recognition (~7.8 MB) engineered for mobile/CPU execution.
2. **Execution Provider Configuration:**
   - Provider: `CPUExecutionProvider`
   - `intra_op_num_threads = 4` (dynamically utilizes available physical cores)
   - `inter_op_num_threads = 2`
   - `graph_optimization_level = ORT_ENABLE_ALL` (constant folding, node fusions, redundant transpose removal).
3. **Batch Cropping:** Cropped text regions are batched together by width to minimize kernel launch overhead and memory re-allocations.
4. **Zero-Copy Vectorization:** Image normalization and color space transformations utilize vectorized NumPy C-bindings.

---

## 📊 Benchmark Latency Breakdown (Average on CPU)

| Document | Preprocessing | DBNet Detection | SVTR Recognition | KV Parsing | **Total CPU Time** |
|---|---|---|---|---|---|
| **German ID 1** | ~18 ms | ~52 ms | ~275 ms | ~0.3 ms | **~345 ms** |
| **German ID 2 (Führerschein)** | ~22 ms | ~58 ms | ~570 ms (38 boxes) | ~0.5 ms | **~651 ms** |
| **Greek ID 1** | ~20 ms | ~52 ms | ~276 ms | ~0.2 ms | **~349 ms** |
| **Greek ID 2** | ~21 ms | ~55 ms | ~360 ms | ~0.3 ms | **~436 ms** |
| **Romanian ID 1** | ~17 ms | ~48 ms | ~215 ms | ~0.2 ms | **~281 ms** |
| **English / UK ID** | ~19 ms | ~51 ms | ~217 ms | ~0.2 ms | **~287 ms** |

---

## 🚀 Quickstart & Setup Instructions

### 1. Prerequisites
- **Python 3.10+ / 3.11**
- **Node.js 18+ & npm**

### 2. Backend Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd lucid_assignment

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download / verify ONNX models
python backend/app/utils/model_downloader.py

# 5. Run test suite
pytest

# 6. Start FastAPI backend server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
API Documentation will be available at: `http://127.0.0.1:8000/docs`

### 3. Frontend Setup

In a separate terminal window:
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install npm packages
npm install

# 3. Start React Vite dev server
npm run dev
```
Open your browser at: `http://localhost:3000`

---

## 🧪 Testing Strategy

The project follows a strict **Test-First development strategy** with 22 unit and integration tests covering all critical components:

```bash
# Run all tests with coverage report
pytest --cov=backend/app backend/tests/
```

- `test_preprocessor.py`: Verifies RGBA alpha flattening, aspect-ratio preserved scaling (32-px alignment), and 4-point perspective warp.
- `test_geometry.py`: Tests clockwise corner ordering, ClipperLib polygon unclipping, and reading order sorting.
- `test_detector.py`: Tests DBNet ONNX inference and bounding box score filtering.
- `test_recognizer.py`: Tests CTC greedy argmax decoding and multilingual dictionary mapping.
- `test_kv_extractor.py`: Tests semantic field extraction across Greek, German, Romanian, and English layouts.
- `test_engine.py`: Tests end-to-end pipeline execution on real international ID card images.
- `test_api.py`: Tests FastAPI REST endpoints (`/api/health`, `/api/samples`, `/api/extract`).

---

## 📁 Repository Structure

```
lucid_assignment/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI application & API endpoints
│   │   ├── config.py             # Global configuration, model paths, thresholds
│   │   ├── pipeline/
│   │   │   ├── preprocessor.py   # RGBA alpha compositing & normalization
│   │   │   ├── detector.py       # ONNX DBNet text detection
│   │   │   ├── recognizer.py     # ONNX SVTR text recognition & CTC decoder
│   │   │   ├── mrz_parser.py     # TD1, TD2, TD3 MRZ parser with check-digits
│   │   │   ├── kv_extractor.py   # 2D Spatial anchor-to-value resolver
│   │   │   ├── benchmarker.py    # Sub-millisecond latency timer
│   │   │   └── ocr_engine.py     # End-to-end pipeline coordinator
│   │   ├── models/
│   │   │   └── schemas.py        # Pydantic schemas (ExtractionResponse, etc.)
│   │   └── utils/
│   │       ├── geometry.py       # Polygon unclip & perspective warp
│   │       ├── logger.py         # Structured logging
│   │       └── model_downloader.py # Automated ONNX weights downloader
│   └── tests/                    # Complete pytest test suite
├── frontend/                     # React (Vite) UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx        # Navbar & CPU status badge
│   │   │   ├── SamplePicker.jsx  # 1-Click evaluation presets
│   │   │   ├── BBoxCanvas.jsx    # SVG Bounding Box overlay with tooltips
│   │   │   ├── KeyValueCards.jsx # Extracted fields & copy buttons
│   │   │   ├── LatencyGauge.jsx  # Micro-latency breakdown
│   │   │   └── JsonViewer.jsx    # JSON export & viewer
│   │   ├── App.jsx               # Main React dashboard
│   │   └── index.css             # High-contrast precision dark slate design system
│   ├── package.json
│   └── vite.config.js
├── models/                       # Downloaded ONNX models
│   ├── det.onnx                  # DBNet detection head (~2.3 MB)
│   ├── rec_latin.onnx            # SVTR Latin recognition head (~7.5 MB)
│   ├── rec_greek.onnx            # SVTR Greek recognition head (~7.4 MB)
│   ├── dict_latin.txt            # Latin character dictionary (DE, RO, EN)
│   └── dict_greek.txt            # Greek character dictionary
├── samples/                      # Assessment Deliverables
│   ├── input/                    # Tested sample images (Greek, German, Romanian, UK)
│   ├── json/                     # Extracted JSON results per sample
│   └── output/                   # UI output screenshots (8 verified presets)
├── requirements.txt              # Backend Python dependencies
├── project.md                    # Project requirements & deliverables tracking
└── README.md                     # This file
```

---

## 📜 License & Compliance

Built for the **LucidPlus Candidate Assessment**. Free to evaluate. Models are distributed under Apache 2.0 / open-source licenses.

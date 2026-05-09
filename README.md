# 🏏 Automatic Cricket Highlight Generation

**EE655 Course Project · IIT Kanpur · Group 34**

A modular, interpretable pipeline for automatically generating cricket highlights from full-length cricket match broadcasts.

> **85.7% F1-score** on a full ENG vs. IND T20 match (93.4% precision · 79.2% recall) across Sixes, Fours, and Wickets.

---

## Overview

Cricket matches run 3–8 hours. This system fuses three complementary modalities — scoreboard OCR, audio excitement analysis, and LLM-based commentary understanding — into a layered **Sensor → Brain → Scalpel** architecture that automatically extracts and assembles highlight clips.

```
Match Video
    │
    ├── Sensor 0 — HSV Scene-Change Detection
    ├── Sensor 1 — YOLOv8 Scoreboard OCR  (Fours / Sixes / Wickets)
    ├── Sensor 2 — Audio Excitement Analysis (Robust Z-Score)
    │
    ├── Brain   — Targeted Gemini Flash/Pro LLM Verification
    │
    └── Scalpel — Audio-Anchored, Scene-Snapped FFmpeg Clip Extraction
                        │
                  Highlight Reel (chronological)
```

---

## Results

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Sixes    | 90.9%     | 95.2%  | 93.0%    |
| Fours    | 100.0%    | 74.3%  | 85.2%    |
| Wickets  | 84.6%     | 68.8%  | 75.9%    |
| **Overall** | **93.4%** | **79.2%** | **85.7%** |

Evaluated on a full 3.5-hour ENG vs. IND T20 broadcast. Mean temporal alignment error: **5.8 s** (54/57 matched events within 15 s of ground truth).

---

## Pipeline Details

### Sensor 0 — Scene-Change Detection
- Downscales each frame to 320×240 and computes a joint HSV histogram (50×60 bins)
- Flags timestamps where the L2 distance between consecutive normalised histograms exceeds 0.5
- Raw detections within 0.2 s are merged; results feed the Scalpel layer for clip-boundary snapping

### Sensor 1 — Scoreboard OCR
- **YOLOv8** detector trained on a custom Roboflow dataset; restricted to the bottom 25% of the frame
- Cropped ROI is upscaled (min dimension → 500 px) and **Otsu-binarised** before EasyOCR
- A **persistence filter (P=3)** requires three consecutive identical readings before accepting a score state
- Rule-based validator maps `(Δruns, Δwickets)` deltas to `{Four, Six, Wicket, New Innings}` labels

### Sensor 2 — Audio Excitement Analysis
- Streams audio (mono 16 kHz PCM) with a 1.25 s window / 0.25 s hop
- Computes five features: dB-RMS, crest factor, spectral flux, spectral centroid, Δ-loudness
- Uses a **robust z-score (median/MAD)** to handle cricket audio's heavy-tailed distribution
- Detects both instantaneous peaks (`prominence ≥ 0.8`) and sustained loudness regions (`≥ 2 s`)

### Brain — LLM Verification
- Constructs **targeted Regions of Interest** (ROIs) from OCR events (`[t−25s, t+5s]`) and keyword triggers (`review`, `no ball`, `dropped`, `appeal`, ...)
- **Two-stage verification:** Gemini Flash screens each ROI (YES/NO); Gemini Pro extracts structured `SemanticEvent` objects (with Pydantic schema enforcement) only for positive ROIs
- Replay-suppression deduplicator discards same-type events within 45 s of a prior accepted event

### Scalpel — Clip Extraction
- Each verified event is anchored to the nearest audio peak within a confidence-dependent window
- **Event-type-specific padding:**

  | Event Type   | Pre (s) | Post (s) |
  |--------------|---------|----------|
  | Wicket       | 9       | 22       |
  | Six          | 6       | 14       |
  | Four         | 5       | 10       |
  | DRS Review   | 5       | 75       |

- Clip boundaries are snapped to stable HSV scene-change shots (≥ 2 s stability)
- Extracted via FFmpeg re-encoding with `-avoid_negative_ts make_zero -af aresample=async=1000` to eliminate audio desync

---

## Repository Structure

This repository contains the code, documentation, and research notebooks for the automatic cricket highlight generation pipeline.

```text
├── sample_data/                                                 # Pre-extracted transcripts, metadata, and timestamps for pipeline testing
├── YOLO+OCR ONLY Approach/
│   └── (Directory containing isolated scripts and experiments that rely strictly on YOLO object detection and scoreboard OCR)
├── FINAL_INTEGRATED_CODE(WITH TRANSCRIPT GENERATION).ipynb      # Full "Verify then Anchor" pipeline utilising both scoreboard OCR as the source of truth and audio analysis/transcripts
├── FINAL_INTEGRATED_CODE(WITHOUT TRANSCRIPT GENERATION).ipynb   # Integrated pipeline utilising visual features and scoreboard OCR, excluding audio transcription analysis
├── Most Replayed Graph - Approach.ipynb                         # Notebook analysing viewer retention/replay graphs to identify potential highlight peaks
├── optical-flow-implementation.ipynb                            # Implementation and exploration of optical flow techniques for motion tracking in cricket footage
├── Presentation Automatic Cricket Highlight Generation          # Slide deck outlining the project methodology, pipeline architecture, and final results
├── Report_Automatic_Cricket_Highlight_Generation                # Comprehensive project report detailing the theoretical background and implementation
└── README.md                                                    # Project overview, setup instructions, and repository guide
```

## Datasets & Resources

| Resource | Source |
|----------|--------|
| Match video (ENG vs. IND, ENG vs. PAK) | Kaggle: `yashkumar008/t20-full-match-dataset` |
| YOLO scoreboard weights | Kaggle: `loak2055/yolo-weights` |
| Whisper transcript | Kaggle: `kuldipmanvar/transcript-indvseng` |
| Scoreboard training data | Roboflow (link in `data/data.yaml`) |

> The match recordings are not redistributed in this repository for copyright reasons.

---

## Setup & Reproducibility

### Requirements

```bash
pip install ultralytics easyocr faster-whisper google-genai pydantic \
            soundfile scipy opencv-python numpy pandas matplotlib tqdm
```

FFmpeg must be available on `PATH`.

### API Keys

Store your Gemini API key as a Kaggle Secret named `GEMINI_API_KEY`. The pipeline reads it via:

```python
from kaggle_secrets import UserSecretsClient
GEMINI_API_KEY = UserSecretsClient().get_secret("GEMINI_API_KEY")
```

### GPU Memory Notes

The T4's 16 GB VRAM cannot hold Whisper large-v3, YOLOv8, and EasyOCR simultaneously. The pipeline runs Whisper first, then explicitly frees it before loading the vision models:

```python
del model
torch.cuda.empty_cache()
```

If a pre-existing transcript is available (recommended), skip the Whisper step entirely — this saves ~45 minutes of runtime.

### LLM Non-Determinism

Gemini Flash screening runs at `temperature=0.0`; Gemini Pro extraction at `temperature=0.1`. For strict reproducibility, set both to `0`, though slightly lower candidate coverage may result.

---

## Explored Alternatives

Three alternative approaches were implemented, evaluated, and ultimately not adopted as the primary method:

**YouTube "Most-Replayed" SVG Heatmap** — Analytically inverts the cubic Bézier heatmap path to extract crowdsourced highlight peaks. Works elegantly on viral content but has 15.4% recall (F1: 25.2%) due to the fixed top-10 cutoff, and is unavailable for fresh uploads.

**Farneback Dense Optical Flow** — Used as a fallback within the OCR pipeline when no score change is detected. Fires on genuine high-motion moments but also on replays and camera pans, and reduces processing speed by ~10× on a T4.

**Scoreboard-Only Baseline** — OCR in isolation achieves 92.9% recall for trackable events but 60.9% precision (F1: 73.6%), and misses DRS reviews, dropped catches, and milestone celebrations entirely.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Object Detection | `ultralytics` (YOLOv8) |
| OCR | `EasyOCR` |
| ASR | `faster-whisper` (large-v3) |
| Audio | `soundfile`, `scipy` |
| Vision | `opencv-python` |
| LLM | `google-genai` (Gemini 2.5 Flash / Pro) |
| Schema | `pydantic` |
| Video | `FFmpeg` |
| Platform | Kaggle (NVIDIA T4, Python 3.10) |

---

## Team

| Name | Roll No. | GitHub |
|------|----------|--------|
| Kuldip Manvar | 240575 | [@kuldip-max](https://github.com/kuldip-max) |
| Yash Pandit | 231183 | [@hsaycode](https://github.com/hsaycode) |
| Aryan Kadam | 230221 | [@loak20](https://github.com/loak20) |
| Yash Kumar | 231186 | [@kyash23](https://github.com/kyash23) |
| Sanjay Jangir | 230913 |  |

---

## Citation

If you use this work, please cite:

```
Kuldip Manvar, Yash Pandit, Aryan Kadam, Yash Kumar, and Sanjay Jangir.
"Automatic Cricket Highlight Generation via Multimodal Fusion of Scoreboard OCR,
Audio Excitement Cues, and LLM-based Commentary Understanding."
EE655 Course Project, IIT Kanpur, 2026.
```

---

## Acknowledgements

We thank course instructor Prof. Koteswar Rao Jerripothula at the Department of Electrical Engineering, IIT Kanpur, for his guidance, and the open-source maintainers of `ultralytics`, `EasyOCR`, `faster-whisper`, `scipy`, and `ffmpeg`.

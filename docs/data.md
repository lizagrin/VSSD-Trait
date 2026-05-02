# Dataset

## First Impressions V2

| Property             | Value                                                          |
| -------------------- | -------------------------------------------------------------- |
| Source               | ChaLearn LAP, ECCV 2016 / ECCV 2018 challenges                  |
| Modalities           | RGB video, audio (44.1 kHz, downsampled to 16 kHz), transcripts |
| Clips                | 10 000 (6 000 train / 2 000 val / 2 000 test)                  |
| Clip length          | 15 seconds (≈ 375 frames @ 25 FPS)                             |
| Targets              | Big Five soft scores in [0, 1]                                 |
| License              | Research-only (CC BY-NC-SA-like, ChaLearn agreement)            |

The original release ships with three pickle files per split:

* `annotation_<split>.pkl` — soft Big Five scores per video, plus an
  ``interview`` rating;
* `transcription_<split>.pkl` — speaker transcripts (single string per video);
* the videos themselves as `*.mp4`.

## On-disk layout expected by the project

```
$DATA_ROOT/
├── TRAIN/
│   ├── *.mp4
│   └── Annotation/
│       ├── annotation_training.pkl
│       └── transcription_training.pkl
├── VALIDATION/
│   ├── *.mp4
│   └── Annotation/
│       ├── annotation_validation.pkl
│       └── transcription_validation.pkl
└── TEST/
    ├── *.mp4
    └── Annotation/
        ├── annotation_test.pkl
        └── transcription_test.pkl
```

`av_traits.data.manifests.build_manifest_from_local` walks this tree
and produces a `pandas.DataFrame` with one row per clip:

```
split | video_id | video_path | transcription | openness | conscientiousness | extraversion | agreeableness | neuroticism
```

## Feature cache

Each clip is preprocessed once and the results are dumped to a single
`*.pt` file under `$CACHE_ROOT/video/<split>/<video_id>.pt` containing:

| Key            | dtype       | Shape                       | Description                                  |
| -------------- | ----------- | --------------------------- | -------------------------------------------- |
| `faces`        | uint8       | `[W, K, H, W, 3]`           | MTCNN-cropped 224×224 faces (RGB).           |
| `face_scores`  | float32     | `[W, K]`                    | Confidence per face crop (0/1 = MTCNN/center fallback). |
| `mel`          | float32     | `[128, T]`                  | log-mel spectrogram (z-score normalised).    |
| `prosody`      | float32     | `[38]`                      | rms / zcr / f0 statistics + mean/std MFCC.   |
| `ege_maps`     | float32 \| None | `[88]`                  | eGeMAPSv02 functionals (None if openSMILE failed). |
| `target`       | float32     | `[5]`                       | Soft Big Five scores in [0, 1].              |
| `transcription`| str         | —                           | Full transcript string.                      |

W = `cfg.train_windows` / `cfg.eval_windows`; K = frames per window.

Run-once command:

```bash
python scripts/precache_features.py \
    --data-root $DATA_ROOT --cache-root $CACHE_ROOT
```

The script also extracts a 16 kHz mono `*.wav` per clip into
`$CACHE_ROOT/audio/`, used by `librosa` and `openSMILE` so that audio
decoding happens at most once.

## Splits used by experiments

* All training is on the **train** split.
* Validation metrics drive model selection (best-by `ccc + 0.25·macc`).
* All headline numbers (Tables 1–11) are reported on the **test** split.

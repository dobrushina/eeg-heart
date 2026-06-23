"""eeg_psv_bhi_prep.py — Prepare MATLAB inputs for compute_psv_sdg (PSV-BHI).

This script does NOT run MATLAB. It prepares one MAT bundle per
subject-condition with the exact variables expected by compute_psv_sdg.m:

    data_eeg   — FieldTrip raw struct (trial, time, label, fsample)
    pks_indx   — 1-based integer R-peak sample indices within the segment
    file_output — target path for MATLAB to write the result

Key differences from eeg_bhi_prep.py (SDGM):
- data_eeg is FieldTrip format, not EEGLAB
- No RR/RRi/t_RRi: compute_psv_sdg derives IBI from pks_indx internally
- No FS_bhi or TV parameters
- EEG frequency bands are fixed inside compute_psv_sdg (delta/theta/alpha/beta/gamma)
- Requires FieldTrip on MATLAB path at runtime
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import mne
from scipy.io import savemat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from eeg_config import DERIVATIVES_DIR, FINAL_PIPELINE_DIR
from eeg_common import (
    crop_to_state_annotations,
    load_condition_segments,
    load_rpeaks_csv,
    rename_channels_drop_reference,
    set_channel_types_from_names,
)


# ---------------------------------------------------------------------------
# ── Simple toggles ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

PROCESS_SELECTED_SUBJECTS: bool = False
SELECTED_SUBJECTS: List[str] = ["sub-eeg-1"]

CONDITIONS: List[str] = ["EC"]

# When a condition has multiple segments, use the single longest one.
# This keeps the signal continuous, which is required by compute_psv_sdg.
USE_LONGEST_SEGMENT_ONLY: bool = True

# RR outlier cleaning (for validation only — the MAT contains pks_indx, not RR)
RR_MIN_SEC: float = 0.30
RR_MAX_SEC: float = 2.00
MIN_PEAKS_REQUIRED: int = 10

# Output locations
PSV_PIPELINE_DIR = "eeg-psv-bhi"
PSV_INPUT_DIR = ROOT / DERIVATIVES_DIR / PSV_PIPELINE_DIR / "inputs"
PSV_OUTPUT_DIR = ROOT / DERIVATIVES_DIR / PSV_PIPELINE_DIR / "outputs"
PSV_GROUP_DIR = ROOT / DERIVATIVES_DIR / "group" / PSV_PIPELINE_DIR
CONDITIONS_CSV = ROOT / "manual" / "conditions.csv"


# ---------------------------------------------------------------------------
# ── Helpers ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _discover_subjects(final_dir: Path) -> List[str]:
    if not final_dir.exists():
        return []
    subjects = sorted(
        p.name for p in final_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-eeg-")
    )
    if PROCESS_SELECTED_SUBJECTS:
        selected = set(SELECTED_SUBJECTS)
        subjects = [s for s in subjects if s in selected]
    return subjects


def _find_state_record_onset(raw: mne.io.BaseRaw) -> float | None:
    for ann in raw.annotations:
        if ann["description"].strip().lower() == "state record":
            return float(ann["onset"])
    return None


def _relative_segments_for_raw(
    raw: mne.io.BaseRaw,
    segments_abs: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Shift absolute condition windows to the cropped raw time axis."""
    record_onset = _find_state_record_onset(raw)
    if record_onset is None:
        return [(float(s), float(e)) for s, e in segments_abs]

    tmax = float(raw.times[-1])
    rel: List[Tuple[float, float]] = []
    for start, end in segments_abs:
        s = max(0.0, float(start) - record_onset)
        e = min(tmax, float(end) - record_onset)
        if e > s:
            rel.append((s, e))
    return rel


def _make_fieldtrip_struct(
    eeg_data_uv: np.ndarray,
    eeg_names: Sequence[str],
    sfreq: float,
) -> dict:
    """Build a minimal FieldTrip raw data struct for ft_freqanalysis.

    MATLAB representation after loadmat:
      data_eeg.trial{1}   — double(n_channels x n_samples)
      data_eeg.time{1}    — double(1 x n_samples), starting at 0
      data_eeg.label      — cell(n_channels x 1)
      data_eeg.fsample    — double scalar
    """
    n_ch, n_samp = eeg_data_uv.shape

    # FieldTrip time vector starts at 0
    time_vec = np.arange(n_samp, dtype=float) / sfreq  # (n_samp,)

    # MATLAB cell arrays are stored as numpy object arrays in scipy savemat
    trial_cell = np.empty((1, 1), dtype=object)
    trial_cell[0, 0] = eeg_data_uv.astype(np.float64)

    time_cell = np.empty((1, 1), dtype=object)
    time_cell[0, 0] = time_vec.reshape(1, -1).astype(np.float64)  # (1 x n_samp)

    label_cell = np.empty((n_ch, 1), dtype=object)
    for i, ch in enumerate(eeg_names):
        label_cell[i, 0] = str(ch)

    return {
        "trial": trial_cell,
        "time": time_cell,
        "label": label_cell,
        "fsample": float(sfreq),
    }


def _get_pks_indx(
    rpeaks_sec: np.ndarray,
    start_sec: float,
    end_sec: float,
    sfreq: float,
    n_samples: int,
) -> np.ndarray:
    """Compute 1-based MATLAB sample indices of R-peaks within a segment.

    Parameters
    ----------
    rpeaks_sec : np.ndarray
        All R-peak times in seconds (full recording reference).
    start_sec, end_sec : float
        Segment boundaries in the raw time axis after cropping.
    sfreq : float
        EEG sampling frequency.
    n_samples : int
        Total number of samples in the exported EEG segment.

    Returns
    -------
    np.ndarray of int
        1-based sample indices, clamped to [1, n_samples].
    """
    mask = (rpeaks_sec >= start_sec) & (rpeaks_sec < end_sec)
    rpeaks_in_seg = rpeaks_sec[mask]

    if rpeaks_in_seg.size == 0:
        return np.array([], dtype=np.int64)

    # Make relative to segment start, then convert to 1-based index.
    rpeaks_rel = rpeaks_in_seg - start_sec
    idx = np.round(rpeaks_rel * sfreq).astype(np.int64) + 1  # 1-based

    # Clamp to valid range
    idx = idx[(idx >= 1) & (idx <= n_samples)]
    return idx


def _validate_peaks(pks_indx: np.ndarray, sfreq: float) -> dict:
    """Quick QC on peak intervals."""
    if pks_indx.size < 2:
        return {"n_peaks": int(pks_indx.size), "n_rr": 0, "ok": False}

    rr_sec = np.diff(pks_indx.astype(float)) / sfreq
    valid = (rr_sec >= RR_MIN_SEC) & (rr_sec <= RR_MAX_SEC)
    return {
        "n_peaks": int(pks_indx.size),
        "n_rr": int(rr_sec.size),
        "n_rr_valid": int(valid.sum()),
        "mean_rr_sec": float(np.mean(rr_sec[valid])) if valid.any() else None,
        "ok": int(valid.sum()) >= MIN_PEAKS_REQUIRED,
    }


def _prepare_subject_condition(subject: str, condition: str) -> dict:
    subj_dir = ROOT / DERIVATIVES_DIR / FINAL_PIPELINE_DIR / subject
    edf_path = subj_dir / f"{subject}_desc-final_denoised_eeg.edf"
    rpeaks_path = subj_dir / f"{subject}_desc-rpeaks_events.csv"

    for p, label in [(edf_path, "final EDF"), (rpeaks_path, "R-peaks CSV"), (CONDITIONS_CSV, "conditions CSV")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    rename_channels_drop_reference(raw, verbose=False)
    set_channel_types_from_names(raw, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore")

    raw, cropped = crop_to_state_annotations(raw)
    if cropped:
        print(f"  Cropped to State record/stop: {raw.times[-1]:.1f} s remaining")

    rpeaks_sec = load_rpeaks_csv(rpeaks_path)
    segments_abs = load_condition_segments(CONDITIONS_CSV, subject, condition)
    if not segments_abs:
        raise ValueError(f"No condition segments found for {subject} / {condition}")

    segments = _relative_segments_for_raw(raw, segments_abs)
    if not segments:
        raise ValueError(f"No usable segments remain after cropping for {subject} / {condition}")

    # Select the single longest segment.
    start_sec, end_sec = max(segments, key=lambda x: x[1] - x[0])
    print(f"  Using longest segment: {start_sec:.1f}–{end_sec:.1f} s "
          f"({end_sec - start_sec:.1f} s, "
          f"from {len(segments)} interval(s))")

    eeg_picks = mne.pick_types(raw.info, eeg=True, ecg=False, eog=False, misc=False)
    eeg_names = [raw.ch_names[p] for p in eeg_picks]
    sfreq = float(raw.info["sfreq"])

    seg = raw.copy().crop(tmin=start_sec, tmax=end_sec, include_tmax=False)
    eeg_data_uv = seg.get_data(picks=eeg_picks) * 1e6  # (n_ch x n_samples)
    n_samples = int(eeg_data_uv.shape[1])

    pks_indx = _get_pks_indx(rpeaks_sec, start_sec, end_sec, sfreq, n_samples)
    qc = _validate_peaks(pks_indx, sfreq)

    if not qc["ok"]:
        raise ValueError(
            f"Too few valid RR intervals for {subject} / {condition}: "
            f"{qc.get('n_rr_valid', 0)} valid peaks"
        )

    data_eeg = _make_fieldtrip_struct(eeg_data_uv, eeg_names, sfreq)

    file_output = PSV_OUTPUT_DIR / subject / f"{subject}_desc-psv-bhi-{condition}_result.mat"
    input_mat = PSV_INPUT_DIR / subject / f"{subject}_desc-psv-bhi-{condition}_input.mat"
    meta_json = PSV_INPUT_DIR / subject / f"{subject}_desc-psv-bhi-{condition}_input.json"

    payload = {
        "data_eeg": data_eeg,
        "pks_indx": pks_indx.reshape(1, -1).astype(np.float64),  # row vector for MATLAB
        "file_output": str(file_output),
    }

    meta = {
        "subject": subject,
        "condition": condition,
        "source_edf": str(edf_path),
        "source_rpeaks_csv": str(rpeaks_path),
        "segments_absolute": [list(x) for x in segments_abs],
        "segments_relative": [list(x) for x in segments],
        "selected_segment": [float(start_sec), float(end_sec)],
        "segment_policy": "longest",
        "n_channels": int(eeg_data_uv.shape[0]),
        "n_samples": n_samples,
        "sfreq": sfreq,
        "n_peaks": qc["n_peaks"],
        "n_rr_valid": qc.get("n_rr_valid"),
        "file_output": str(file_output),
        "input_mat": str(input_mat),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return {
        "payload": payload,
        "meta": meta,
        "input_mat": input_mat,
        "meta_json": meta_json,
        "file_output": file_output,
    }


def _write_bundle(result: dict) -> None:
    input_mat: Path = result["input_mat"]
    meta_json: Path = result["meta_json"]

    input_mat.parent.mkdir(parents=True, exist_ok=True)
    PSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    savemat(str(input_mat), result["payload"], do_compression=True)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(result["meta"], f, indent=2, ensure_ascii=False)

    print(f"  Saved MAT  → {input_mat}")
    print(f"  Saved JSON → {meta_json}")
    print(f"  MATLAB writes to → {result['file_output']}")


def main() -> int:
    final_dir = ROOT / DERIVATIVES_DIR / FINAL_PIPELINE_DIR
    subjects = _discover_subjects(final_dir)
    if not subjects:
        print(f"ERROR: no subject folders found under {final_dir}")
        return 1

    PSV_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PSV_GROUP_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "subjects": subjects,
        "conditions": CONDITIONS,
        "segment_policy": "longest",
        "items": [],
    }

    for subject in subjects:
        for condition in CONDITIONS:
            print(f"\n{'='*60}")
            print(f"Preparing PSV-BHI input — {subject} / {condition}")
            print(f"{'='*60}")
            try:
                result = _prepare_subject_condition(subject, condition)
                _write_bundle(result)
                summary["items"].append(result["meta"])
            except Exception as exc:
                print(f"  ERROR: {exc}")
                summary["items"].append({
                    "subject": subject,
                    "condition": condition,
                    "error": str(exc),
                })

    summary_path = PSV_GROUP_DIR / "group_desc-psv-bhi_prep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary → {summary_path}")

    ok = sum(1 for x in summary["items"] if "error" not in x)
    err = sum(1 for x in summary["items"] if "error" in x)
    print(f"Done: {ok} ok, {err} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""eeg_bhi_prep.py — Prepare MATLAB inputs for SDGM_LFHF.

This script does not run MATLAB. It prepares one MAT bundle per
subject-condition (EC/EO) with the exact variables expected by
SDGM_LFHF.m:

    EEG, f_lims, RR, RRi, t_RRi, FS_rri, FS_bhi, TV, file_output

Default behavior is intentionally simple:
- process one subject or a small selected list
- process one condition or a small selected list
- if a condition has multiple segments, use the single longest segment

Notes
-----
- EEG data are exported in microvolts, matching EEGLAB-style conventions.
- RR intervals are exported in seconds.
- Because SDGM_LFHF uses all EEG channels, all EEG channels are retained.
- Non-EEG channels (e.g. ECG) are excluded from the EEG matrix.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

# Set True to only process SELECTED_SUBJECTS.
PROCESS_SELECTED_SUBJECTS: bool = False
SELECTED_SUBJECTS: List[str] = ["sub-eeg-1"]

# Process one or more conditions. For the paper workflow, keep EC and EO
# separate and run them independently.
CONDITIONS: List[str] = ["EC"]

# Segment policy: when a condition has multiple intervals, use the single
# longest interval only (instead of concatenation).
USE_LONGEST_SEGMENT_ONLY: bool = True

# EEG bands passed to SDGM_LFHF as f_lims (editable if needed).
# Default uses a standard theta/alpha/beta split.
F_LIMS = np.array([
    [4.0, 8.0],
    [8.0, 12.0],
    [12.0, 30.0],
], dtype=float)

# SDGM_LFHF parameters
FS_RRI: float = 4.0
FS_BHI: float = 8.0
TV: int = 1

# Output locations
BHI_INPUT_DIR = ROOT / DERIVATIVES_DIR / "eeg-bhi" / "inputs"
BHI_OUTPUT_DIR = ROOT / DERIVATIVES_DIR / "eeg-bhi" / "outputs"
BHI_GROUP_DIR = ROOT / DERIVATIVES_DIR / "group" / "eeg-bhi"
CONDITIONS_CSV = ROOT / "manual" / "conditions.csv"

# Internal quality settings
MIN_RR_INTERVALS = 10


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
        desc = ann["description"].strip().lower()
        if desc == "state record":
            return float(ann["onset"])
    return None


def _relative_segments_for_raw(
    raw: mne.io.BaseRaw,
    segments_abs: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Convert absolute condition windows to the current raw time axis.

    If the raw object has been cropped to 'State record'/'State stop', the
    returned windows are shifted so that they align with the cropped time axis.
    If no crop is present, the windows are returned unchanged.
    """
    record_onset = _find_state_record_onset(raw)
    if record_onset is None:
        return [(float(s), float(e)) for s, e in segments_abs]

    rel: List[Tuple[float, float]] = []
    tmax = float(raw.times[-1])
    for start, end in segments_abs:
        s = max(0.0, float(start) - record_onset)
        e = min(tmax, float(end) - record_onset)
        if e > s:
            rel.append((s, e))
    return rel


def _make_chanlocs(raw: mne.io.BaseRaw, eeg_names: Sequence[str]) -> List[dict]:
    """Create EEGLAB-like chanlocs records from the montage, if available."""
    montage = raw.get_montage()
    positions = {}
    if montage is not None:
        try:
            positions = montage.get_positions().get("ch_pos", {}) or {}
        except Exception:
            positions = {}

    chanlocs: List[dict] = []
    for ch in eeg_names:
        pos = positions.get(ch)
        if pos is None:
            x = y = z = np.nan
        else:
            x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
        chanlocs.append(
            {
                "labels": ch,
                "X": x,
                "Y": y,
                "Z": z,
                "type": "EEG",
            }
        )
    return chanlocs


def _segment_to_arrays(
    raw: mne.io.BaseRaw,
    start_sec: float,
    end_sec: float,
    eeg_picks: np.ndarray,
    rpeaks_sec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract one condition segment as EEG + RR arrays.

    Returns
    -------
    eeg_uv : np.ndarray
        Shape (n_channels, n_samples), in microvolts.
    rr_sec : np.ndarray
        Clean RR intervals within this segment, in seconds.
    rr_mid_sec : np.ndarray
        Time stamps for each RR interval (midpoints), relative to the segment
        start, in seconds.
    event_dict : dict
        Boundary/event metadata for the segment.
    """
    seg = raw.copy().crop(tmin=start_sec, tmax=end_sec, include_tmax=False)
    eeg_uv = seg.get_data(picks=eeg_picks) * 1e6

    rpeaks_in_seg = rpeaks_sec[(rpeaks_sec >= start_sec) & (rpeaks_sec < end_sec)]
    rpeaks_rel = rpeaks_in_seg - start_sec

    if rpeaks_rel.size < 2:
        rr_sec = np.array([], dtype=float)
        rr_mid_sec = np.array([], dtype=float)
    else:
        rr_sec = np.diff(rpeaks_rel)
        rr_mid_sec = (rpeaks_rel[:-1] + rpeaks_rel[1:]) / 2.0

    event_dict = {
        "type": "segment_start",
        "condition": "",
        "latency": 1,
        "duration": float(end_sec - start_sec),
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
    }
    return eeg_uv, rr_sec, rr_mid_sec, event_dict


def _clean_rr(rr_sec: np.ndarray, rr_mid_sec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove obvious RR outliers while preserving RR↔time alignment."""
    rr_sec = np.asarray(rr_sec, dtype=float)
    rr_mid_sec = np.asarray(rr_mid_sec, dtype=float)

    if rr_sec.size == 0 or rr_mid_sec.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Keep arrays strictly aligned.
    n = min(rr_sec.size, rr_mid_sec.size)
    rr_sec = rr_sec[:n]
    rr_mid_sec = rr_mid_sec[:n]

    # Finite values only.
    keep = np.isfinite(rr_sec) & np.isfinite(rr_mid_sec)
    rr_sec = rr_sec[keep]
    rr_mid_sec = rr_mid_sec[keep]
    if rr_sec.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Broad physiological bounds for artifact removal.
    keep = (rr_sec >= 0.30) & (rr_sec <= 2.00)
    rr_sec = rr_sec[keep]
    rr_mid_sec = rr_mid_sec[keep]
    if rr_sec.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Robust local outlier trimming using MAD when enough samples exist.
    if rr_sec.size >= 5:
        med = float(np.median(rr_sec))
        mad = float(np.median(np.abs(rr_sec - med)))
        if mad > 0:
            robust_z = 0.6745 * np.abs(rr_sec - med) / mad
            keep = robust_z <= 4.0
            rr_sec = rr_sec[keep]
            rr_mid_sec = rr_mid_sec[keep]

    return rr_sec, rr_mid_sec


def _interpolate_rr(
    rr_sec: np.ndarray,
    rr_mid_sec: np.ndarray,
    fs_rri: float,
    total_duration_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate RR intervals onto a uniform time grid."""
    rr_sec = np.asarray(rr_sec, dtype=float)
    rr_mid_sec = np.asarray(rr_mid_sec, dtype=float)

    if rr_sec.size == 0 or rr_mid_sec.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    start_t = 0.0
    end_t = float(total_duration_sec)
    if end_t <= start_t:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_rri = np.arange(start_t, end_t, 1.0 / fs_rri)
    if t_rri.size == 0 or t_rri[-1] < end_t:
        t_rri = np.append(t_rri, end_t)
    if t_rri.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Use edge values for extrapolation so SDGM_LFHF does not encounter NaNs.
    rri = np.interp(t_rri, rr_mid_sec, rr_sec, left=rr_sec[0], right=rr_sec[-1])
    return rri.astype(float), t_rri.astype(float)


def _make_eeg_struct(
    raw: mne.io.BaseRaw,
    eeg_data_uv: np.ndarray,
    eeg_names: Sequence[str],
    event_parts: Sequence[dict],
    subject: str,
    condition: str,
) -> dict:
    """Build an EEGLAB-like EEG struct for MATLAB."""
    sfreq = float(raw.info["sfreq"])
    chanlocs = _make_chanlocs(raw, eeg_names)

    eeg = {
        "setname": f"{subject}_{condition}",
        "filename": f"{subject}_desc-final_denoised_eeg.edf",
        "filepath": str((ROOT / DERIVATIVES_DIR / FINAL_PIPELINE_DIR / subject).resolve()),
        "data": eeg_data_uv.astype(float),
        "srate": sfreq,
        "nbchan": int(len(eeg_names)),
        "pnts": int(eeg_data_uv.shape[1]),
        "trials": 1,
        "xmin": 0.0,
        "xmax": float((eeg_data_uv.shape[1] - 1) / sfreq) if eeg_data_uv.shape[1] > 0 else 0.0,
        "chanlocs": chanlocs,
        "event": list(event_parts),
    }
    return eeg


def _prepare_subject_condition(
    subject: str,
    condition: str,
) -> dict:
    """Load EEG/RR data and prepare a MATLAB bundle."""
    subj_dir = ROOT / DERIVATIVES_DIR / FINAL_PIPELINE_DIR / subject
    edf_path = subj_dir / f"{subject}_desc-final_denoised_eeg.edf"
    rpeaks_path = subj_dir / f"{subject}_desc-rpeaks_events.csv"

    if not edf_path.exists():
        raise FileNotFoundError(f"Missing final EEG EDF: {edf_path}")
    if not rpeaks_path.exists():
        raise FileNotFoundError(f"Missing R-peaks CSV: {rpeaks_path}")
    if not CONDITIONS_CSV.exists():
        raise FileNotFoundError(f"Missing conditions CSV: {CONDITIONS_CSV}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    rename_channels_drop_reference(raw, verbose=False)
    set_channel_types_from_names(raw, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore")

    # Respect the State record / State stop interval, then shift condition windows
    # to the cropped raw time axis.
    raw, cropped = crop_to_state_annotations(raw)
    if cropped:
        print(f"  Cropped to State record/stop for {subject}")

    rpeaks_sec = load_rpeaks_csv(rpeaks_path)
    segments_abs = load_condition_segments(CONDITIONS_CSV, subject, condition)
    if not segments_abs:
        raise ValueError(f"No condition segments found for {subject} / {condition}")

    segments = _relative_segments_for_raw(raw, segments_abs)
    if not segments:
        raise ValueError(f"No usable segments remain after cropping for {subject} / {condition}")

    eeg_picks = mne.pick_types(raw.info, eeg=True, ecg=False, eog=False, misc=False, meg=False)
    eeg_names = [raw.ch_names[p] for p in eeg_picks]

    # Use the single longest condition interval.
    start_sec, end_sec = max(segments, key=lambda x: float(x[1] - x[0]))
    eeg_data_uv, rr_sec, rr_mid_sec, event_dict = _segment_to_arrays(
        raw=raw,
        start_sec=start_sec,
        end_sec=end_sec,
        eeg_picks=eeg_picks,
        rpeaks_sec=rpeaks_sec,
    )
    rr_clean, rr_mid_clean = _clean_rr(rr_sec, rr_mid_sec)
    if rr_clean.size < MIN_RR_INTERVALS:
        raise ValueError(
            f"Too few valid RR intervals for {subject} / {condition}: "
            f"{rr_clean.size} < {MIN_RR_INTERVALS}"
        )

    total_duration = float(eeg_data_uv.shape[1] / float(raw.info["sfreq"]))
    rri, t_rri = _interpolate_rr(rr_clean, rr_mid_clean, FS_RRI, total_duration)
    if rri.size == 0:
        raise ValueError(f"Unable to interpolate RR for {subject} / {condition}")
    rr = rr_clean.reshape(1, -1)
    rri = rri.reshape(1, -1)
    t_rri = t_rri.reshape(1, -1)
    event_dict = dict(event_dict)
    event_dict["type"] = condition
    event_dict["condition"] = condition
    event_parts = [event_dict]
    segment_summaries = [{"segment_index": 1, "start_sec": float(start_sec), "end_sec": float(end_sec), "selection": "longest"}]

    eeg_struct = _make_eeg_struct(
        raw=raw,
        eeg_data_uv=eeg_data_uv,
        eeg_names=eeg_names,
        event_parts=event_parts,
        subject=subject,
        condition=condition,
    )

    file_output = BHI_OUTPUT_DIR / subject / f"{subject}_desc-sdgm-lfhf-{condition}_BHI.mat"
    input_mat = BHI_INPUT_DIR / subject / f"{subject}_desc-sdgm-lfhf-{condition}_input.mat"
    meta_json = BHI_INPUT_DIR / subject / f"{subject}_desc-sdgm-lfhf-{condition}_input.json"

    payload = {
        "EEG": eeg_struct,
        "f_lims": np.asarray(F_LIMS, dtype=float),
        "RR": rr,
        "RRi": rri,
        "t_RRi": t_rri,
        "FS_rri": float(FS_RRI),
        "FS_bhi": float(FS_BHI),
        "TV": int(TV),
        "file_output": str(file_output),
    }

    return {
        "payload": payload,
        "meta": {
            "subject": subject,
            "condition": condition,
            "source_edf": str(edf_path),
            "source_rpeaks_csv": str(rpeaks_path),
            "segments_absolute": [list(x) for x in segments_abs],
            "segments_relative": [list(x) for x in segments],
            "segment_summaries": segment_summaries,
            "segment_policy": "longest",
            "n_channels": int(eeg_data_uv.shape[0]),
            "n_samples": int(eeg_data_uv.shape[1]),
            "n_rr": int(rr.shape[1]),
            "n_rri": int(rri.shape[1]),
            "file_output": str(file_output),
            "input_mat": str(input_mat),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "input_mat": input_mat,
        "meta_json": meta_json,
        "file_output": file_output,
    }


def _write_bundle(result: dict) -> None:
    input_mat: Path = result["input_mat"]
    meta_json: Path = result["meta_json"]
    payload: dict = result["payload"]
    meta: dict = result["meta"]

    input_mat.parent.mkdir(parents=True, exist_ok=True)
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    BHI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    savemat(str(input_mat), payload, do_compression=True)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved MAT → {input_mat}")
    print(f"Saved JSON → {meta_json}")
    print(f"MATLAB output should be written to → {result['file_output']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    subjects = _discover_subjects(ROOT / DERIVATIVES_DIR / FINAL_PIPELINE_DIR)
    if not subjects:
        print("ERROR: no final subjects found")
        return 1

    BHI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    BHI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BHI_GROUP_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "subjects": subjects,
        "conditions": CONDITIONS,
        "segment_policy": "longest",
        "use_longest_segment_only": USE_LONGEST_SEGMENT_ONLY,
        "fs_rri": FS_RRI,
        "fs_bhi": FS_BHI,
        "tv": TV,
        "f_lims": F_LIMS.tolist(),
        "items": [],
    }

    for subject in subjects:
        for condition in CONDITIONS:
            print(f"\n{'='*60}")
            print(f"Preparing BHI input — {subject} / {condition}")
            print(f"{'='*60}")
            try:
                result = _prepare_subject_condition(subject, condition)
                _write_bundle(result)
                summary["items"].append(result["meta"])
            except Exception as exc:
                print(f"ERROR: {subject} / {condition}: {exc}")
                summary["items"].append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "error": str(exc),
                    }
                )

    summary_path = BHI_GROUP_DIR / "group_desc-sdgm-lfhf_prep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary → {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

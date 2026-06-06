"""ecg_hrv.py — Condition-specific HRV analysis from RR intervals.

Pipeline
--------
For each subject:
1. Load ``*_desc-rpeaks_events.csv`` from ``derivatives/eeg-final/<subject>/``.
2. Filter R-peaks to condition windows from ``manual/conditions.csv``.
3. Compute RR intervals (ms).
4. Remove severe outliers (hard physiological bounds + robust MAD filter).
5. Compute time-domain and frequency-domain HRV metrics.
6. Export one wide-format CSV per condition to ``derivatives/group/hrv/``.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from eeg_config import DERIVATIVES_DIR, FINAL_PIPELINE_DIR
from eeg_common import filter_rpeaks_to_segments, load_condition_segments, load_rpeaks_csv


# ---------------------------------------------------------------------------
# ── Analysis settings ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

CONDITIONS: List[str] = ["EC", "EO"]

PROCESS_SELECTED_SUBJECTS: bool = False
SELECTED_SUBJECTS: List[str] = ["sub-eeg-1"]
EXCLUDE_SUBJECTS: List[str] = []

# RR cleaning
RR_MIN_MS: float = 300.0
RR_MAX_MS: float = 2000.0
USE_MAD_FILTER: bool = True
MAD_Z_THRESHOLD: float = 4.0
MIN_RR_REQUIRED: int = 30

# Frequency-domain settings
INTERP_FS_HZ: float = 4.0
VLF_BAND_HZ: Tuple[float, float] = (0.0033, 0.04)
LF_BAND_HZ: Tuple[float, float] = (0.04, 0.15)
HF_BAND_HZ: Tuple[float, float] = (0.15, 0.40)
TOTAL_BAND_HZ: Tuple[float, float] = (0.0033, 0.40)


# ---------------------------------------------------------------------------
# ── Paths ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

FINAL_DIR = ROOT / DERIVATIVES_DIR / FINAL_PIPELINE_DIR
CONDITIONS_CSV = ROOT / "manual" / "conditions.csv"
GROUP_HRV_DIR = ROOT / DERIVATIVES_DIR / "group" / "hrv"


@dataclass
class RRQuality:
    n_peaks: int
    n_rr_raw: int
    n_rr_clean: int
    n_rr_rejected: int
    pct_rr_rejected: float


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if np.sum(mask) < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _peak_freq_in_band(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.nan
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    if band_psd.size == 0:
        return np.nan
    return float(band_freqs[np.argmax(band_psd)])


def compute_rr_ms(rpeaks_sec: np.ndarray) -> np.ndarray:
    """Compute RR intervals in milliseconds from sorted R-peak times (seconds)."""
    if rpeaks_sec.size < 2:
        return np.array([], dtype=float)
    r = np.sort(np.unique(rpeaks_sec.astype(float)))
    rr_ms = np.diff(r) * 1000.0
    return rr_ms[np.isfinite(rr_ms)]


def clean_rr_intervals(rr_ms: np.ndarray) -> Tuple[np.ndarray, RRQuality]:
    """Clean RR intervals by hard bounds + robust MAD filter."""
    rr_ms = np.asarray(rr_ms, dtype=float)
    rr_ms = rr_ms[np.isfinite(rr_ms)]

    n_rr_raw = int(rr_ms.size)
    if n_rr_raw == 0:
        quality = RRQuality(0, 0, 0, 0, np.nan)
        return rr_ms, quality

    # 1) Hard physiological bounds
    mask = (rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)
    rr_keep = rr_ms[mask]

    # 2) Robust MAD filter
    if USE_MAD_FILTER and rr_keep.size >= 5:
        med = float(np.median(rr_keep))
        mad = float(np.median(np.abs(rr_keep - med)))
        if mad > 0:
            robust_z = 0.6745 * np.abs(rr_keep - med) / mad
            rr_keep = rr_keep[robust_z <= MAD_Z_THRESHOLD]

    n_rr_clean = int(rr_keep.size)
    n_rr_rejected = int(n_rr_raw - n_rr_clean)
    pct_rej = float((n_rr_rejected / n_rr_raw) * 100.0) if n_rr_raw > 0 else np.nan

    quality = RRQuality(
        n_peaks=n_rr_raw + 1,
        n_rr_raw=n_rr_raw,
        n_rr_clean=n_rr_clean,
        n_rr_rejected=n_rr_rejected,
        pct_rr_rejected=pct_rej,
    )
    return rr_keep, quality


def compute_time_domain_metrics(rr_ms: np.ndarray) -> Dict[str, float]:
    """Compute standard time-domain HRV metrics from cleaned RR intervals (ms)."""
    rr_ms = np.asarray(rr_ms, dtype=float)
    rr_ms = rr_ms[np.isfinite(rr_ms)]

    if rr_ms.size < MIN_RR_REQUIRED:
        return {
            "meanNN_ms": np.nan,
            "medianNN_ms": np.nan,
            "SDNN_ms": np.nan,
            "RMSSD_ms": np.nan,
            "pNN50_pct": np.nan,
            "IQRNN_ms": np.nan,
            "HR_mean_bpm": np.nan,
        }

    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr**2)) if diff_rr.size else np.nan
    pnn50 = 100.0 * np.mean(np.abs(diff_rr) > 50.0) if diff_rr.size else np.nan

    return {
        "meanNN_ms": float(np.mean(rr_ms)),
        "medianNN_ms": float(np.median(rr_ms)),
        "SDNN_ms": float(np.std(rr_ms, ddof=1)) if rr_ms.size > 1 else np.nan,
        "RMSSD_ms": float(rmssd),
        "pNN50_pct": float(pnn50),
        "IQRNN_ms": float(np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25)),
        "HR_mean_bpm": float(60000.0 / np.mean(rr_ms)),
    }


def compute_frequency_domain_metrics(rr_ms: np.ndarray) -> Dict[str, float]:
    """Compute frequency-domain HRV metrics using interpolated tachogram + Welch PSD."""
    rr_ms = np.asarray(rr_ms, dtype=float)
    rr_ms = rr_ms[np.isfinite(rr_ms)]

    if rr_ms.size < max(MIN_RR_REQUIRED, 16):
        return {
            "VLF_power_ms2": np.nan,
            "LF_power_ms2": np.nan,
            "HF_power_ms2": np.nan,
            "total_power_ms2": np.nan,
            "LF_HF_ratio": np.nan,
            "LF_nu": np.nan,
            "HF_nu": np.nan,
            "peak_LF_Hz": np.nan,
            "peak_HF_Hz": np.nan,
        }

    # Assign each RR interval to cumulative beat time
    rr_sec = rr_ms / 1000.0
    t_rr = np.cumsum(rr_sec)

    if t_rr.size < 4 or t_rr[-1] <= t_rr[0]:
        return {
            "VLF_power_ms2": np.nan,
            "LF_power_ms2": np.nan,
            "HF_power_ms2": np.nan,
            "total_power_ms2": np.nan,
            "LF_HF_ratio": np.nan,
            "LF_nu": np.nan,
            "HF_nu": np.nan,
            "peak_LF_Hz": np.nan,
            "peak_HF_Hz": np.nan,
        }

    # Uniform interpolation for spectral estimation
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / INTERP_FS_HZ)
    if t_uniform.size < 16:
        return {
            "VLF_power_ms2": np.nan,
            "LF_power_ms2": np.nan,
            "HF_power_ms2": np.nan,
            "total_power_ms2": np.nan,
            "LF_HF_ratio": np.nan,
            "LF_nu": np.nan,
            "HF_nu": np.nan,
            "peak_LF_Hz": np.nan,
            "peak_HF_Hz": np.nan,
        }

    rr_interp = np.interp(t_uniform, t_rr, rr_ms)
    rr_interp = rr_interp - np.mean(rr_interp)

    nperseg = min(256, rr_interp.size)
    freqs, psd = welch(rr_interp, fs=INTERP_FS_HZ, nperseg=nperseg, detrend="constant")

    vlf = _band_power(freqs, psd, VLF_BAND_HZ)
    lf = _band_power(freqs, psd, LF_BAND_HZ)
    hf = _band_power(freqs, psd, HF_BAND_HZ)
    total = _band_power(freqs, psd, TOTAL_BAND_HZ)

    lf_hf = lf / hf if np.isfinite(lf) and np.isfinite(hf) and hf > 0 else np.nan
    lf_hf_sum = lf + hf
    lf_nu = (lf / lf_hf_sum) * 100.0 if np.isfinite(lf_hf_sum) and lf_hf_sum > 0 else np.nan
    hf_nu = (hf / lf_hf_sum) * 100.0 if np.isfinite(lf_hf_sum) and lf_hf_sum > 0 else np.nan

    return {
        "VLF_power_ms2": _safe_float(vlf),
        "LF_power_ms2": _safe_float(lf),
        "HF_power_ms2": _safe_float(hf),
        "total_power_ms2": _safe_float(total),
        "LF_HF_ratio": _safe_float(lf_hf),
        "LF_nu": _safe_float(lf_nu),
        "HF_nu": _safe_float(hf_nu),
        "peak_LF_Hz": _safe_float(_peak_freq_in_band(freqs, psd, LF_BAND_HZ)),
        "peak_HF_Hz": _safe_float(_peak_freq_in_band(freqs, psd, HF_BAND_HZ)),
    }


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

    excluded = set(EXCLUDE_SUBJECTS)
    subjects = [s for s in subjects if s not in excluded]
    return subjects


def _base_row(subject: str, condition: str, status: str) -> Dict[str, object]:
    row = {
        "subject": subject,
        "condition": condition,
        "status": status,
        "n_peaks": "",
        "n_rr_raw": "",
        "n_rr_clean": "",
        "n_rr_rejected": "",
        "pct_rr_rejected": "",
        "meanNN_ms": "",
        "medianNN_ms": "",
        "SDNN_ms": "",
        "RMSSD_ms": "",
        "pNN50_pct": "",
        "IQRNN_ms": "",
        "HR_mean_bpm": "",
        "VLF_power_ms2": "",
        "LF_power_ms2": "",
        "HF_power_ms2": "",
        "total_power_ms2": "",
        "LF_HF_ratio": "",
        "LF_nu": "",
        "HF_nu": "",
        "peak_LF_Hz": "",
        "peak_HF_Hz": "",
    }
    return row


def process_subject_condition(subject: str, condition: str) -> Dict[str, object]:
    subj_dir = FINAL_DIR / subject
    rpeaks_csv = subj_dir / f"{subject}_desc-rpeaks_events.csv"

    if not rpeaks_csv.exists():
        return _base_row(subject, condition, "missing_rpeaks_csv")

    segments = load_condition_segments(CONDITIONS_CSV, subject, condition)
    if not segments:
        return _base_row(subject, condition, "missing_condition_segments")

    rpeaks_sec = load_rpeaks_csv(rpeaks_csv)
    rpeaks_sec = filter_rpeaks_to_segments(rpeaks_sec, segments)

    if rpeaks_sec.size < 2:
        row = _base_row(subject, condition, "insufficient_rpeaks")
        row["n_peaks"] = int(rpeaks_sec.size)
        return row

    rr_raw = compute_rr_ms(rpeaks_sec)
    rr_clean, quality = clean_rr_intervals(rr_raw)

    time_metrics = compute_time_domain_metrics(rr_clean)
    freq_metrics = compute_frequency_domain_metrics(rr_clean)

    status = "ok" if quality.n_rr_clean >= MIN_RR_REQUIRED else "insufficient_clean_rr"

    row = _base_row(subject, condition, status)
    row.update({
        "n_peaks": quality.n_peaks,
        "n_rr_raw": quality.n_rr_raw,
        "n_rr_clean": quality.n_rr_clean,
        "n_rr_rejected": quality.n_rr_rejected,
        "pct_rr_rejected": quality.pct_rr_rejected,
    })
    row.update(time_metrics)
    row.update(freq_metrics)
    return row


def write_wide_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    cols = [
        "subject",
        "condition",
        "status",
        "n_peaks",
        "n_rr_raw",
        "n_rr_clean",
        "n_rr_rejected",
        "pct_rr_rejected",
        "meanNN_ms",
        "medianNN_ms",
        "SDNN_ms",
        "RMSSD_ms",
        "pNN50_pct",
        "IQRNN_ms",
        "HR_mean_bpm",
        "VLF_power_ms2",
        "LF_power_ms2",
        "HF_power_ms2",
        "total_power_ms2",
        "LF_HF_ratio",
        "LF_nu",
        "HF_nu",
        "peak_LF_Hz",
        "peak_HF_Hz",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    GROUP_HRV_DIR.mkdir(parents=True, exist_ok=True)

    if not CONDITIONS_CSV.exists():
        print(f"ERROR: conditions file not found: {CONDITIONS_CSV}")
        return 1

    subjects = _discover_subjects(FINAL_DIR)
    if not subjects:
        print(f"ERROR: No subject folders found under {FINAL_DIR}")
        return 1

    all_rows: List[Dict[str, object]] = []
    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_subjects": len(subjects),
        "conditions": CONDITIONS,
        "settings": {
            "RR_MIN_MS": RR_MIN_MS,
            "RR_MAX_MS": RR_MAX_MS,
            "USE_MAD_FILTER": USE_MAD_FILTER,
            "MAD_Z_THRESHOLD": MAD_Z_THRESHOLD,
            "MIN_RR_REQUIRED": MIN_RR_REQUIRED,
            "INTERP_FS_HZ": INTERP_FS_HZ,
            "VLF_BAND_HZ": list(VLF_BAND_HZ),
            "LF_BAND_HZ": list(LF_BAND_HZ),
            "HF_BAND_HZ": list(HF_BAND_HZ),
            "TOTAL_BAND_HZ": list(TOTAL_BAND_HZ),
        },
        "per_condition": {},
    }

    for condition in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"HRV analysis — condition={condition}")
        print(f"{'='*60}")

        rows: List[Dict[str, object]] = []
        for subject in subjects:
            row = process_subject_condition(subject, condition)
            rows.append(row)

        path = GROUP_HRV_DIR / f"group_desc-hrv-{condition}_metrics_wide.csv"
        write_wide_csv(path, rows)
        print(f"Saved: {path.name}")

        all_rows.extend(rows)

        n_ok = sum(1 for r in rows if r["status"] == "ok")
        run_summary["per_condition"][condition] = {
            "n_rows": len(rows),
            "n_ok": n_ok,
            "n_not_ok": len(rows) - n_ok,
            "csv": str(path),
        }

    combined_path = GROUP_HRV_DIR / "group_desc-hrv_metrics_wide.csv"
    write_wide_csv(combined_path, all_rows)
    print(f"Saved: {combined_path.name}")

    processing_path = GROUP_HRV_DIR / "group_desc-hrv_processing.json"
    with open(processing_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Saved: {processing_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

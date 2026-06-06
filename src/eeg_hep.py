"""eeg_hep.py — Heartbeat Evoked Potential (HEP) analysis.

Pipeline
--------
For each subject:
1. Load ``*_desc-final_denoised_eeg.edf`` from ``derivatives/eeg-final/<subj>/``
2. Crop raw to 'State record' → 'State stop' annotations (warns if absent).
3. Load ``manual/conditions.csv`` → filter rows by subject + CONDITION.
4. Load ``*_desc-rpeaks_events.csv`` → keep only R-peaks in condition windows.
5. Epoch around each R-peak (EPOCH_TMIN … EPOCH_TMAX, baseline BASELINE).
6. Electrode-based rejection: drop epochs exceeding AMPLITUDE_REJECT_THRESH_UV
   peak-to-peak on any EEG channel.
7. Compute HEP = average of accepted epochs.
8. Export results to ``derivatives/eeg-hep/<subj>/``.
9. Generate a multi-page PDF report.

Outputs (all in ``derivatives/eeg-hep/<subj>/``)
-------------------------------------------------
``*_desc-hep-{CONDITION}_avg.fif``         — HEP evoked (MNE Evoked)
``*_desc-hep-{CONDITION}_epochs.fif``      — Accepted epochs
``*_desc-hep-{CONDITION}_stats.json``      — Epoch counts, rejection log
``*_desc-hep-{CONDITION}_report.pdf``      — Multi-page PDF report
``*_desc-hep-{CONDITION}_processing.json`` — Full provenance
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# ── Analysis settings ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

CONDITION: str = "EC"           # "EO" or "EC"

EPOCH_TMIN: float = -0.2        # seconds before R-peak
EPOCH_TMAX: float = 0.8         # seconds after R-peak
BASELINE: Tuple[float, float] = (-0.2, -0.05)   # baseline correction window
TOPO_TIMES_SEC: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6]  # 200, 300, 400, 500, 600 ms

AMPLITUDE_REJECT_THRESH_UV: float = 150.0        # per-electrode peak-to-peak µV

# ---------------------------------------------------------------------------
# ── Processing toggles ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

PROCESS_SELECTED_SUBJECTS: bool = False
SELECTED_SUBJECTS: List[str] = ["sub-eeg-1"]

GENERATE_SHAM: bool = True          # toggle sham (jittered) R-peak analysis
SHAM_JITTER_RANGE_MS: float = 500.0  # ±N ms jitter window

# ---------------------------------------------------------------------------
# ── Paths (relative to project root) ────────────────────────────────────────
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent          # project root
CONDITIONS_CSV = ROOT / "manual" / "conditions.csv"
BAD_ELECTRODES_CSV = ROOT / "manual" / "bad_electrodes.csv"

sys.path.insert(0, str(Path(__file__).parent))

from eeg_config import DERIVATIVES_DIR, FINAL_PIPELINE_DIR, HEP_PIPELINE_DIR
from eeg_common import (
    crop_to_state_annotations,
    filter_rpeaks_to_segments,
    load_condition_segments,
    load_rpeaks_csv,
    rename_channels_drop_reference,
    reject_epochs_by_electrode,
    rpeaks_to_mne_events,
    set_channel_types_from_names,
    sort_eeg_picks_conventional,
)

FINAL_DIR = ROOT / DERIVATIVES_DIR / FINAL_PIPELINE_DIR
HEP_DIR = ROOT / DERIVATIVES_DIR / HEP_PIPELINE_DIR


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------

def _load_bad_electrodes_for_subject(path: Path, subject: str) -> List[str]:
    """Load manual bad EEG electrodes for one subject."""
    if not path.exists():
        return []

    bads: List[str] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_subject = row.get("subject", "").strip()
            if row_subject != subject:
                continue
            ch = row.get("bad_electrode", "").strip()
            if ch:
                bads.append(ch)
    return sorted(set(bads))


def _build_channelwise_hep_evoked(epochs, rejection_log: dict, bad_channels: List[str]):
    """Create an Evoked where each channel averages only its own good epochs."""
    import mne as _mne

    evoked = epochs.average()
    eeg_picks = _mne.pick_types(epochs.info, eeg=True)
    eeg_names = [epochs.ch_names[p] for p in eeg_picks]

    used_channels = rejection_log.get("used_channels", [])
    used_to_idx = {ch: i for i, ch in enumerate(used_channels)}
    bad_mask = np.asarray(rejection_log.get("bad_mask", []), dtype=bool)

    data = epochs.get_data(picks=eeg_picks)  # (n_epochs, n_eeg, n_times)
    out = np.full((len(eeg_picks), data.shape[-1]), np.nan, dtype=float)
    good_counts: dict = {}

    for local_idx, ch_name in enumerate(eeg_names):
        if ch_name in bad_channels:
            good_counts[ch_name] = 0
            continue

        used_idx = used_to_idx.get(ch_name)
        if used_idx is None or bad_mask.size == 0:
            out[local_idx] = np.nanmean(data[:, local_idx, :], axis=0)
            good_counts[ch_name] = int(data.shape[0])
            continue

        good_ep = ~bad_mask[:, used_idx]
        good_counts[ch_name] = int(np.sum(good_ep))
        if np.any(good_ep):
            out[local_idx] = np.nanmean(data[good_ep, local_idx, :], axis=0)

    evoked.data[eeg_picks, :] = out
    evoked.info["bads"] = sorted(set(evoked.info.get("bads", []) + list(bad_channels)))
    return evoked, good_counts


def _generate_sham_rpeaks(
    rpeaks_sec: np.ndarray,
    segments: List[Tuple[float, float]],
    jitter_range_ms: float = 500.0,
    conflict_margin_sec: float = 0.2,
) -> np.ndarray:
    """Generate jittered sham R-peaks to match real R-peaks in count/distribution.

    Parameters
    ----------
    rpeaks_sec : np.ndarray
        Real R-peak times in seconds.
    segments : list of (float, float)
        Condition segments ``[start, end)`` in seconds.
    jitter_range_ms : float
        Jitter range (±N ms) for each sham peak.
    conflict_margin_sec : float
        Exclude sham peaks within this margin of real R-peaks.

    Returns
    -------
    np.ndarray
        Sham R-peak times in seconds.
    """
    jitter_range_sec = jitter_range_ms / 1000.0
    np.random.seed(42)  # reproducibility

    sham_peaks = []
    for real_rpeak in rpeaks_sec:
        # Random jitter ∈ [-range, +range]
        jitter = np.random.uniform(-jitter_range_sec, jitter_range_sec)
        sham_time = real_rpeak + jitter

        # Check: not too close to any real R-peak
        if np.any(np.abs(rpeaks_sec - sham_time) < conflict_margin_sec):
            continue

        # Check: within condition segments
        in_segment = any(s <= sham_time < e for s, e in segments)
        if in_segment:
            sham_peaks.append(sham_time)

    return np.asarray(sorted(sham_peaks), dtype=float)


def _save_hep_report(
    report_path: Path,
    evoked,
    rejection_log: dict,
    subject: str,
    condition: str,
    bad_channels: List[str],
    topo_times: List[float] | None = None,
) -> None:
    """Generate a multi-page PDF HEP report.

    Pages
    -----
    1. Summary (subject, condition, epoch counts, per-electrode rejections)
    2. Butterfly plot (all EEG channels)
    3. Per-channel HEP traces (grid, 4 columns)
    4. Topographic maps at key latencies
    """
    import mne as _mne

    if topo_times is None:
        topo_times = TOPO_TIMES_SEC

    report_path.parent.mkdir(parents=True, exist_ok=True)
    cond_label = f"HEP · {subject} · condition={condition}"

    with PdfPages(report_path) as pdf:
        # ── Page 1: Summary ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")

        summary_lines = [
            f"Subject:   {subject}",
            f"Condition: {condition}",
            "",
            f"Epoch window:  {EPOCH_TMIN:.3f} s  →  {EPOCH_TMAX:.3f} s",
            f"Baseline:      {BASELINE[0]:.3f} s  →  {BASELINE[1]:.3f} s",
            f"Reject thresh: {AMPLITUDE_REJECT_THRESH_UV:.0f} µV (peak-to-peak, channel-wise)",
            "",
            f"R-peaks in condition:  {rejection_log['n_total']}",
            f"Global epoch drop:     none (channel-wise masking only)",
            f"Epochs kept:           {rejection_log['n_accepted']}",
        ]

        if bad_channels:
            summary_lines += ["", f"Manual bad channels excluded: {', '.join(bad_channels)}"]

        nonzero_by_channel = {
            ch: cnt for ch, cnt in rejection_log["by_channel"].items() if cnt > 0
        }
        if nonzero_by_channel:
            summary_lines += ["", "Rejections by channel (top 10):"]
            sorted_ch = sorted(
                nonzero_by_channel.items(), key=lambda x: -x[1]
            )
            for ch, cnt in sorted_ch[:10]:
                summary_lines.append(f"  {ch:12s}  {cnt}")
        else:
            summary_lines += ["", "No channel exceeded the rejection threshold."]

        ax.text(
            0.05, 0.95, "\n".join(summary_lines),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=11,
            fontfamily="monospace",
        )
        ax.set_title(cond_label, fontsize=13, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Butterfly plot ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 5))
        times = evoked.times * 1000  # ms
        eeg_picks_idx = _mne.pick_types(evoked.info, eeg=True)
        data_uv = evoked.data[eeg_picks_idx] * 1e6

        for trace in data_uv:
            ax.plot(times, trace, color="steelblue", lw=0.6, alpha=0.5)

        mean_trace = np.nanmean(data_uv, axis=0)
        ax.plot(times, mean_trace, color="navy", lw=1.8, label="Grand mean")

        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="red", lw=0.8, ls="--", label="R-peak")
        ax.axvspan(BASELINE[0] * 1000, BASELINE[1] * 1000,
                   color="yellow", alpha=0.15, label="Baseline")
        for t in topo_times:
            ax.axvline(t * 1000, color="gray", lw=0.8, ls=":", alpha=0.7)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"Butterfly plot — {cond_label}\n(n={rejection_log['n_accepted']} epochs)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Pretty scalp-layout per-channel traces ─────────────────
        ch_positions = [
            ["", "Fp1", "", "Fp2", ""],
            ["F7", "F3", "Fz", "F4", "F8"],
            ["T3", "C3", "Cz", "C4", "T4"],
            ["T5", "P3", "Pz", "P4", "T6"],
            ["", "O1", "", "O2", ""],
        ]

        fig, axes = plt.subplots(5, 5, figsize=(12, 10), sharex=True, sharey=True)
        fig.patch.set_facecolor("white")

        eeg_evoked = evoked.copy().pick("eeg")
        ch_to_idx = {ch: i for i, ch in enumerate(eeg_evoked.ch_names)}
        t_ms = eeg_evoked.times * 1000
        y_lim = np.nanmax(np.abs(eeg_evoked.data * 1e6))
        y_lim = max(3.0, float(y_lim) if np.isfinite(y_lim) else 10.0)

        for r in range(5):
            for c in range(5):
                ax_c = axes[r, c]
                ch = ch_positions[r][c]
                if ch == "":
                    ax_c.axis("off")
                    continue

                idx = ch_to_idx.get(ch)
                if idx is None:
                    ax_c.axis("off")
                    continue

                trace = eeg_evoked.data[idx] * 1e6
                ax_c.set_facecolor("#f8f9fa")
                ax_c.axhline(0, color="#999", lw=0.6, ls="--")
                ax_c.axvline(0, color="#d62728", lw=0.8, ls="--")
                for t in topo_times:
                    ax_c.axvline(t * 1000, color="#bbb", lw=0.6, ls=":")

                if np.all(~np.isfinite(trace)):
                    ax_c.text(0.5, 0.5, "BAD", transform=ax_c.transAxes,
                              ha="center", va="center", fontsize=8, color="#b22222")
                else:
                    ax_c.plot(t_ms, trace, color="#1f77b4", lw=1.2)

                ax_c.set_title(ch, fontsize=9, pad=2)
                ax_c.set_xlim(t_ms[0], t_ms[-1])
                ax_c.set_ylim(-y_lim, y_lim)
                ax_c.tick_params(labelsize=7, length=2)

        fig.suptitle(f"Per-channel HEP (scalp layout) — {cond_label}", fontsize=13, y=0.98)
        fig.text(0.5, 0.03, "Time (ms)", ha="center", fontsize=10)
        fig.text(0.03, 0.5, "Amplitude (µV)", va="center", rotation=90, fontsize=10)
        fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 4: Topographic maps ──────────────────────────────────────────
        valid_topo = [t for t in topo_times if EPOCH_TMIN <= t <= EPOCH_TMAX]
        if valid_topo:
            try:
                evoked_topo = evoked.copy().pick("eeg")
                valid_ch_mask = np.all(np.isfinite(evoked_topo.data), axis=1)
                if not np.any(valid_ch_mask):
                    raise RuntimeError("No valid EEG channels available for topomap")
                valid_ch = [ch for ch, ok in zip(evoked_topo.ch_names, valid_ch_mask) if ok]
                evoked_topo.pick(valid_ch)

                fig = evoked_topo.plot_topomap(
                    times=valid_topo,
                    show=False,
                    colorbar=True,
                    units="µV",
                    scalings={"eeg": 1e6},
                )
                fig.suptitle(f"Topographic maps — {cond_label}", fontsize=11, y=1.02)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                # Topomap requires a valid montage; warn but don't crash
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.axis("off")
                ax.text(0.5, 0.5, f"Topomap unavailable: {exc}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10)
                pdf.savefig(fig)
                plt.close(fig)

    print(f"  Saved HEP report → {report_path}")


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(subject: str) -> None:
    import mne

    print(f"\n{'='*60}")
    print(f"  HEP: {subject}  (condition={CONDITION})")
    print(f"{'='*60}")

    # ── Paths ────────────────────────────────────────────────────────────────
    final_subj_dir = FINAL_DIR / subject
    hep_subj_dir = HEP_DIR / subject
    hep_subj_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{subject}_desc-hep-{CONDITION}"

    edf_path = final_subj_dir / f"{subject}_desc-final_denoised_eeg.edf"
    rpeak_csv_path = final_subj_dir / f"{subject}_desc-rpeaks_events.csv"

    for p, label in [(edf_path, "final EDF"), (rpeak_csv_path, "rpeaks CSV"),
                     (CONDITIONS_CSV, "conditions CSV")]:
        if not p.exists():
            print(f"  ERROR: Required file not found: {p}  ({label}) — skipping.")
            return

    # ── 1. Load EEG ──────────────────────────────────────────────────────────
    print("  Loading EDF …")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    rename_channels_drop_reference(raw, verbose=False)
    set_channel_types_from_names(raw, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore")
    sfreq = raw.info["sfreq"]

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    n_ecg = len(mne.pick_types(raw.info, ecg=True))
    n_eog = len(mne.pick_types(raw.info, eog=True))
    n_misc = len(mne.pick_types(raw.info, misc=True))
    print(f"  Channels: {len(raw.ch_names)}, sfreq={sfreq} Hz, "
          f"duration={raw.times[-1]:.1f} s")
    print(f"  Channel types: EEG={n_eeg}, ECG={n_ecg}, EOG={n_eog}, MISC={n_misc}")

    manual_bad_channels = _load_bad_electrodes_for_subject(BAD_ELECTRODES_CSV, subject)
    present_bad_channels = [ch for ch in manual_bad_channels if ch in raw.ch_names]
    if present_bad_channels:
        print(f"  Manual bad channels excluded from analysis: {present_bad_channels}")

    # ── 2. Crop to State record / State stop ─────────────────────────────────
    raw, cropped = crop_to_state_annotations(raw)
    if cropped:
        print(f"  Cropped to State annotations: {raw.times[-1]:.1f} s remaining")

    # ── 3. Load condition segments ───────────────────────────────────────────
    segments = load_condition_segments(CONDITIONS_CSV, subject, CONDITION)
    if not segments:
        print(f"  WARNING: No '{CONDITION}' segments found in conditions.csv for {subject} — skipping.")
        return
    total_cond_sec = sum(e - s for s, e in segments)
    print(f"  Condition '{CONDITION}': {len(segments)} segment(s), "
          f"{total_cond_sec:.1f} s total")

    # ── 4. Load & filter R-peaks ─────────────────────────────────────────────
    rpeaks_all = load_rpeaks_csv(rpeak_csv_path)
    rpeaks_cond = filter_rpeaks_to_segments(rpeaks_all, segments)
    print(f"  R-peaks: {len(rpeaks_all)} total, {len(rpeaks_cond)} in condition")

    if len(rpeaks_cond) == 0:
        print("  ERROR: No R-peaks in condition window — skipping.")
        return

    # ── 5. Epoch ─────────────────────────────────────────────────────────────
    events = rpeaks_to_mne_events(rpeaks_cond, sfreq)
    event_id = {"R_peak": 999}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        preload=True,
        reject=None,       # manual rejection below
        verbose=False,
    )
    print(f"  Epochs created: {len(epochs)} (before amplitude rejection)")

    # ── 6. Electrode-based rejection (channel-wise, no epoch drops) ─────────
    epochs_clean, rejection_log = reject_epochs_by_electrode(
        epochs,
        thresh_uv=AMPLITUDE_REJECT_THRESH_UV,
        excluded_channels=present_bad_channels,
    )
    print(f"  Epochs kept: {rejection_log['n_accepted']} / {rejection_log['n_total']} "
          f"(channel-wise masking; no global epoch rejection)")

    nonzero_top = [(ch, cnt) for ch, cnt in rejection_log["by_channel"].items() if cnt > 0]
    if nonzero_top:
        top = sorted(nonzero_top, key=lambda x: -x[1])[:5]
        print(f"  Top channels by rejected epochs (channel-wise): {top}")
    else:
        print("  No channel exceeded threshold in this condition.")

    # ── 7. Compute HEP (per-channel good-epoch averaging) ───────────────────
    evoked, good_counts_by_channel = _build_channelwise_hep_evoked(
        epochs_clean,
        rejection_log,
        present_bad_channels,
    )
    min_good = int(min(good_counts_by_channel.values())) if good_counts_by_channel else 0
    max_good = int(max(good_counts_by_channel.values())) if good_counts_by_channel else 0
    print(f"  HEP computed (channel-wise): good epochs per channel min={min_good}, max={max_good}")

    # ── 8. Save FIF outputs ──────────────────────────────────────────────────
    avg_path = hep_subj_dir / f"{tag}_avg.fif"
    epochs_path = hep_subj_dir / f"{tag}_epochs.fif"
    evoked.save(str(avg_path), overwrite=True)
    epochs_clean.save(str(epochs_path), overwrite=True)
    print(f"  Saved evoked → {avg_path.name}")
    print(f"  Saved epochs → {epochs_path.name}")

    # ── 9. Stats JSON ─────────────────────────────────────────────────────────
    stats = {
        "subject": subject,
        "condition": CONDITION,
        "n_rpeaks_total": int(len(rpeaks_all)),
        "n_rpeaks_in_condition": int(len(rpeaks_cond)),
        "n_epochs_total": rejection_log["n_total"],
        "n_epochs_accepted": rejection_log["n_accepted"],
        "n_epochs_rejected": rejection_log["n_rejected"],
        "amplitude_reject_thresh_uv": AMPLITUDE_REJECT_THRESH_UV,
        "manual_bad_channels": present_bad_channels,
        "manual_bad_channels_missing_in_data": [
            ch for ch in manual_bad_channels if ch not in raw.ch_names
        ],
        "rejection_by_channel": rejection_log["by_channel"],
        "good_counts_by_channel": good_counts_by_channel,
        "epoch_tmin": EPOCH_TMIN,
        "epoch_tmax": EPOCH_TMAX,
        "baseline": list(BASELINE),
    }
    stats_path = hep_subj_dir / f"{tag}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats → {stats_path.name}")

    # ── 10. PDF report ────────────────────────────────────────────────────────
    report_path = hep_subj_dir / f"{tag}_report.pdf"
    _save_hep_report(
        report_path=report_path,
        evoked=evoked,
        rejection_log=rejection_log,
        subject=subject,
        condition=CONDITION,
        bad_channels=present_bad_channels,
        topo_times=TOPO_TIMES_SEC,
    )

    # ── 11. Provenance JSON ───────────────────────────────────────────────────
    provenance = {
        "script": "eeg_hep.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subject": subject,
        "condition": CONDITION,
        "inputs": {
            "edf": str(edf_path),
            "rpeaks_csv": str(rpeak_csv_path),
            "conditions_csv": str(CONDITIONS_CSV),
        },
        "settings": {
            "epoch_tmin": EPOCH_TMIN,
            "epoch_tmax": EPOCH_TMAX,
            "baseline": list(BASELINE),
            "amplitude_reject_thresh_uv": AMPLITUDE_REJECT_THRESH_UV,
        },
        "outputs": {
            "avg_fif": str(avg_path),
            "epochs_fif": str(epochs_path),
            "stats_json": str(stats_path),
            "report_pdf": str(report_path),
        },
        **stats,
    }
    prov_path = hep_subj_dir / f"{tag}_processing.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"  Saved provenance → {prov_path.name}")

    # ── SHAM HEP (if enabled) ────────────────────────────────────────────────
    if GENERATE_SHAM:
        sham_condition = f"sham-{CONDITION}"
        print(f"\n  Generating sham HEP ({subject} / {sham_condition})…")

        # Generate jittered sham R-peaks
        rpeaks_sham = _generate_sham_rpeaks(
            rpeaks_cond,
            segments,
            jitter_range_ms=SHAM_JITTER_RANGE_MS,
            conflict_margin_sec=0.2,
        )
        print(f"  Sham R-peaks: {len(rpeaks_cond)} real → {len(rpeaks_sham)} sham")

        if len(rpeaks_sham) == 0:
            print("  WARNING: No valid sham R-peaks generated — skipping sham HEP.")
        else:
            # Epoch around sham R-peaks
            events_sham = rpeaks_to_mne_events(rpeaks_sham, sfreq)
            epochs_sham = mne.Epochs(
                raw,
                events_sham,
                event_id={"R_peak": 999},
                tmin=EPOCH_TMIN,
                tmax=EPOCH_TMAX,
                baseline=BASELINE,
                preload=True,
                reject=None,
                verbose=False,
            )
            print(f"  Sham epochs created: {len(epochs_sham)}")

            # Electrode-based rejection (same as real)
            epochs_sham_clean, rejection_log_sham = reject_epochs_by_electrode(
                epochs_sham,
                thresh_uv=AMPLITUDE_REJECT_THRESH_UV,
                excluded_channels=present_bad_channels,
            )
            print(f"  Sham epochs kept: {rejection_log_sham['n_accepted']} / {rejection_log_sham['n_total']}")

            # Compute sham HEP
            evoked_sham, good_counts_sham = _build_channelwise_hep_evoked(
                epochs_sham_clean,
                rejection_log_sham,
                present_bad_channels,
            )
            min_good_sham = int(min(good_counts_sham.values())) if good_counts_sham else 0
            max_good_sham = int(max(good_counts_sham.values())) if good_counts_sham else 0
            print(f"  Sham HEP computed: good epochs per channel min={min_good_sham}, max={max_good_sham}")

            # Save sham outputs
            tag_sham = f"{subject}_desc-hep-{sham_condition}"
            avg_sham_path = hep_subj_dir / f"{tag_sham}_avg.fif"
            epochs_sham_path = hep_subj_dir / f"{tag_sham}_epochs.fif"
            evoked_sham.save(str(avg_sham_path), overwrite=True)
            epochs_sham_clean.save(str(epochs_sham_path), overwrite=True)
            print(f"  Saved sham evoked → {avg_sham_path.name}")
            print(f"  Saved sham epochs → {epochs_sham_path.name}")

            # Sham stats JSON
            stats_sham = {
                "subject": subject,
                "condition": sham_condition,
                "source_condition": CONDITION,
                "n_rpeaks_real": int(len(rpeaks_cond)),
                "n_rpeaks_sham": int(len(rpeaks_sham)),
                "n_epochs_total": rejection_log_sham["n_total"],
                "n_epochs_accepted": rejection_log_sham["n_accepted"],
                "n_epochs_rejected": rejection_log_sham["n_rejected"],
                "amplitude_reject_thresh_uv": AMPLITUDE_REJECT_THRESH_UV,
                "manual_bad_channels": present_bad_channels,
                "rejection_by_channel": rejection_log_sham["by_channel"],
                "good_counts_by_channel": good_counts_sham,
                "epoch_tmin": EPOCH_TMIN,
                "epoch_tmax": EPOCH_TMAX,
                "baseline": list(BASELINE),
            }
            stats_sham_path = hep_subj_dir / f"{tag_sham}_stats.json"
            with open(stats_sham_path, "w", encoding="utf-8") as f:
                json.dump(stats_sham, f, indent=2)
            print(f"  Saved sham stats → {stats_sham_path.name}")

            # Sham PDF report
            report_sham_path = hep_subj_dir / f"{tag_sham}_report.pdf"
            _save_hep_report(
                report_path=report_sham_path,
                evoked=evoked_sham,
                rejection_log=rejection_log_sham,
                subject=subject,
                condition=sham_condition,
                bad_channels=present_bad_channels,
                topo_times=TOPO_TIMES_SEC,
            )

            # Sham provenance JSON
            provenance_sham = {
                "script": "eeg_hep.py (sham)",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "subject": subject,
                "condition": sham_condition,
                "source_condition": CONDITION,
                "inputs": {
                    "edf": str(edf_path),
                    "rpeaks_csv": str(rpeak_csv_path),
                    "conditions_csv": str(CONDITIONS_CSV),
                },
                "settings": {
                    "epoch_tmin": EPOCH_TMIN,
                    "epoch_tmax": EPOCH_TMAX,
                    "baseline": list(BASELINE),
                    "amplitude_reject_thresh_uv": AMPLITUDE_REJECT_THRESH_UV,
                    "sham_jitter_range_ms": SHAM_JITTER_RANGE_MS,
                },
                "outputs": {
                    "avg_fif": str(avg_sham_path),
                    "epochs_fif": str(epochs_sham_path),
                    "stats_json": str(stats_sham_path),
                    "report_pdf": str(report_sham_path),
                },
                **stats_sham,
            }
            prov_sham_path = hep_subj_dir / f"{tag_sham}_processing.json"
            with open(prov_sham_path, "w", encoding="utf-8") as f:
                json.dump(provenance_sham, f, indent=2)
            print(f"  Saved sham provenance → {prov_sham_path.name}")

    print(f"  ✓ Done: {subject} / {CONDITION}")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if PROCESS_SELECTED_SUBJECTS:
        subjects = SELECTED_SUBJECTS
    else:
        # Auto-discover all subjects that have a final folder
        subjects = sorted(
            p.name for p in FINAL_DIR.iterdir()
            if p.is_dir() and p.name.startswith("sub-eeg-")
        )

    print(f"HEP analysis — {len(subjects)} subject(s), condition={CONDITION}")
    for subject in subjects:
        try:
            process_subject(subject)
        except Exception as exc:
            print(f"  ERROR processing {subject}: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

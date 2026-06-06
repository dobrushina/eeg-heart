"""eeg_hep_group.py — Group-level Heartbeat Evoked Potential analysis.

Aggregates individual-subject HEP results into group-level statistics and reports.

Pipeline
--------
1. Load all subject HEP evoked files from ``derivatives/eeg-hep/sub-eeg-*/``
2. Compute grand average across subjects
3. Per-channel statistics (mean, std, min, max)
4. Per-channel outlier detection (subjects > N SD from mean)
5. Generate group report with visualizations

Outputs (all in ``derivatives/group/eeg-hep/``)
-----------------------------------------------
``group_desc-hep-{CONDITION}_avg.fif``         — Grand-average evoked
``group_desc-hep-{CONDITION}_stats.json``      — Per-channel statistics + outliers
``group_desc-hep-{CONDITION}_report.pdf``      — Multi-page PDF report
``group_desc-hep-{CONDITION}_processing.json`` — Full provenance
"""

from __future__ import annotations

import json
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from eeg_config import DERIVATIVES_DIR, HEP_PIPELINE_DIR
from eeg_common import sort_eeg_picks_conventional

# ---------------------------------------------------------------------------
# ── Analysis settings ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

CONDITION: List[str] | str = ["EC", "EO"]  # "EC", "EO", or ["EC", "EO"]
OUTLIER_THRESHOLD_SD: float = 2.0  # exclude per-channel if > N SD from mean
INCLUDE_SUBJECTS: List[str] | None = None  # None = all available
EXCLUDE_SUBJECTS: List[str] = []  # explicit exclusion

HEP_WINDOWS_SEC = {
    "250_400": (0.250, 0.400),
    "250_350": (0.250, 0.350),
    "350_550": (0.350, 0.550),
    "400_600": (0.400, 0.600),
    "250_600": (0.250, 0.600),
}
HEP_CHANNELS = ["Fz", "Cz", "Pz"]
HEP_ROI_CHANNELS = ["Fz", "Cz", "Pz", "C3", "C4"]
HEP_ROI_NAME = "CentralROI"

# ---------------------------------------------------------------------------
# ── Paths ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

HEP_DIR = ROOT / DERIVATIVES_DIR / HEP_PIPELINE_DIR
GROUP_HEP_DIR = ROOT / DERIVATIVES_DIR / "group" / HEP_PIPELINE_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_subject_heps(
    hep_dir: Path,
    condition: str,
    include_subjects: List[str] | None = None,
    exclude_subjects: List[str] | None = None,
) -> Tuple[Dict[str, dict], List[str]]:
    """Load HEP evoked + stats for all subjects in condition.

    Subjects without HEP files are included with evoked=None and stats={}.

    Returns
    -------
    subject_data : dict
        Keys: subject names. Values: {"evoked": MNE Evoked or None, "stats": dict}.
    subject_order : list
        Sorted subject list (includes all subjects, even missing ones).
    """
    import mne

    exclude = set(exclude_subjects or [])
    subject_data: Dict[str, dict] = {}

    # Discover subjects
    subj_dirs = sorted(
        p for p in hep_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-eeg-")
    )

    for subj_dir in subj_dirs:
        subject = subj_dir.name
        if subject in exclude:
            continue
        if include_subjects is not None and subject not in include_subjects:
            continue

        avg_fif = subj_dir / f"{subject}_desc-hep-{condition}_avg.fif"
        stats_json = subj_dir / f"{subject}_desc-hep-{condition}_stats.json"

        if not avg_fif.exists():
            # Include subject with missing data
            subject_data[subject] = {
                "evoked": None,
                "stats": {},
            }
            continue

        evoked = mne.read_evokeds(str(avg_fif))[0]

        stats = {}
        if stats_json.exists():
            with open(stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)

        subject_data[subject] = {
            "evoked": evoked,
            "stats": stats,
        }

    subject_order = sorted(subject_data.keys())
    return subject_data, subject_order


def compute_grand_average(subject_data: Dict[str, dict]) -> dict:
    """Compute grand-average evoked from all subjects with data."""
    import mne

    if not subject_data:
        raise ValueError("No subject data to average")

    # Filter out subjects with no evoked data (None)
    evokeds = [data["evoked"] for data in subject_data.values() if data["evoked"] is not None]
    if not evokeds:
        raise ValueError("No subjects with evoked data available")
    
    grand_avg = mne.grand_average(evokeds)
    return {"grand_avg": grand_avg, "n_subjects": len(evokeds)}


def compute_stats_per_channel(
    subject_data: Dict[str, dict],
    subject_order: List[str],
    outlier_threshold_sd: float = 2.0,
) -> dict:
    """Compute per-channel statistics and detect outliers.

    Returns
    -------
    stats : dict
        Keys: channel names. Values: {
            "mean_uv": float,
            "std_uv": float,
            "min_uv": float,
            "max_uv": float,
            "min_subject": str,
            "max_subject": str,
            "outlier_subjects": [(subject, value_uv, n_sd_from_mean), ...],
        }
    """
    stats: dict = {}

    # Get all channel names from first subject WITH data
    first_evoked = None
    for subject in subject_order:
        if subject_data[subject]["evoked"] is not None:
            first_evoked = subject_data[subject]["evoked"]
            break
    
    if first_evoked is None:
        return {}  # No subjects with data
    
    ch_names = first_evoked.ch_names

    for ch_name in ch_names:
        ch_values_uv: dict = {}  # subject → peak (max abs value)

        for subject in subject_order:
            evoked = subject_data[subject]["evoked"]
            if evoked is None:
                ch_values_uv[subject] = np.nan
                continue
            
            ch_idx = evoked.ch_names.index(ch_name)
            trace = evoked.data[ch_idx] * 1e6  # convert to µV
            peak = np.nanmax(np.abs(trace)) if np.any(np.isfinite(trace)) else np.nan
            ch_values_uv[subject] = peak

        valid_vals = [v for v in ch_values_uv.values() if np.isfinite(v)]
        if not valid_vals:
            stats[ch_name] = {
                "mean_uv": np.nan,
                "std_uv": np.nan,
                "min_uv": np.nan,
                "max_uv": np.nan,
                "min_subject": None,
                "max_subject": None,
                "outlier_subjects": [],
            }
            continue

        mean_val = float(np.mean(valid_vals))
        std_val = float(np.std(valid_vals))
        min_val = float(np.min(valid_vals))
        max_val = float(np.max(valid_vals))
        min_subj = [s for s, v in ch_values_uv.items() if v == min_val][0]
        max_subj = [s for s, v in ch_values_uv.items() if v == max_val][0]

        # Per-channel outlier detection
        outliers: List[Tuple[str, float, float]] = []
        if std_val > 0:
            for subject, val in ch_values_uv.items():
                if not np.isfinite(val):
                    continue
                n_sd = abs((val - mean_val) / std_val)
                if n_sd > outlier_threshold_sd:
                    outliers.append((subject, float(val), float(n_sd)))

        stats[ch_name] = {
            "mean_uv": mean_val,
            "std_uv": std_val,
            "min_uv": min_val,
            "max_uv": max_val,
            "min_subject": min_subj,
            "max_subject": max_subj,
            "outlier_subjects": sorted(outliers, key=lambda x: -x[2])[:10],  # top 10
        }

    return stats


def _extract_n_epochs(stats: dict) -> int | float:
    for key in ("n_epochs_accepted", "n_accepted", "n_epochs", "accepted_epochs"):
        if key in stats:
            try:
                return int(stats[key])
            except Exception:
                pass
    return np.nan


def build_subject_long_rows(
    subject_data: Dict[str, dict],
    subject_order: List[str],
    condition: str,
) -> List[dict]:
    """Build long-format rows: subject, condition, channel, window, mean, sd, n_epochs.
    
    Missing data (evoked=None) results in empty string cells for mean/sd/n_epochs.
    """
    rows: List[dict] = []

    for subject in subject_order:
        evoked_obj = subject_data[subject]["evoked"]
        subj_stats = subject_data[subject].get("stats", {})
        n_epochs = _extract_n_epochs(subj_stats)

        # If no evoked data, skip window processing (all cells empty)
        if evoked_obj is None:
            for win_label in HEP_WINDOWS_SEC.keys():
                for ch in HEP_CHANNELS:
                    rows.append({
                        "subject": subject,
                        "condition": condition,
                        "channel": ch,
                        "window": win_label,
                        "mean": "",
                        "sd": "",
                        "n_epochs": "",
                    })

                rows.append({
                    "subject": subject,
                    "condition": condition,
                    "channel": HEP_ROI_NAME,
                    "window": win_label,
                    "mean": "",
                    "sd": "",
                    "n_epochs": "",
                })
            continue

        # Subject has data
        evoked = evoked_obj.copy().pick("eeg")
        times = evoked.times

        for win_label, (t0, t1) in HEP_WINDOWS_SEC.items():
            mask = (times >= t0) & (times <= t1)

            # Individual channels
            for ch in HEP_CHANNELS:
                if ch in evoked.ch_names and np.any(mask):
                    trace_uv = evoked.get_data(picks=[ch])[0] * 1e6
                    vals = trace_uv[mask]
                    mean_v = float(np.nanmean(vals)) if vals.size else ""
                    sd_v = float(np.nanstd(vals)) if vals.size else ""
                else:
                    mean_v = ""
                    sd_v = ""

                rows.append({
                    "subject": subject,
                    "condition": condition,
                    "channel": ch,
                    "window": win_label,
                    "mean": mean_v,
                    "sd": sd_v,
                    "n_epochs": n_epochs if n_epochs == n_epochs else "",  # use n_epochs or empty
                })

            # ROI channel average
            roi_present = [ch for ch in HEP_ROI_CHANNELS if ch in evoked.ch_names]
            if roi_present and np.any(mask):
                roi_data_uv = evoked.get_data(picks=roi_present) * 1e6
                roi_trace_uv = np.nanmean(roi_data_uv, axis=0)
                vals = roi_trace_uv[mask]
                roi_mean = float(np.nanmean(vals)) if vals.size else ""
                roi_sd = float(np.nanstd(vals)) if vals.size else ""
            else:
                roi_mean = ""
                roi_sd = ""

            rows.append({
                "subject": subject,
                "condition": condition,
                "channel": HEP_ROI_NAME,
                "window": win_label,
                "mean": roi_mean,
                "sd": roi_sd,
                "n_epochs": n_epochs if n_epochs == n_epochs else "",
            })

    return rows


def write_long_csv(path: Path, rows: List[dict]) -> None:
    cols = ["subject", "condition", "channel", "window", "mean", "sd", "n_epochs"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _save_group_report(
    report_path: Path,
    grand_avg,
    subject_data: Dict[str, dict],
    subject_order: List[str],
    stats_per_channel: dict,
    condition: str,
    sham_grand_avg=None,
    sham_condition: str | None = None,
) -> None:
    """Generate multi-page group HEP report."""
    import mne as _mne

    n_subjects = len(subject_order)
    cond_label = f"Group HEP · n={n_subjects} · condition={condition}"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(report_path) as pdf:
        # ── Page 1: Summary ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")

        summary_lines = [
            f"Condition:      {condition}",
            f"N subjects:     {n_subjects}",
            f"Subjects:       {', '.join(subject_order[:10])}",
        ]
        if n_subjects > 10:
            summary_lines.append(f"                + {n_subjects - 10} more")

        total_outlier_pairs = sum(
            len(ch_stats["outlier_subjects"]) for ch_stats in stats_per_channel.values()
        )
        summary_lines += ["", f"Total outlier subject-channel pairs: {total_outlier_pairs}"]

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

        # ── Page 2: Grand-average butterfly ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 5))
        times = grand_avg.times * 1000  # ms
        eeg_picks = _mne.pick_types(grand_avg.info, eeg=True)
        data_uv = grand_avg.data[eeg_picks] * 1e6

        for trace in data_uv:
            ax.plot(times, trace, color="steelblue", lw=0.6, alpha=0.5)

        mean_trace = np.nanmean(data_uv, axis=0)
        ax.plot(times, mean_trace, color="navy", lw=2.0, label="Grand mean")

        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="red", lw=0.8, ls="--", label="R-peak")

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"Grand-average HEP butterfly — {cond_label}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Per-channel scalp layout (grand-average) ────────────────
        ch_positions = [
            ["", "Fp1", "", "Fp2", ""],
            ["F7", "F3", "Fz", "F4", "F8"],
            ["T3", "C3", "Cz", "C4", "T4"],
            ["T5", "P3", "Pz", "P4", "T6"],
            ["", "O1", "", "O2", ""],
        ]

        fig, axes = plt.subplots(5, 5, figsize=(12, 10), sharex=True, sharey=True)
        fig.patch.set_facecolor("white")

        ga_eeg = grand_avg.copy().pick("eeg")
        ch_to_idx = {ch: i for i, ch in enumerate(ga_eeg.ch_names)}
        t_ms = ga_eeg.times * 1000
        y_lim = np.nanmax(np.abs(ga_eeg.data * 1e6))
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

                trace = ga_eeg.data[idx] * 1e6
                ax_c.set_facecolor("#f8f9fa")
                ax_c.axhline(0, color="#999", lw=0.6, ls="--")
                ax_c.axvline(0, color="#d62728", lw=0.8, ls="--")

                if np.all(~np.isfinite(trace)):
                    ax_c.text(0.5, 0.5, "BAD", transform=ax_c.transAxes,
                              ha="center", va="center", fontsize=8, color="#b22222")
                else:
                    ax_c.plot(t_ms, trace, color="#1f77b4", lw=1.2)

                ax_c.set_title(ch, fontsize=9, pad=2)
                ax_c.set_xlim(t_ms[0], t_ms[-1])
                ax_c.set_ylim(-y_lim, y_lim)
                ax_c.tick_params(labelsize=7, length=2)

        fig.suptitle(f"Grand-average HEP (scalp layout) — {cond_label}", fontsize=13, y=0.98)
        fig.text(0.5, 0.03, "Time (ms)", ha="center", fontsize=10)
        fig.text(0.03, 0.5, "Amplitude (µV)", va="center", rotation=90, fontsize=10)
        fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 4: Topographic maps (grand-average) ───────────────────────
        try:
            valid_topo = [0.2, 0.3, 0.4, 0.5, 0.6]
            ga_topo = grand_avg.copy().pick("eeg")
            valid_ch_mask = np.all(np.isfinite(ga_topo.data), axis=1)
            if np.any(valid_ch_mask):
                valid_ch = [ch for ch, ok in zip(ga_topo.ch_names, valid_ch_mask) if ok]
                ga_topo.pick(valid_ch)

                fig = ga_topo.plot_topomap(
                    times=valid_topo,
                    show=False,
                    colorbar=True,
                    units="µV",
                    scalings={"eeg": 1e6},
                )
                fig.suptitle(f"Grand-average topomaps — {cond_label}", fontsize=11, y=1.02)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            print(f"  Warning: topomap generation failed: {exc}")

        # ── Page 5: Grand-average SHAM HEP (scalp layout) ───────────────────
        if sham_grand_avg is not None:
            fig, axes = plt.subplots(5, 5, figsize=(12, 10), sharex=True, sharey=True)
            fig.patch.set_facecolor("white")

            sham_eeg = sham_grand_avg.copy().pick("eeg")
            ch_to_idx = {ch: i for i, ch in enumerate(sham_eeg.ch_names)}
            t_ms = sham_eeg.times * 1000
            y_lim = np.nanmax(np.abs(sham_eeg.data * 1e6))
            y_lim = max(3.0, float(y_lim) if np.isfinite(y_lim) else 10.0)

            ch_positions = [
                ["", "Fp1", "", "Fp2", ""],
                ["F7", "F3", "Fz", "F4", "F8"],
                ["T3", "C3", "Cz", "C4", "T4"],
                ["T5", "P3", "Pz", "P4", "T6"],
                ["", "O1", "", "O2", ""],
            ]

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

                    trace = sham_eeg.data[idx] * 1e6
                    ax_c.set_facecolor("#f8f9fa")
                    ax_c.axhline(0, color="#999", lw=0.6, ls="--")
                    ax_c.axvline(0, color="#d62728", lw=0.8, ls="--")

                    if np.all(~np.isfinite(trace)):
                        ax_c.text(0.5, 0.5, "BAD", transform=ax_c.transAxes,
                                  ha="center", va="center", fontsize=8, color="#b22222")
                    else:
                        ax_c.plot(t_ms, trace, color="#1f77b4", lw=1.2)

                    ax_c.set_title(ch, fontsize=9, pad=2)
                    ax_c.set_xlim(t_ms[0], t_ms[-1])
                    ax_c.set_ylim(-y_lim, y_lim)
                    ax_c.tick_params(labelsize=7, length=2)

            fig.suptitle(
                f"Grand-average SHAM HEP (scalp layout) — Group HEP · condition={sham_condition}",
                fontsize=13,
                y=0.98,
            )
            fig.text(0.5, 0.03, "Time (ms)", ha="center", fontsize=10)
            fig.text(0.03, 0.5, "Amplitude (µV)", va="center", rotation=90, fontsize=10)
            fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 6: Grand-average topomaps Group SHAM HEP ──────────────
            try:
                valid_topo = [0.2, 0.3, 0.4, 0.5, 0.6]
                sham_topo = sham_grand_avg.copy().pick("eeg")
                valid_ch_mask = np.all(np.isfinite(sham_topo.data), axis=1)
                if np.any(valid_ch_mask):
                    valid_ch = [ch for ch, ok in zip(sham_topo.ch_names, valid_ch_mask) if ok]
                    sham_topo.pick(valid_ch)

                    fig = sham_topo.plot_topomap(
                        times=valid_topo,
                        show=False,
                        colorbar=True,
                        units="µV",
                        scalings={"eeg": 1e6},
                    )
                    fig.suptitle(
                        f"Grand-average topomaps Group SHAM HEP — condition={sham_condition}",
                        fontsize=11,
                        y=1.02,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            except Exception as exc:
                print(f"  Warning: sham topomap generation failed: {exc}")

        # ── Final page: outlier counts summary only ────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ch_names_sorted = sorted(stats_per_channel.keys())
        n_outliers_per_ch = [
            len(stats_per_channel[ch]["outlier_subjects"]) for ch in ch_names_sorted
        ]
        x_pos = np.arange(len(ch_names_sorted))

        axes[0].bar(x_pos, n_outliers_per_ch, color="salmon")
        axes[0].set_ylabel("Number of outlier subjects")
        axes[0].set_title(f"Outliers per channel (threshold: {OUTLIER_THRESHOLD_SD} SD)")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(ch_names_sorted, rotation=45, fontsize=8)
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].axis("off")
        outlier_summary = [
            f"Channels with outliers: {sum(1 for o in n_outliers_per_ch if o > 0)}",
            f"Max outliers per channel: {max(n_outliers_per_ch) if n_outliers_per_ch else 0}",
            f"Total subject-channel outlier pairs: {sum(n_outliers_per_ch)}",
        ]
        axes[1].text(
            0.05, 0.5, "\n".join(outlier_summary),
            fontsize=11,
            fontfamily="monospace",
            va="center",
        )

        fig.suptitle(f"Per-channel outlier summary — {cond_label}", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  Saved group HEP report → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conditions = CONDITION if isinstance(CONDITION, list) else [CONDITION]
    all_long_rows: List[dict] = []

    for cond in conditions:
        print(f"\nGroup HEP analysis — condition={cond}")
        print(f"{'='*60}")

        sham_condition = f"sham-{cond}"

        # ── Load all subject HEPs ────────────────────────────────────────────
        print("Loading subject HEP evoked files…")
        subject_data, subject_order = load_subject_heps(
            HEP_DIR,
            cond,
            include_subjects=INCLUDE_SUBJECTS,
            exclude_subjects=EXCLUDE_SUBJECTS,
        )

        if not subject_data:
            print(f"ERROR: No subject data found for condition={cond} — skipping.")
            continue

        print(f"Loaded {len(subject_order)} subjects: {subject_order}")

        # ── Load SHAM data for this condition (optional) ───────────────────
        sham_subject_data, sham_subject_order = load_subject_heps(
            HEP_DIR,
            sham_condition,
            include_subjects=INCLUDE_SUBJECTS,
            exclude_subjects=EXCLUDE_SUBJECTS,
        )
        sham_grand_avg = None
        if sham_subject_data:
            print(f"Loaded {len(sham_subject_order)} sham subjects for {sham_condition}")
            sham_grand_avg = compute_grand_average(sham_subject_data)["grand_avg"]
        else:
            print(f"No sham data found for {sham_condition} (continuing without sham pages)")

        # ── Compute grand average ───────────────────────────────────────────
        print("Computing grand average…")
        ga_data = compute_grand_average(subject_data)
        grand_avg = ga_data["grand_avg"]

        # ── Per-channel statistics ──────────────────────────────────────────
        print("Computing per-channel statistics…")
        stats_per_channel = compute_stats_per_channel(
            subject_data,
            subject_order,
            outlier_threshold_sd=OUTLIER_THRESHOLD_SD,
        )

        total_outliers = sum(
            len(s["outlier_subjects"]) for s in stats_per_channel.values()
        )
        print(f"Outliers detected: {total_outliers} subject-channel pairs")

        # ── Output directory ────────────────────────────────────────────────
        GROUP_HEP_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"group_desc-hep-{cond}"

        avg_path = GROUP_HEP_DIR / f"{tag}_avg.fif"
        grand_avg.save(str(avg_path), overwrite=True)
        print(f"Saved grand-average → {avg_path.name}")

        stats_path = GROUP_HEP_DIR / f"{tag}_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_per_channel, f, indent=2, default=str)
        print(f"Saved stats → {stats_path.name}")

        # ── Long-format subject table ──────────────────────────────────────
        long_rows = build_subject_long_rows(subject_data, subject_order, cond)
        long_path = GROUP_HEP_DIR / f"{tag}_subjects_long.csv"
        write_long_csv(long_path, long_rows)
        print(f"Saved long-format table → {long_path.name}")
        all_long_rows.extend(long_rows)

        # ── Save report ─────────────────────────────────────────────────────
        report_path = GROUP_HEP_DIR / f"{tag}_report.pdf"
        _save_group_report(
            report_path=report_path,
            grand_avg=grand_avg,
            subject_data=subject_data,
            subject_order=subject_order,
            stats_per_channel=stats_per_channel,
            condition=cond,
            sham_grand_avg=sham_grand_avg,
            sham_condition=sham_condition,
        )

        # ── Save provenance JSON ────────────────────────────────────────────
        provenance = {
            "script": "eeg_hep_group.py",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "condition": cond,
            "n_subjects": len(subject_order),
            "subjects": subject_order,
            "outlier_threshold_sd": OUTLIER_THRESHOLD_SD,
            "inputs": {
                "hep_dir": str(HEP_DIR),
                "sham_condition": sham_condition,
            },
            "outputs": {
                "avg_fif": str(avg_path),
                "stats_json": str(stats_path),
                "report_pdf": str(report_path),
                "subjects_long_csv": str(long_path),
            },
            "n_outlier_pairs": total_outliers,
        }
        prov_path = GROUP_HEP_DIR / f"{tag}_processing.json"
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)
        print(f"Saved provenance → {prov_path.name}")

    if all_long_rows:
        combined_long_path = GROUP_HEP_DIR / "group_desc-hep-subjects_long.csv"
        write_long_csv(combined_long_path, all_long_rows)
        print(f"Saved combined long-format table → {combined_long_path.name}")

    print(f"\n✓ Group HEP analysis complete.")
    print(f"  Output folder: {GROUP_HEP_DIR}")


if __name__ == "__main__":
    main()

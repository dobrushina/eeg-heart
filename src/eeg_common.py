import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from eeg_config import CONVENTIONAL_EEG_ORDER, NOTCH_FREQS, L_FREQ, H_FREQ, DERIVATIVES_DIR, HEP_PIPELINE_DIR


def set_channel_types_from_names(raw, verbose: bool = False):
    """Set channel types for common non-EEG channels based on name heuristics."""
    ch_type_map = {}
    for ch in raw.ch_names:
        up = ch.upper()
        if "ECG" in up:
            ch_type_map[ch] = "ecg"
        elif "FPG" in up or up.endswith("GND") or up.endswith("G"):
            ch_type_map[ch] = "misc"
        elif "EOG" in up:
            ch_type_map[ch] = "eog"

    if ch_type_map:
        raw.set_channel_types(ch_type_map)
        if verbose:
            print("Set channel types:", ch_type_map)

    return ch_type_map


def rename_channels_drop_reference(raw, verbose: bool = False):
    """Drop channel reference suffixes (e.g., Fz-A1A2 -> Fz) while keeping names unique."""
    orig_names = raw.ch_names[:]
    mapping = {}
    used = set(orig_names)
    for ch in orig_names:
        new = re.sub(r"[-_:].*$", "", ch)
        if new == ch:
            continue
        if new in used:
            idx = 1
            candidate = f"{new}_{idx}"
            while candidate in used:
                idx += 1
                candidate = f"{new}_{idx}"
            new = candidate
        mapping[ch] = new
        used.add(new)

    if mapping:
        raw.rename_channels(mapping)
        if verbose:
            print("Renamed channels:", mapping)

    return mapping


def preprocess_raw_for_pipeline(raw, notch_freqs, l_freq, h_freq, verbose: bool = False):
    """Apply common preprocessing used by auto-denoising and manual-correction flows."""
    set_channel_types_from_names(raw, verbose=verbose)
    rename_channels_drop_reference(raw, verbose=verbose)

    try:
        raw.set_montage("standard_1020", on_missing="warn")
    except Exception as exc:
        print("Warning setting montage:", exc)

    try:
        raw.set_eeg_reference("average")
    except Exception as exc:
        print("Warning setting average EEG reference:", exc)

    raw.notch_filter(notch_freqs)
    raw.filter(l_freq=l_freq, h_freq=h_freq)


def sort_channel_names_conventional(ch_names: Sequence[str]) -> List[str]:
    """Sort channel names using conventional 10-20 order; keep unknown channels last."""
    preferred = [ch for ch in CONVENTIONAL_EEG_ORDER if ch in ch_names]
    extras = [ch for ch in ch_names if ch not in CONVENTIONAL_EEG_ORDER]
    return preferred + extras


def sort_eeg_picks_conventional(raw, eeg_picks) -> Tuple[np.ndarray, List[str]]:
    """Return EEG picks sorted in conventional order with matching channel names."""
    names = [raw.ch_names[p] for p in eeg_picks]
    ordered_names = sort_channel_names_conventional(names)
    name_to_pick = {raw.ch_names[p]: p for p in eeg_picks}
    ordered_picks = np.array([name_to_pick[ch] for ch in ordered_names], dtype=int)
    return ordered_picks, ordered_names


def save_eeg_ecg_report(
    report_path: Path,
    raw_before_ica,
    raw_denoised,
    raw,
    ecg_ch: str,
    events,
    bad_times: list,
    annotations: list,
    ica_exclude: list,
    subject_name: str,
    page_duration: float = 10.0,
    eeg_gain_uv: float = 50.0,
    ecg_gain_uv: float = 500.0,
):
    """
    Generate a multi-page PDF overlaying raw and denoised EEG with ECG / R-peaks.

    Parameters
    ----------
    report_path : Path
        Output PDF path.
    raw_before_ica : mne.io.Raw
        Raw data *before* ICA (grey traces).
    raw_denoised : mne.io.Raw
        Data *after* ICA removal (blue traces).
    raw : mne.io.Raw
        Preprocessed raw used for annotation / montage metadata.
    ecg_ch : str
        Name of the ECG channel in ``raw``.
    events : ndarray
        R-peak events array (shape N×3).
    bad_times : list of (onset, duration) tuples
        Bad segment intervals in seconds.
    annotations : list
        Non-BAD annotation dicts with keys ``onset``, ``description``.
    ica_exclude : list of int
        Final ICA exclusion list (printed in footer).
    subject_name : str
        Label used in page titles (e.g., ``"sub-eeg-1.edf"``).
    page_duration : float
        Seconds shown per page.
    eeg_gain_uv : float
        Amplitude normalisation for EEG display (µV per unit).
    ecg_gain_uv : float
        Amplitude normalisation for ECG display (µV per unit).
    """
    from matplotlib.backends.backend_pdf import PdfPages

    import mne as _mne

    eeg_picks = _mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        print("No EEG channels to plot in report.")
        return

    eeg_picks, eeg_names = sort_eeg_picks_conventional(raw, eeg_picks)
    raw_eeg_uv = raw_before_ica.get_data(picks=eeg_picks) * 1e6
    den_eeg_uv = raw_denoised.get_data(picks=eeg_picks) * 1e6

    sfreq = raw.info["sfreq"]
    total_sec = raw.n_times / sfreq
    n_pages = int(np.ceil(total_sec / page_duration))
    ecg_full_uv = raw.get_data(picks=[raw.ch_names.index(ecg_ch)])[0] * 1e6
    r_times_all = events[:, 0].astype(int) / sfreq

    colors_ann = {"eyes open": "green", "eyes closed": "red"}

    montage_obj = raw.get_montage()
    montage_name = (
        montage_obj.kind
        if montage_obj is not None and hasattr(montage_obj, "kind")
        else "standard_1020"
    )
    footer = (
        f"Montage: {montage_name}    Notch: {NOTCH_FREQS} Hz    "
        f"Bandpass: {L_FREQ}-{H_FREQ} Hz    ICA excluded: {ica_exclude}    "
        f"Display gains: EEG={eeg_gain_uv:.0f}µV, ECG={ecg_gain_uv:.0f}µV"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(report_path) as pdf:
        for page in range(n_pages):
            t0 = page * page_duration
            t1 = min((page + 1) * page_duration, total_sec)
            s0 = int(t0 * sfreq)
            s1 = int(t1 * sfreq)
            times = np.arange(s0, s1) / sfreq
            duration = t1 - t0

            fig, (ax_eeg, ax_ecg) = plt.subplots(
                2, 1, figsize=(11, 8), sharex=True,
                gridspec_kw={"height_ratios": [5, 0.6]},
            )

            eeg_spacing = 1.2
            eeg_offsets = np.arange(len(eeg_names))[::-1] * eeg_spacing
            for i in range(len(eeg_names)):
                ax_eeg.plot(times, raw_eeg_uv[i, s0:s1] / eeg_gain_uv + eeg_offsets[i],
                            color="0.50", lw=0.75, alpha=0.9)
                ax_eeg.plot(times, den_eeg_uv[i, s0:s1] / eeg_gain_uv + eeg_offsets[i],
                            color="C0", lw=0.75)

            ax_eeg.plot([], [], color="0.50", lw=1.2, label="Raw")
            ax_eeg.plot([], [], color="C0", lw=1.2, label="Denoised")
            ax_eeg.legend(loc="upper right", fontsize=8)
            ax_eeg.set_yticks(eeg_offsets)
            ax_eeg.set_yticklabels(eeg_names, fontsize=7)
            ax_eeg.set_ylim(-eeg_spacing, eeg_offsets[0] + eeg_spacing)
            ax_eeg.set_xlim(t0, t0 + page_duration)
            ax_eeg.grid(True, axis="x", lw=0.3, alpha=0.5)

            for bad_onset, bad_dur in bad_times:
                bad_end = bad_onset + bad_dur
                if bad_end > t0 and bad_onset < (t0 + page_duration):
                    shade_s = max(bad_onset, t0)
                    shade_e = min(bad_end, t0 + page_duration)
                    ax_eeg.axvspan(shade_s, shade_e, color="red", alpha=0.08, zorder=0)
                    ax_ecg.axvspan(shade_s, shade_e, color="red", alpha=0.08, zorder=0)

            for ann in annotations:
                ann_time = ann["onset"]
                ann_desc = ann["description"].strip()
                if t0 <= ann_time < (t0 + page_duration):
                    color = colors_ann.get(ann_desc.lower(), "gray")
                    ax_eeg.axvline(ann_time, color=color, lw=1.5, alpha=0.7, linestyle="--")
                    ax_eeg.text(ann_time, ax_eeg.get_ylim()[1] * 0.95, ann_desc,
                                fontsize=6, rotation=90, va="top", ha="right", color=color)

            ax_eeg.set_title(
                f"Raw + denoised EEG / ECG report — {subject_name} (page {page + 1}/{n_pages})"
            )
            ax_eeg.set_ylabel("EEG")

            x_bar = t0 + duration * 0.01
            eeg_bar_y0 = -0.6
            ax_eeg.plot([x_bar, x_bar], [eeg_bar_y0, eeg_bar_y0 + 1], color="k", lw=2)
            ax_eeg.text(x_bar + duration * 0.01, eeg_bar_y0 + 0.5,
                        f"{eeg_gain_uv:.0f} µV", fontsize=7, va="center")

            ecg_seg = ecg_full_uv[s0:s1] / ecg_gain_uv
            ax_ecg.plot(times, ecg_seg, color="C3", lw=0.8)
            ax_ecg.axhline(0, color="0.4", lw=0.6, ls="--")

            mask = (r_times_all >= t0) & (r_times_all < t1)
            r_t_in = r_times_all[mask]
            r_samps = np.clip((r_t_in * sfreq).astype(int), 0, len(ecg_full_uv) - 1)
            ax_ecg.scatter(r_t_in, ecg_full_uv[r_samps] / ecg_gain_uv, color="red", s=10, zorder=3)

            ax_ecg.set_ylabel("ECG")
            ax_ecg.grid(True, axis="x", lw=0.3, alpha=0.5)
            ax_ecg.set_ylim(-0.8, 3.2)

            ecg_bar_y0 = -2.0
            ax_ecg.plot([x_bar, x_bar], [ecg_bar_y0, ecg_bar_y0 + 1], color="k", lw=2)
            ax_ecg.text(x_bar + duration * 0.01, ecg_bar_y0 + 0.5,
                        f"{ecg_gain_uv:.0f} µV", fontsize=7, va="center")

            ax_ecg.set_xlabel("Time (s)")
            fig.text(0.5, 0.01, footer, ha="center", fontsize=8)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved EEG/ECG report to {report_path}")


# ---------------------------------------------------------------------------
# HEP helpers
# ---------------------------------------------------------------------------

def load_rpeaks_csv(path: Path) -> np.ndarray:
    """Load R-peak times (seconds) from a ``*_desc-rpeaks_events.csv`` file.

    The CSV is expected to have a header row with at least a ``time_sec``
    column (as written by ``eeg_auto_denoising.py``).

    Returns
    -------
    np.ndarray
        1-D array of R-peak onset times in seconds.
    """
    import csv

    times = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_sec"]))
    return np.asarray(times, dtype=float)


def load_condition_segments(
    conditions_csv: Path,
    subject: str,
    condition: str,
) -> List[Tuple[float, float]]:
    """Return (start_sec, end_sec) intervals for *subject* / *condition*.

    Reads ``manual/conditions.csv`` (columns: ``subject,condition,start,end``).
    Multiple rows for the same subject+condition are all returned.

    Returns
    -------
    list of (float, float)
        Possibly empty if the subject/condition combination is not found.
    """
    import csv

    segments: List[Tuple[float, float]] = []
    with open(conditions_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["subject"].strip() == subject and row["condition"].strip() == condition:
                segments.append((float(row["start"]), float(row["end"])))
    return segments


def filter_rpeaks_to_segments(
    rpeaks_sec: np.ndarray,
    segments: List[Tuple[float, float]],
) -> np.ndarray:
    """Keep only R-peaks that fall inside one of the (start, end) segments.

    Parameters
    ----------
    rpeaks_sec : np.ndarray
        R-peak times in seconds.
    segments : list of (float, float)
        Each tuple is a half-open interval ``[start, end)``.

    Returns
    -------
    np.ndarray
        Filtered subset of *rpeaks_sec*.
    """
    if len(segments) == 0 or len(rpeaks_sec) == 0:
        return np.array([], dtype=float)

    mask = np.zeros(len(rpeaks_sec), dtype=bool)
    for start, end in segments:
        mask |= (rpeaks_sec >= start) & (rpeaks_sec < end)
    return rpeaks_sec[mask]


def rpeaks_to_mne_events(rpeaks_sec: np.ndarray, sfreq: float) -> np.ndarray:
    """Convert R-peak times (seconds) to an MNE events array of shape (N, 3).

    The event ID is 999 ("R_peak"), onset column is in samples, previous
    event column is 0.

    Parameters
    ----------
    rpeaks_sec : np.ndarray
        R-peak onset times in seconds.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    samples = np.round(rpeaks_sec * sfreq).astype(int)
    events = np.zeros((len(samples), 3), dtype=int)
    events[:, 0] = samples
    events[:, 2] = 999  # R_peak event ID
    return events


def reject_epochs_by_electrode(
    epochs,
    thresh_uv: float,
    excluded_channels: Optional[Sequence[str]] = None,
) -> Tuple:
    """Build channel-wise rejection mask from peak-to-peak amplitudes.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (must already be baseline-corrected or at least loaded).
    thresh_uv : float
        Peak-to-peak amplitude threshold in µV.
    excluded_channels : sequence of str | None
        EEG channel names to skip entirely (e.g., manual bad electrodes).

    Returns
    -------
    epochs_kept : mne.Epochs
        Unmodified copy of *epochs* (no global epoch drops are applied).
    rejection_log : dict
        Keys include:
        ``n_total`` (all epochs),
        ``n_accepted`` (same as n_total),
        ``n_rejected`` (always 0, because no global epoch rejection),
        ``excluded_channels``,
        ``by_channel`` (dict channel→number of bad epochs for that channel),
        ``good_counts_by_channel`` (dict channel→number of usable epochs),
        ``bad_mask`` (list[list[bool]]; shape n_epochs×n_used_channels),
        ``used_channels`` (channel order corresponding to ``bad_mask``).
    """
    import mne as _mne

    excluded = set(excluded_channels or [])
    eeg_picks_all = _mne.pick_types(epochs.info, eeg=True)
    used_picks = np.array(
        [p for p in eeg_picks_all if epochs.ch_names[p] not in excluded],
        dtype=int,
    )
    used_names = [epochs.ch_names[p] for p in used_picks]

    if len(used_picks) == 0:
        rejection_log = {
            "n_total": int(len(epochs)),
            "n_accepted": int(len(epochs)),
            "n_rejected": 0,
            "excluded_channels": sorted(excluded),
            "by_channel": {},
            "good_counts_by_channel": {},
            "bad_mask": [],
            "used_channels": [],
        }
        return epochs.copy(), rejection_log

    data_uv = epochs.get_data(picks=used_picks) * 1e6  # (n_epochs, n_used_ch, n_times)
    ptp = np.ptp(data_uv, axis=-1)  # (n_epochs, n_used_ch)
    bad_mask = ptp > thresh_uv

    by_channel: dict = {}
    good_counts_by_channel: dict = {}
    for ch_idx, ch_name in enumerate(used_names):
        bad_count = int(np.sum(bad_mask[:, ch_idx]))
        by_channel[ch_name] = bad_count
        good_counts_by_channel[ch_name] = int(len(epochs) - bad_count)

    rejection_log = {
        "n_total": int(len(epochs)),
        "n_accepted": int(len(epochs)),
        "n_rejected": 0,
        "excluded_channels": sorted(excluded),
        "by_channel": by_channel,
        "good_counts_by_channel": good_counts_by_channel,
        "bad_mask": bad_mask.tolist(),
        "used_channels": used_names,
    }
    return epochs.copy(), rejection_log


def crop_to_state_annotations(raw):
    """Crop *raw* to the interval between 'State record' and 'State stop'.

    Searches annotations (case-insensitive, stripped) for those two markers.
    If either is missing the original *raw* is returned unchanged and a
    warning is printed.

    Returns
    -------
    mne.io.Raw
        Cropped (or original) Raw object.  The operation is in-place if
        MNE's ``crop`` is in-place; callers should use the returned value.
    found : bool
        ``True`` if both markers were found and cropping was applied.
    """
    record_t: Optional[float] = None
    stop_t: Optional[float] = None

    for ann in raw.annotations:
        desc = ann["description"].strip().lower()
        if desc == "state record":
            record_t = ann["onset"]
        elif desc == "state stop":
            stop_t = ann["onset"]

    if record_t is None or stop_t is None:
        missing = []
        if record_t is None:
            missing.append("'State record'")
        if stop_t is None:
            missing.append("'State stop'")
        print(f"  WARNING: annotation(s) {', '.join(missing)} not found — skipping crop.")
        return raw, False

    tmax = min(stop_t, raw.times[-1])
    raw_cropped = raw.copy().crop(tmin=record_t, tmax=tmax)
    return raw_cropped, True

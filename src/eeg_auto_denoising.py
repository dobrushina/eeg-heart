from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
import csv
import json
from datetime import datetime, timezone

from eeg_common import preprocess_raw_for_pipeline, sort_channel_names_conventional, sort_eeg_picks_conventional, save_eeg_ecg_report
from eeg_config import (
    NOTCH_FREQS,
    L_FREQ,
    H_FREQ,
    DERIVATIVES_DIR,
    AUTO_PIPELINE_DIR,
)

# Easy run mode toggle:
# - Set to True to process all EDF files in data/raw
# - Set to False to process only RAW_FILE below
RUN_ALL_FILES = True
RAW_FILE = Path("data/raw/sub-eeg-1.edf")


def detect_bad_segments(raw, eeg_picks, ecg_ch, sfreq, window_sec=2.0, 
                        eeg_rms_thresh=200, eeg_pp_thresh=1000, eeg_var_thresh=0.01,
                        ecg_rms_thresh=500, verbose=True):
    """
    Detect bad EEG/ECG segments using signal quality metrics.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object.
    eeg_picks : array
        EEG channel indices.
    ecg_ch : str
        ECG channel name.
    sfreq : float
        Sampling frequency.
    window_sec : float
        Sliding window duration in seconds.
    eeg_rms_thresh : float
        RMS threshold (µV) for EEG; above = bad.
    eeg_pp_thresh : float
        Peak-to-peak threshold (µV) for EEG; above = bad.
    eeg_var_thresh : float
        Variance threshold (µV²) for EEG; below = bad (flat).
    ecg_rms_thresh : float
        RMS threshold (µV) for ECG; above = bad.
    verbose : bool
        Print debug info.
    
    Returns
    -------
    bad_times : list of tuples
        (onset, duration) of bad segments in seconds.
    """
    data = raw.get_data() * 1e6  # convert to µV
    n_samples = data.shape[1]
    window_samples = int(window_sec * sfreq)
    
    bad_segments = []
    for start in range(0, n_samples, window_samples):
        end = min(start + window_samples, n_samples)
        is_bad = False
        bad_reason = []
        
        # Check EEG channels
        for idx in eeg_picks:
            seg = data[idx, start:end]
            rms = np.sqrt(np.mean(seg**2))
            pp = np.ptp(seg)
            var = np.var(seg)
            
            if rms > eeg_rms_thresh:
                is_bad = True
                bad_reason.append(f"EEG RMS {rms:.1f}>{eeg_rms_thresh}")
                break
            if pp > eeg_pp_thresh:
                is_bad = True
                bad_reason.append(f"EEG P-P {pp:.1f}>{eeg_pp_thresh}")
                break
            if var < eeg_var_thresh:
                is_bad = True
                bad_reason.append(f"EEG var {var:.3f}<{eeg_var_thresh}")
                break
        
        # Check ECG if not already bad
        if not is_bad and ecg_ch is not None:
            try:
                ecg_idx = raw.ch_names.index(ecg_ch)
                seg = data[ecg_idx, start:end]
                rms = np.sqrt(np.mean(seg**2))
                if rms > ecg_rms_thresh:
                    is_bad = True
                    bad_reason.append(f"ECG RMS {rms:.1f}>{ecg_rms_thresh}")
            except (ValueError, IndexError):
                pass
        
        if is_bad:
            onset = start / sfreq
            duration = (end - start) / sfreq
            bad_segments.append((onset, duration))
            if verbose:
                print(f"  Bad window at {onset:.2f}s: {', '.join(bad_reason)}")
    
    # Merge adjacent segments
    if bad_segments:
        merged = [bad_segments[0]]
        for onset, duration in bad_segments[1:]:
            last_onset, last_duration = merged[-1]
            if onset <= last_onset + last_duration + 0.1:  # 100ms tolerance
                merged[-1] = (last_onset, onset + duration - last_onset)
            else:
                merged.append((onset, duration))
        return merged
    return []


def save_figures_to_pdf(pdf, fig_or_figs):
    """Save a matplotlib figure or list of figures to a PdfPages object."""
    if isinstance(fig_or_figs, list):
        for fig in fig_or_figs:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
    else:
        if fig_or_figs is not None:
            pdf.savefig(fig_or_figs)
            plt.close(fig_or_figs)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _scores_to_abs_vector(scores, n_components):
    """Convert MNE bad-component scores to abs(score) vector of length n_components."""
    if scores is None:
        return np.zeros(n_components, dtype=float)

    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 0:
        return np.zeros(n_components, dtype=float)

    # Typical shape is (n_components,). If 2D (e.g., multi-target), collapse by max abs.
    if arr.ndim == 1:
        vec = np.abs(arr)
    else:
        # Ensure component axis is first; if not, transpose.
        if arr.shape[0] != n_components and arr.shape[-1] == n_components:
            arr = arr.T
        if arr.shape[0] == n_components:
            vec = np.max(np.abs(arr), axis=1)
        else:
            return np.zeros(n_components, dtype=float)

    if vec.shape[0] != n_components:
        return np.zeros(n_components, dtype=float)
    return vec


def main(raw_file):
    RAW_FILE = Path(raw_file)
    # Load raw data
    raw = mne.io.read_raw_edf(RAW_FILE, preload=True)
    print(raw.info)
    print(raw.ch_names) 

    # Shared preprocessing (channel typing/naming, montage/reference, notch+bandpass)
    preprocess_raw_for_pipeline(raw, notch_freqs=NOTCH_FREQS, l_freq=L_FREQ, h_freq=H_FREQ, verbose=True)

    # Try to find an ECG channel automatically
    ecg_ch = None
    for ch in raw.ch_names:
        if "ECG" in ch.upper():
            ecg_ch = ch
            break

    if ecg_ch is None:
        print("No ECG channel found. Skipping ECG-based HEP epoching.")
        return

    events, _, _ = mne.preprocessing.find_ecg_events(raw, ch_name=ecg_ch)
    print(f"Detected {len(events)} R-peaks")

    # Shared derivatives setup (BIDS-inspired)
    subj_id = RAW_FILE.stem  # e.g., "sub-eeg-3"
    outputs_root = Path(DERIVATIVES_DIR) / AUTO_PIPELINE_DIR / subj_id
    outputs_root.mkdir(parents=True, exist_ok=True)

    sfreq = raw.info["sfreq"]
    total_sec = raw.n_times / sfreq
    page_duration = 10.0  # seconds per page
    n_pages = int(np.ceil(total_sec / page_duration))

    # --- ICA denoising: blinks (Fp1/Fp2), horizontal eye movement (lateral proxy), ECG ---
    print("\nRunning ICA denoising...")
    raw_before_ica = raw.copy()
    raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(
        n_components=0.99,
        method="fastica",
        random_state=97,
        max_iter="auto",
    )
    ica.fit(raw_ica_fit, picks="eeg", reject_by_annotation=True)

    n_components = int(ica.n_components_)
    blink_inds = set()
    heog_inds = set()
    ecg_inds = set()
    blink_scores = np.zeros(n_components, dtype=float)
    heog_scores = np.zeros(n_components, dtype=float)
    ecg_scores = np.zeros(n_components, dtype=float)

    # Blink candidates from Fp1/Fp2 (no dedicated EOG channels)
    for ch in ["Fp1", "Fp2"]:
        if ch in raw_ica_fit.ch_names:
            inds, scores = ica.find_bads_eog(raw_ica_fit, ch_name=ch, threshold=3.0)
            blink_inds.update(inds[:2])  # keep conservative
            blink_scores = np.maximum(blink_scores, _scores_to_abs_vector(scores, n_components))

    # Horizontal eye movement via lateral bipolar proxy (F7-F8 preferred, fallback T3-T4)
    lateral_pairs = [("F7", "F8"), ("T3", "T4")]
    for anode, cathode in lateral_pairs:
        if anode in raw_ica_fit.ch_names and cathode in raw_ica_fit.ch_names:
            raw_heog = mne.set_bipolar_reference(
                raw_ica_fit.copy(),
                anode=anode,
                cathode=cathode,
                ch_name="HEOG_proxy",
                drop_refs=False,
                copy=True,
            )
            inds, scores = ica.find_bads_eog(raw_heog, ch_name="HEOG_proxy", threshold=3.0)
            heog_inds.update(inds[:2])
            heog_scores = np.maximum(heog_scores, _scores_to_abs_vector(scores, n_components))
            break

    # ECG-related components
    if ecg_ch in raw_ica_fit.ch_names:
        inds, scores = ica.find_bads_ecg(raw_ica_fit, ch_name=ecg_ch, method="correlation", threshold="auto")
        ecg_inds.update(inds[:2])
        ecg_scores = np.maximum(ecg_scores, _scores_to_abs_vector(scores, n_components))

    ica.exclude = sorted(blink_inds | heog_inds | ecg_inds)
    print(f"ICA exclude components: {ica.exclude}")
    print(f"  blink (Fp1/Fp2): {sorted(blink_inds)}")
    print(f"  horizontal (lateral proxy): {sorted(heog_inds)}")
    print(f"  ECG: {sorted(ecg_inds)}")

    # Save fitted ICA for deterministic/manual-override reuse without re-fitting
    ica_fif_path = outputs_root / f"{RAW_FILE.stem}_desc-fitted_ica.fif"
    try:
        ica.save(ica_fif_path, overwrite=True)
        print(f"Saved fitted ICA to {ica_fif_path}")
    except Exception as exc:
        print(f"Could not save fitted ICA ({exc}).")

    # Save per-component automatic ICA assignment table for human review/override
    ica_assignments_csv = outputs_root / f"{RAW_FILE.stem}_desc-auto_ica-assignments.csv"
    ica_assignments_json = outputs_root / f"{RAW_FILE.stem}_desc-auto_ica-assignments.json"
    label_priority = ["ecg", "blink", "heog"]
    assignment_rows = []
    with open(ica_assignments_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "component",
                "auto_label",
                "auto_exclude",
                "auto_score",
                "blink_score",
                "heog_score",
                "ecg_score",
            ]
        )

        for comp_idx in range(n_components):
            score_by_label = {
                "blink": float(blink_scores[comp_idx]),
                "heog": float(heog_scores[comp_idx]),
                "ecg": float(ecg_scores[comp_idx]),
            }

            best_label = "other"
            best_score = 0.0
            for label in label_priority:
                s = score_by_label[label]
                if s > best_score:
                    best_score = s
                    best_label = label

            if comp_idx in ecg_inds:
                best_label = "ecg"
                best_score = max(best_score, score_by_label["ecg"])
            elif comp_idx in blink_inds:
                best_label = "blink"
                best_score = max(best_score, score_by_label["blink"])
            elif comp_idx in heog_inds:
                best_label = "heog"
                best_score = max(best_score, score_by_label["heog"])

            row = [
                RAW_FILE.stem,
                comp_idx,
                best_label,
                int(comp_idx in ica.exclude),
                f"{best_score:.6f}",
                f"{score_by_label['blink']:.6f}",
                f"{score_by_label['heog']:.6f}",
                f"{score_by_label['ecg']:.6f}",
            ]
            assignment_rows.append(
                {
                    "subject": RAW_FILE.stem,
                    "component": comp_idx,
                    "auto_label": best_label,
                    "auto_exclude": int(comp_idx in ica.exclude),
                    "auto_score": f"{best_score:.6f}",
                    "blink_score": f"{score_by_label['blink']:.6f}",
                    "heog_score": f"{score_by_label['heog']:.6f}",
                    "ecg_score": f"{score_by_label['ecg']:.6f}",
                }
            )
            writer.writerow(
                row
            )
    print(f"Saved ICA auto assignments to {ica_assignments_csv}")
    _write_json(
        ica_assignments_json,
        {
            "subject": RAW_FILE.stem,
            "stage": "auto",
            "n_components": n_components,
            "rows": assignment_rows,
        },
    )
    print(f"Saved ICA auto assignments JSON to {ica_assignments_json}")

    raw_denoised = raw.copy()
    ica.apply(raw_denoised)

    # Export denoised EDF in derivatives
    denoised_edf_path = outputs_root / f"{RAW_FILE.stem}_desc-denoised_eeg.edf"
    try:
        mne.export.export_raw(denoised_edf_path, raw_denoised, fmt="edf", overwrite=True)
        print(f"Saved denoised EDF to {denoised_edf_path}")
    except Exception as exc:
        print(f"Could not export EDF ({exc}).")

    # Detect bad EEG/ECG segments on denoised data (thresholds tuned for filtered data)
    bad_times = detect_bad_segments(
        raw_denoised,
        eeg_picks=mne.pick_types(raw_denoised.info, eeg=True),
        ecg_ch=ecg_ch,
        sfreq=raw_denoised.info["sfreq"],
        eeg_rms_thresh=150,
        eeg_pp_thresh=600,
        eeg_var_thresh=0.01,
        ecg_rms_thresh=300,
    )
    print(f"Detected {len(bad_times)} bad segments (on denoised signal)")
    for onset, duration in bad_times:
        print(f"  Bad segment: {onset:.2f} - {onset + duration:.2f} s (duration {duration:.2f} s)")

    # Add detected bad segments as annotations (for report + epoch rejection)
    if bad_times:
        bad_annotations = mne.Annotations(
            onset=[t[0] for t in bad_times],
            duration=[t[1] for t in bad_times],
            description=["BAD_signal"] * len(bad_times),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + bad_annotations)
        raw_denoised.set_annotations(raw_denoised.annotations + bad_annotations)
        print(f"Added {len(bad_times)} bad segment annotations")

    # Keep only non-BAD annotations as condition/event markers in report
    annotations = [ann for ann in raw.annotations if not ann["description"].upper().startswith("BAD")]
    if len(annotations) > 0:
        print(f"Found {len(annotations)} non-BAD annotations:")
        for ann in annotations:
            print(f"  {ann['description']} at {ann['onset']:.2f} s, duration {ann['duration']:.2f} s")

    # Combined multi-page PDF: raw EEG + denoised EEG overlay + ECG + R-peaks + bad shadows
    report_path = outputs_root / f"{RAW_FILE.stem}_desc-auto_eeg-ecg-report.pdf"
    save_eeg_ecg_report(
        report_path=report_path,
        raw_before_ica=raw_before_ica,
        raw_denoised=raw_denoised,
        raw=raw,
        ecg_ch=ecg_ch,
        events=events,
        bad_times=bad_times,
        annotations=annotations,
        ica_exclude=ica.exclude,
        subject_name=RAW_FILE.name,
    )

    # Export detected R-peak times to CSV
    rpeak_csv_path = outputs_root / f"{RAW_FILE.stem}_desc-rpeaks_events.csv"
    r_peak_samples = events[:, 0].astype(int)
    r_peak_times_sec = r_peak_samples / sfreq
    with open(rpeak_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_peak_index", "sample", "time_sec"])
        for i, (sample, time_sec) in enumerate(zip(r_peak_samples, r_peak_times_sec), start=1):
            writer.writerow([i, int(sample), f"{time_sec:.6f}"])
    print(f"Saved R-peak times to {rpeak_csv_path}")

    # Export bad segments to CSV
    bad_csv_path = outputs_root / f"{RAW_FILE.stem}_desc-bad-segments_segments.csv"
    with open(bad_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_sec", "duration_sec", "end_sec"])
        for onset, duration in bad_times:
            writer.writerow([f"{onset:.3f}", f"{duration:.3f}", f"{onset + duration:.3f}"])
    print(f"Saved bad segments to {bad_csv_path}")

    # Additional PDF 2: ICA inspection (topography + reconstructed signals for excluded components)
    outputs_root.mkdir(parents=True, exist_ok=True)
    ica_pdf_path = outputs_root / f"{RAW_FILE.stem}_desc-auto_ica-components.pdf"
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(ica_pdf_path) as pdf:
        # All component topographies
        figs = ica.plot_components(show=False)
        save_figures_to_pdf(pdf, figs)

        # For each component (excluded first): topography + stacked reconstructed EEG signals
        comp_maps = ica.get_components()  # shape: (n_eeg_channels_used_for_ICA, n_components)
        eeg_names_inspect = sort_channel_names_conventional(ica.ch_names)
        name_to_idx = {name: i for i, name in enumerate(ica.ch_names)}
        reorder_idx = [name_to_idx[name] for name in eeg_names_inspect]

        excluded_set = set(ica.exclude)
        all_components = list(range(int(ica.n_components_)))
        ordered_components = [c for c in all_components if c in excluded_set] + [
            c for c in all_components if c not in excluded_set
        ]

        for comp_idx in ordered_components:
            is_excluded = comp_idx in excluded_set
            # Human-readable artifact label for page title
            if comp_idx in blink_inds:
                artifact_label = "Blink Artefact"
            elif comp_idx in heog_inds:
                artifact_label = "Lateral Eye Movement Artefact"
            elif comp_idx in ecg_inds:
                artifact_label = "ECG Artefact"
            else:
                artifact_label = "ICA Artefact"

            exclusion_status = "SELECTED FOR EXCLUSION" if is_excluded else "NOT SELECTED FOR EXCLUSION"

            # Get component source time series (last 20 seconds for readability)
            sources = ica.get_sources(raw_ica_fit).get_data(picks=[comp_idx]).ravel()
            inspect_duration = min(20.0, raw_ica_fit.times[-1])
            inspect_start = max(0.0, raw_ica_fit.times[-1] - inspect_duration)
            inspect_end = raw_ica_fit.times[-1]
            inspect_s0 = int(inspect_start * sfreq)
            inspect_s1 = int(inspect_end * sfreq)
            inspect_samples = int(inspect_duration * sfreq)
            print(
                f"ICA inspection component {comp_idx}: showing last {inspect_duration:.1f}s "
                f"({inspect_start:.1f}–{inspect_end:.1f}s)"
            )
            sources_seg = sources[inspect_s0:inspect_s1]
            times_seg = np.arange(inspect_samples) / sfreq
            times_seg = inspect_start + times_seg

            # ECG segment (for heart artifact inspection context)
            ecg_seg_uv = None
            if ecg_ch in raw_ica_fit.ch_names:
                ecg_idx = raw_ica_fit.ch_names.index(ecg_ch)
                ecg_seg_uv = raw_ica_fit.get_data(picks=[ecg_idx])[0, inspect_s0:inspect_s1] * 1e6

            # Reconstruct EEG signal explained by this component in channel space
            # channel_contribution(t, ch) = source(t) * component_map(ch)
            component_map = comp_maps[:, comp_idx]
            reconstructed_eeg = np.outer(sources_seg, component_map) * 1e6  # convert to µV
            reconstructed_eeg = reconstructed_eeg[:, reorder_idx]

            # Create figure with topography on left and stacked traces on right
            fig = plt.figure(figsize=(14, 9))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)
            fig.suptitle(
                f"Component {comp_idx}: {artifact_label} — {exclusion_status}\n"
                f"Showing last {inspect_duration:.1f}s ({inspect_start:.1f}–{inspect_end:.1f}s)",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )

            # Left: topography
            ax_topo = fig.add_subplot(gs[0, 0])
            ica.plot_components([comp_idx], axes=ax_topo, show=False)

            # Right: stacked reconstructed EEG traces (one per channel)
            ax_recon = fig.add_subplot(gs[0, 1])
            trace_height = 100  # µV per channel (vertical spacing)
            trace_scale = 25    # normalize each trace to ±25 µV amplitude
            
            for i, ch_name in enumerate(eeg_names_inspect):
                trace = reconstructed_eeg[:, i]
                # Normalize trace: scale to fit within ±trace_scale
                trace_max = np.max(np.abs(trace))
                if trace_max > 0:
                    trace_norm = trace * (trace_scale / trace_max)
                else:
                    trace_norm = trace
                # Put first channel (Fp1) at the top
                y_offset = (len(eeg_names_inspect) - 1 - i) * trace_height
                ax_recon.plot(times_seg, trace_norm + y_offset, lw=0.7, color="C0", alpha=0.8)
                ax_recon.text(
                    inspect_start - 0.5,
                    y_offset,
                    ch_name,
                    fontsize=7,
                    va="center",
                    ha="right",
                    fontweight="bold",
                )

            # Add ECG at the bottom for heart artifact inspection
            if ecg_seg_uv is not None:
                ecg_max = np.max(np.abs(ecg_seg_uv))
                if ecg_max > 0:
                    ecg_norm = ecg_seg_uv * (trace_scale / ecg_max)
                else:
                    ecg_norm = ecg_seg_uv
                ecg_y = -trace_height
                ax_recon.plot(times_seg, ecg_norm + ecg_y, lw=0.8, color="C3", alpha=0.9)
                ax_recon.text(
                    inspect_start - 0.5,
                    ecg_y,
                    "ECG",
                    fontsize=7,
                    va="center",
                    ha="right",
                    fontweight="bold",
                    color="C3",
                )

            ax_recon.set_xlim(inspect_start, inspect_end)
            ax_recon.set_ylim(-trace_height * 1.5, trace_height * (len(eeg_names_inspect) - 1) + trace_height * 0.5)
            ax_recon.set_xlabel("Time (s)", fontsize=10)
            ax_recon.set_title(f"Component {comp_idx}: Reconstructed EEG across channels", 
                             fontsize=11, fontweight="bold")
            ax_recon.grid(True, axis="x", lw=0.3, alpha=0.4)
            ax_recon.set_yticks([])

            fig.subplots_adjust(top=0.90)
            pdf.savefig(fig)
            plt.close(fig)



    print(f"Saved ICA inspection PDF to {ica_pdf_path}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"R_peak": 999},
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, -0.05),
        preload=True,
        reject_by_annotation=True,
    )
    print(epochs)

    hep = epochs.average()
    print(f"HEP epochs averaged. N={len(epochs)}, baseline=(-0.2, -0.05) s")

    processing_json = outputs_root / f"{RAW_FILE.stem}_desc-processing_auto.json"
    _write_json(
        processing_json,
        {
            "subject": RAW_FILE.stem,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "auto",
            "inputs": {
                "raw_eeg": str(RAW_FILE),
            },
            "preprocessing": {
                "notch_freqs": NOTCH_FREQS,
                "l_freq": L_FREQ,
                "h_freq": H_FREQ,
            },
            "ica": {
                "n_components": n_components,
                "exclude": ica.exclude,
                "blink": sorted(blink_inds),
                "heog": sorted(heog_inds),
                "ecg": sorted(ecg_inds),
            },
            "outputs": {
                "fitted_ica": str(ica_fif_path),
                "auto_assignments_csv": str(ica_assignments_csv),
                "auto_assignments_json": str(ica_assignments_json),
                "denoised_eeg": str(denoised_edf_path),
                "bad_segments_csv": str(bad_csv_path),
                "rpeaks_csv": str(rpeak_csv_path),
                "eeg_ecg_report_pdf": str(report_path),
                "ica_components_pdf": str(ica_pdf_path),
            },
            "hep": {
                "n_epochs": int(len(epochs)),
                "tmin": -0.2,
                "tmax": 0.8,
                "baseline": [-0.2, -0.05],
            },
        },
    )
    print(f"Saved auto processing JSON to {processing_json}")


if __name__ == "__main__":
    if RUN_ALL_FILES:
        raw_files = sorted(Path("data/raw").glob("*.edf"))
    else:
        raw_files = [RAW_FILE]

    if not raw_files:
        print("No EDF files found to process.")
    else:
        print(f"Processing {len(raw_files)} file(s)...")
        for f in raw_files:
            print(f"\n=== Processing {f} ===")
            try:
                main(f)
            except Exception as exc:
                print(f"Failed on {f}: {exc}")
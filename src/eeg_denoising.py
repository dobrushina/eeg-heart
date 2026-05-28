from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import mne
import csv


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

RAW_FILE = Path("data/raw/sub-eeg-1.edf")

CONVENTIONAL_EEG_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2",
]


def sort_channel_names_conventional(ch_names):
    """Sort channel names using conventional 10-20 order; keep unknown channels last."""
    preferred = [ch for ch in CONVENTIONAL_EEG_ORDER if ch in ch_names]
    extras = [ch for ch in ch_names if ch not in CONVENTIONAL_EEG_ORDER]
    return preferred + extras


def sort_eeg_picks_conventional(raw, eeg_picks):
    """Return EEG picks sorted in conventional order with matching channel names."""
    names = [raw.ch_names[p] for p in eeg_picks]
    ordered_names = sort_channel_names_conventional(names)
    name_to_pick = {raw.ch_names[p]: p for p in eeg_picks}
    ordered_picks = np.array([name_to_pick[ch] for ch in ordered_names], dtype=int)
    return ordered_picks, ordered_names


def main():
    # Load raw data
    raw = mne.io.read_raw_edf(RAW_FILE, preload=True)
    print(raw.info)
    print(raw.ch_names) 

    # Set channel types for common non-EEG channels so they are excluded
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
        print("Set channel types:", ch_type_map)

    # Clean channel names: drop reference suffixes like '-A1A2' -> 'Fz'
    # Keep names unique by appending an index if necessary
    orig_names = raw.ch_names[:]
    mapping = {}
    used = set(orig_names)
    for ch in orig_names:
        new = re.sub(r'[-_:].*$', '', ch)
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
        print("Renamed channels:", mapping)

    # Set a standard montage (if available). Montage is matched by channel name.
    try:
        raw.set_montage("standard_1020", on_missing="warn")
    except Exception as exc:
        print("Warning setting montage:", exc)

    # Make sure only EEG channels are used for average referencing
    try:
        raw.set_eeg_reference("average")
    except Exception as exc:
        print("Warning setting average EEG reference:", exc)

    # Cleaning parameters
    notch_freqs = [50, 100]
    l_freq = 0.5
    h_freq = 40

    # Clean: notch and bandpass
    raw.notch_filter(notch_freqs)
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Store plot scaling constant for later use if needed
    eeg_scaling = 50e-6

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

    # Detect bad EEG/ECG segments (thresholds tuned for filtered data)
    bad_times = detect_bad_segments(
        raw, 
        eeg_picks=mne.pick_types(raw.info, eeg=True), 
        ecg_ch=ecg_ch, 
        sfreq=raw.info["sfreq"],
        eeg_rms_thresh=150,
        eeg_pp_thresh=600,
        eeg_var_thresh=0.01,
        ecg_rms_thresh=300
    )
    print(f"Detected {len(bad_times)} bad segments")
    for onset, duration in bad_times:
        print(f"  Bad segment: {onset:.2f} - {onset + duration:.2f} s (duration {duration:.2f} s)")

    # Add detected bad segments as annotations to raw so they integrate with MNE
    if bad_times:
        bad_annotations = mne.Annotations(
            onset=[t[0] for t in bad_times],
            duration=[t[1] for t in bad_times],
            description=["BAD_signal"] * len(bad_times),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + bad_annotations)
        print(f"Added {len(bad_times)} bad segment annotations to raw data")

    # Extract annotations (condition marks, events) from the raw file
    annotations = raw.annotations
    if len(annotations) > 0:
        print(f"Found {len(annotations)} annotations:")
        for ann in annotations:
            print(f"  {ann['description']} at {ann['onset']:.2f} s, duration {ann['duration']:.2f} s")
    else:
        annotations = []

    # --- Export multi-page PDF report with EEG pages and ECG overlay ---
    # Create per-subject output folder
    subj_id = RAW_FILE.stem  # e.g., "sub-eeg-3"
    outputs_root = Path("outputs") / subj_id
    outputs_root.mkdir(parents=True, exist_ok=True)

    sfreq = raw.info["sfreq"]
    total_sec = raw.n_times / sfreq
    page_duration = 10.0  # seconds per page
    n_pages = int(np.ceil(total_sec / page_duration))

    # Prepare EEG picks and data
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        print("No EEG channels to plot in report.")
    else:
        eeg_picks, eeg_names = sort_eeg_picks_conventional(raw, eeg_picks)
        eeg_data = raw.get_data(picks=eeg_picks) * 1e6  # convert to µV

        from matplotlib.backends.backend_pdf import PdfPages

        # Fixed display gains in µV. These directly control visible amplitude.
        EEG_GAIN_UV = 50
        ECG_GAIN_UV = 500

        report_name = f"{RAW_FILE.stem}_eeg_ecg_report.pdf"
        report_path = outputs_root / report_name
        with PdfPages(report_path) as pdf:
            for page in range(n_pages):
                t0 = page * page_duration
                t1 = min((page + 1) * page_duration, total_sec)
                s0 = int(t0 * sfreq)
                s1 = int(t1 * sfreq)

                times = np.arange(s0, s1) / sfreq

                fig, (ax_eeg, ax_ecg) = plt.subplots(
                    2,
                    1,
                    figsize=(11, 8),
                    sharex=True,
                    gridspec_kw={"height_ratios": [5, 0.8]},
                )

                # EEG panel: normalize by gain so changing the gain changes plot size
                eeg_spacing = 1.2
                eeg_offsets = np.arange(len(eeg_names))[::-1] * eeg_spacing
                for i, ch_name in enumerate(eeg_names):
                    trace_uv = eeg_data[i, s0:s1]
                    trace_display = trace_uv / EEG_GAIN_UV
                    ax_eeg.plot(times, trace_display + eeg_offsets[i], color="C0", lw=0.6)

                ax_eeg.set_yticks(eeg_offsets)
                ax_eeg.set_yticklabels(eeg_names, fontsize=7)
                ax_eeg.set_ylim(-eeg_spacing, eeg_offsets[0] + eeg_spacing)
                ax_eeg.set_xlim(t0, t0 + page_duration)
                ax_eeg.grid(True, axis="x", lw=0.3, alpha=0.5)

                # Shade bad segments
                for bad_onset, bad_duration in bad_times:
                    bad_end = bad_onset + bad_duration
                    if bad_end > t0 and bad_onset < (t0 + page_duration):
                        shade_start = max(bad_onset, t0)
                        shade_end = min(bad_end, t0 + page_duration)
                        ax_eeg.axvspan(shade_start, shade_end, color="red", alpha=0.15, zorder=0)
                        ax_ecg.axvspan(shade_start, shade_end, color="red", alpha=0.15, zorder=0)

                # Add annotation markers (condition marks, eyes open/closed, etc.)
                colors_ann = {"eyes open": "green", "eyes closed": "red"}
                for ann in annotations:
                    ann_time = ann["onset"]
                    ann_desc = ann["description"].strip()
                    if t0 <= ann_time < (t0 + page_duration):
                        color = colors_ann.get(ann_desc.lower(), "gray")
                        ax_eeg.axvline(ann_time, color=color, lw=1.5, alpha=0.7, linestyle="--")
                        ax_eeg.text(
                            ann_time,
                            ax_eeg.get_ylim()[1] * 0.95,
                            ann_desc,
                            fontsize=6,
                            rotation=90,
                            va="top",
                            ha="right",
                            color=color,
                        )
                ax_eeg.set_title(f"EEG / ECG report — {RAW_FILE.name} (page {page + 1}/{n_pages})")
                ax_eeg.set_ylabel("EEG")

                # EEG scale bar in normalized display units
                duration = t1 - t0
                x_bar = t0 + duration * 0.01
                eeg_bar_y0 = -0.6
                ax_eeg.plot([x_bar, x_bar], [eeg_bar_y0, eeg_bar_y0 + 1], color="k", lw=2)
                ax_eeg.text(
                    x_bar + duration * 0.01,
                    eeg_bar_y0 + 0.5,
                    f"{EEG_GAIN_UV} µV",
                    fontsize=7,
                    va="center",
                )

                # ECG panel with its own zero baseline
                try:
                    ecg_raw = raw.copy().pick([ecg_ch])
                    ecg_full_uv = ecg_raw.get_data(picks=[0])[0] * 1e6
                    ecg_seg_display = ecg_full_uv[s0:s1] / ECG_GAIN_UV
                    ax_ecg.plot(times, ecg_seg_display, color="C3", lw=0.8)
                    ax_ecg.axhline(0, color="0.4", lw=0.6, ls="--")

                    r_times = events[:, 0].astype(int) / sfreq
                    mask = (r_times >= t0) & (r_times < t1)
                    r_t_in = r_times[mask]
                    r_samps = (r_t_in * sfreq).astype(int)
                    r_samps = np.clip(r_samps, 0, len(ecg_full_uv) - 1)
                    r_vals_display = ecg_full_uv[r_samps] / ECG_GAIN_UV
                    ax_ecg.scatter(r_t_in, r_vals_display, color="red", s=10, zorder=3)

                    ax_ecg.set_ylabel("ECG")
                    ax_ecg.grid(True, axis="x", lw=0.3, alpha=0.5)
                    ax_ecg.set_ylim(-0.8, 3.2)

                    ecg_bar_y0 = -2.0
                    ax_ecg.plot([x_bar, x_bar], [ecg_bar_y0, ecg_bar_y0 + 1], color="k", lw=2)
                    ax_ecg.text(
                        x_bar + duration * 0.01,
                        ecg_bar_y0 + 0.5,
                        f"{ECG_GAIN_UV} µV",
                        fontsize=7,
                        va="center",
                    )
                except Exception:
                    pass

                ax_ecg.set_xlabel("Time (s)")
                # footer with montage, filter params and display gains
                montage_obj = raw.get_montage()
                montage_name = montage_obj.kind if montage_obj is not None and hasattr(montage_obj, "kind") else (
                    "standard_1020"
                )
                footer = f"Montage: {montage_name}    Notch: {notch_freqs} Hz    Bandpass: {l_freq}-{h_freq} Hz    Display gains: EEG={EEG_GAIN_UV}µV, ECG={ECG_GAIN_UV}µV"
                fig.text(0.5, 0.01, footer, ha="center", fontsize=8)
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved EEG/ECG multi-page report to {report_path}")

        # Export bad segments to CSV
        bad_csv_path = outputs_root / f"{RAW_FILE.stem}_bad_segments.csv"
        with open(bad_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["onset_sec", "duration_sec", "end_sec"])
            for onset, duration in bad_times:
                writer.writerow([f"{onset:.3f}", f"{duration:.3f}", f"{onset + duration:.3f}"])
        print(f"Saved bad segments to {bad_csv_path}")

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

    blink_inds = set()
    heog_inds = set()
    ecg_inds = set()

    # Blink candidates from Fp1/Fp2 (no dedicated EOG channels)
    for ch in ["Fp1", "Fp2"]:
        if ch in raw_ica_fit.ch_names:
            inds, _ = ica.find_bads_eog(raw_ica_fit, ch_name=ch, threshold=3.0)
            blink_inds.update(inds[:2])  # keep conservative

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
            inds, _ = ica.find_bads_eog(raw_heog, ch_name="HEOG_proxy", threshold=3.0)
            heog_inds.update(inds[:2])
            break

    # ECG-related components
    if ecg_ch in raw_ica_fit.ch_names:
        inds, _ = ica.find_bads_ecg(raw_ica_fit, ch_name=ecg_ch, method="correlation", threshold="auto")
        ecg_inds.update(inds[:2])

    ica.exclude = sorted(blink_inds | heog_inds | ecg_inds)
    print(f"ICA exclude components: {ica.exclude}")
    print(f"  blink (Fp1/Fp2): {sorted(blink_inds)}")
    print(f"  horizontal (lateral proxy): {sorted(heog_inds)}")
    print(f"  ECG: {sorted(ecg_inds)}")

    raw_denoised = raw.copy()
    ica.apply(raw_denoised)

    # Export denoised EDF directly under data/denoised (no per-subject folder)
    denoised_root = Path("data/denoised")
    denoised_root.mkdir(parents=True, exist_ok=True)
    denoised_edf_path = denoised_root / f"{RAW_FILE.stem}_denoised.edf"
    try:
        mne.export.export_raw(denoised_edf_path, raw_denoised, fmt="edf", overwrite=True)
        print(f"Saved denoised EDF to {denoised_edf_path}")
    except Exception as exc:
        print(f"Could not export EDF ({exc}).")

    # Additional PDF 1: cleaned EEG overlaid on raw EEG
    overlay_pdf_path = outputs_root / f"{RAW_FILE.stem}_raw_vs_denoised_overlay.pdf"
    eeg_picks_overlay = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks_overlay) > 0:
        eeg_picks_overlay, eeg_names_overlay = sort_eeg_picks_conventional(raw, eeg_picks_overlay)
        raw_eeg_uv = raw_before_ica.get_data(picks=eeg_picks_overlay) * 1e6
        den_eeg_uv = raw_denoised.get_data(picks=eeg_picks_overlay) * 1e6

        from matplotlib.backends.backend_pdf import PdfPages

        EEG_GAIN_UV_OVERLAY = 50
        with PdfPages(overlay_pdf_path) as pdf:
            for page in range(n_pages):
                t0 = page * page_duration
                t1 = min((page + 1) * page_duration, total_sec)
                s0 = int(t0 * sfreq)
                s1 = int(t1 * sfreq)
                times = np.arange(s0, s1) / sfreq

                fig, ax = plt.subplots(figsize=(11, 8))
                spacing = 1.2
                offsets = np.arange(len(eeg_names_overlay))[::-1] * spacing

                for i, ch_name in enumerate(eeg_names_overlay):
                    trace_raw = raw_eeg_uv[i, s0:s1] / EEG_GAIN_UV_OVERLAY
                    trace_den = den_eeg_uv[i, s0:s1] / EEG_GAIN_UV_OVERLAY
                    ax.plot(times, trace_raw + offsets[i], color="0.7", lw=0.5)
                    ax.plot(times, trace_den + offsets[i], color="C0", lw=0.7)

                # Legend proxy lines
                ax.plot([], [], color="0.7", lw=1.0, label="Raw")
                ax.plot([], [], color="C0", lw=1.0, label="Denoised")
                ax.legend(loc="upper right", fontsize=8)

                ax.set_yticks(offsets)
                ax.set_yticklabels(eeg_names_overlay, fontsize=7)
                ax.set_ylim(-spacing, offsets[0] + spacing)
                ax.set_xlim(t0, t0 + page_duration)
                ax.grid(True, axis="x", lw=0.3, alpha=0.5)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("EEG")
                ax.set_title(f"Raw vs denoised EEG — {RAW_FILE.name} (page {page + 1}/{n_pages})")

                footer = f"ICA excluded components: {ica.exclude}    Gain: EEG={EEG_GAIN_UV_OVERLAY}µV"
                fig.text(0.5, 0.01, footer, ha="center", fontsize=8)
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved raw-vs-denoised overlay PDF to {overlay_pdf_path}")

    # Additional PDF 2: ICA inspection (topography + reconstructed signals for excluded components)
    outputs_root.mkdir(parents=True, exist_ok=True)
    ica_pdf_path = outputs_root / f"{RAW_FILE.stem}_ica_components_inspection.pdf"
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(ica_pdf_path) as pdf:
        # All component topographies
        figs = ica.plot_components(show=False)
        save_figures_to_pdf(pdf, figs)

        # For each excluded component: topography + stacked reconstructed EEG signals
        comp_maps = ica.get_components()  # shape: (n_eeg_channels_used_for_ICA, n_components)
        eeg_names_inspect = sort_channel_names_conventional(ica.ch_names)
        name_to_idx = {name: i for i, name in enumerate(ica.ch_names)}
        reorder_idx = [name_to_idx[name] for name in eeg_names_inspect]
        
        for comp_idx in ica.exclude:
            # Human-readable artifact label for page title
            if comp_idx in blink_inds:
                artifact_label = "Blink Artefact"
            elif comp_idx in heog_inds:
                artifact_label = "Lateral Eye Movement Artefact"
            elif comp_idx in ecg_inds:
                artifact_label = "ECG Artefact"
            else:
                artifact_label = "ICA Artefact"

            # Get component source time series (first 20 seconds for readability)
            sources = ica.get_sources(raw_ica_fit).get_data(picks=[comp_idx]).ravel()
            inspect_duration = min(20.0, raw_ica_fit.times[-1])
            inspect_samples = int(inspect_duration * sfreq)
            sources_seg = sources[:inspect_samples]
            times_seg = np.arange(inspect_samples) / sfreq

            # ECG segment (for heart artifact inspection context)
            ecg_seg_uv = None
            if ecg_ch in raw_ica_fit.ch_names:
                ecg_idx = raw_ica_fit.ch_names.index(ecg_ch)
                ecg_seg_uv = raw_ica_fit.get_data(picks=[ecg_idx])[0, :inspect_samples] * 1e6

            # Reconstruct EEG signal explained by this component in channel space
            # channel_contribution(t, ch) = source(t) * component_map(ch)
            component_map = comp_maps[:, comp_idx]
            reconstructed_eeg = np.outer(sources_seg, component_map) * 1e6  # convert to µV
            reconstructed_eeg = reconstructed_eeg[:, reorder_idx]

            # Create figure with topography on left and stacked traces on right
            fig = plt.figure(figsize=(14, 9))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)
            fig.suptitle(
                f"Component {comp_idx}: {artifact_label}",
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
                ax_recon.text(-0.5, y_offset, ch_name, fontsize=7, va="center", ha="right", fontweight="bold")

            # Add ECG at the bottom for heart artifact inspection
            if ecg_seg_uv is not None:
                ecg_max = np.max(np.abs(ecg_seg_uv))
                if ecg_max > 0:
                    ecg_norm = ecg_seg_uv * (trace_scale / ecg_max)
                else:
                    ecg_norm = ecg_seg_uv
                ecg_y = -trace_height
                ax_recon.plot(times_seg, ecg_norm + ecg_y, lw=0.8, color="C3", alpha=0.9)
                ax_recon.text(-0.5, ecg_y, "ECG", fontsize=7, va="center", ha="right", fontweight="bold", color="C3")

            ax_recon.set_xlim(-1, inspect_duration)
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


if __name__ == "__main__":
    main()
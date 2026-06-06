from pathlib import Path
import csv
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import mne
import numpy as np
import matplotlib.pyplot as plt

from eeg_common import (
    preprocess_raw_for_pipeline,
    save_eeg_ecg_report,
    sort_channel_names_conventional,
)
from eeg_config import (
    NOTCH_FREQS,
    L_FREQ,
    H_FREQ,
    DERIVATIVES_DIR,
    AUTO_PIPELINE_DIR,
    MANUAL_PIPELINE_DIR,
)

# Easy run mode toggle:
# - Set to True to process only subjects listed in SELECTED_SUBJECTS
# - Set to False to process all subjects found in manual CSV files
PROCESS_SELECTED_SUBJECTS = True
SELECTED_SUBJECTS = ["sub-eeg-58"]


def _canonical_label(label: str) -> str:
    if label is None:
        return "other"
    lab = str(label).strip().lower()
    if lab in {"blink", "blinks", "eog", "veog", "vertical_eog"}:
        return "blink"
    if lab in {"heog", "lem", "horizontal", "horizontal_eog", "lateral", "lateral_eye", "lateral eye movement"}:
        return "heog"
    if lab in {"ecg", "cardiac", "heart"}:
        return "ecg"
    if lab in {"other", "none", ""}:
        return "other"
    return lab


def _parse_optional_int(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    return int(float(s))


def _parse_optional_label(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    return _canonical_label(s)


def _load_manual_components(manual_csv_path: Path) -> Dict[str, Dict[int, Dict[str, object]]]:
    if not manual_csv_path.exists():
        raise FileNotFoundError(f"Manual corrections file not found: {manual_csv_path}")

    per_subject: Dict[str, Dict[int, Dict[str, object]]] = {}

    with open(manual_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = (row.get("subject") or "").strip()
            comp_s = (row.get("component") or "").strip()
            if not subject or not comp_s:
                continue

            try:
                component = int(float(comp_s))
            except ValueError:
                print(f"Skipping invalid component '{comp_s}' for subject {subject}")
                continue

            manual_label = _parse_optional_label(row.get("manual_label"))
            manual_exclude = _parse_optional_int(row.get("manual_exclude"))
            reviewer = (row.get("reviewer") or "").strip()
            comment = (row.get("comment") or "").strip()

            per_subject.setdefault(subject, {})[component] = {
                "manual_label": manual_label,
                "manual_exclude": manual_exclude,
                "reviewer": reviewer,
                "comment": comment,
            }

    return per_subject


def _split_bad_electrodes(value: str) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.replace("|", ",").replace(";", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts


def _load_bad_electrodes(manual_bad_csv_path: Path) -> Dict[str, List[str]]:
    if not manual_bad_csv_path.exists():
        return {}

    per_subject: Dict[str, List[str]] = {}
    with open(manual_bad_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return per_subject

        def _norm_header(name: str) -> str:
            return (name or "").replace("\ufeff", "").lower().strip()

        field_map = {_norm_header(name): name for name in reader.fieldnames}
        subject_key = field_map.get("subject") or field_map.get("participant") or field_map.get("subj")
        if subject_key is None:
            print(f"Warning: {manual_bad_csv_path} has no 'subject' column; skipping bad-electrode table.")
            return per_subject

        explicit_bad_cols = [
            key
            for key in [
                "bad_electrode",
                "bad_electrodes",
                "bad_channel",
                "bad_channels",
                "electrode",
                "electrodes",
                "channel",
                "channels",
            ]
            if key in field_map
        ]

        if not explicit_bad_cols:
            print(
                f"Warning: {manual_bad_csv_path} has no bad-electrode column "
                f"(expected e.g. 'bad_electrode'); skipping bad-electrode table."
            )
            return per_subject

        for row in reader:
            subject = (row.get(subject_key) or "").strip()
            if not subject:
                continue

            bad_list: List[str] = []
            for c in explicit_bad_cols:
                bad_list.extend(_split_bad_electrodes(row.get(field_map[c], "")))

            # Deduplicate while preserving order
            existing = per_subject.get(subject, [])
            bad_list = existing + bad_list
            dedup = []
            seen = set()
            for ch in bad_list:
                if ch not in seen:
                    seen.add(ch)
                    dedup.append(ch)

            per_subject[subject] = dedup

    return per_subject


def _load_auto_assignments(auto_csv_path: Path) -> List[Dict[str, object]]:
    if not auto_csv_path.exists():
        raise FileNotFoundError(f"Auto assignments file not found: {auto_csv_path}")

    rows = []
    with open(auto_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            component = int(row["component"])
            auto_label = _canonical_label(row.get("auto_label", "other"))
            auto_exclude = int(float(row.get("auto_exclude", 0)))

            rows.append(
                {
                    "subject": row.get("subject", ""),
                    "component": component,
                    "auto_label": auto_label,
                    "auto_exclude": auto_exclude,
                    "auto_score": row.get("auto_score", ""),
                    "blink_score": row.get("blink_score", ""),
                    "heog_score": row.get("heog_score", ""),
                    "ecg_score": row.get("ecg_score", ""),
                }
            )

    return rows


def _scores_to_abs_vector(scores, n_components):
    if scores is None:
        return np.zeros(n_components, dtype=float)

    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 0:
        return np.zeros(n_components, dtype=float)

    if arr.ndim == 1:
        vec = np.abs(arr)
    else:
        if arr.shape[0] != n_components and arr.shape[-1] == n_components:
            arr = arr.T
        if arr.shape[0] == n_components:
            vec = np.max(np.abs(arr), axis=1)
        else:
            return np.zeros(n_components, dtype=float)

    if vec.shape[0] != n_components:
        return np.zeros(n_components, dtype=float)
    return vec


def _compute_auto_assignments_from_ica(subject: str, ica, raw_ica_fit, ecg_ch: str):
    n_components = int(ica.n_components_)
    blink_inds = set()
    heog_inds = set()
    ecg_inds = set()
    blink_scores = np.zeros(n_components, dtype=float)
    heog_scores = np.zeros(n_components, dtype=float)
    ecg_scores = np.zeros(n_components, dtype=float)

    for ch in ["Fp1", "Fp2"]:
        if ch in raw_ica_fit.ch_names:
            inds, scores = ica.find_bads_eog(raw_ica_fit, ch_name=ch, threshold=3.0)
            blink_inds.update(inds[:2])
            blink_scores = np.maximum(blink_scores, _scores_to_abs_vector(scores, n_components))

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

    if ecg_ch in raw_ica_fit.ch_names:
        inds, scores = ica.find_bads_ecg(raw_ica_fit, ch_name=ecg_ch, method="correlation", threshold="auto")
        ecg_inds.update(inds[:2])
        ecg_scores = np.maximum(ecg_scores, _scores_to_abs_vector(scores, n_components))

    exclude_auto = sorted(blink_inds | heog_inds | ecg_inds)
    label_priority = ["ecg", "blink", "heog"]
    rows = []
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

        rows.append(
            {
                "subject": subject,
                "component": comp_idx,
                "auto_label": best_label,
                "auto_exclude": int(comp_idx in exclude_auto),
                "auto_score": f"{best_score:.6f}",
                "blink_score": f"{score_by_label['blink']:.6f}",
                "heog_score": f"{score_by_label['heog']:.6f}",
                "ecg_score": f"{score_by_label['ecg']:.6f}",
            }
        )

    return rows


def _save_figures_to_pdf(pdf, fig_or_figs):
    if isinstance(fig_or_figs, list):
        for fig in fig_or_figs:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
    else:
        if fig_or_figs is not None:
            pdf.savefig(fig_or_figs)
            plt.close(fig_or_figs)


def _save_manual_ica_components_report(
    report_path: Path,
    subject: str,
    ica,
    raw_ica_fit,
    ecg_ch: str,
    merged_rows: List[Dict[str, object]],
):
    from matplotlib.backends.backend_pdf import PdfPages

    merged_by_comp = {int(r["component"]): r for r in merged_rows}
    excluded_set = {int(r["component"]) for r in merged_rows if int(r["final_exclude"]) == 1}
    all_components = list(range(int(ica.n_components_)))
    ordered_components = [c for c in all_components if c in excluded_set] + [
        c for c in all_components if c not in excluded_set
    ]

    comp_maps = ica.get_components()
    eeg_names = sort_channel_names_conventional(ica.ch_names)
    name_to_idx = {name: i for i, name in enumerate(ica.ch_names)}
    reorder_idx = [name_to_idx[name] for name in eeg_names]
    sfreq = raw_ica_fit.info["sfreq"]

    with PdfPages(report_path) as pdf:
        figs = ica.plot_components(show=False)
        _save_figures_to_pdf(pdf, figs)

        for comp_idx in ordered_components:
            row = merged_by_comp.get(comp_idx, {})
            final_label = str(row.get("final_label", "other"))
            exclusion_status = "SELECTED FOR EXCLUSION" if comp_idx in excluded_set else "NOT SELECTED FOR EXCLUSION"

            sources = ica.get_sources(raw_ica_fit).get_data(picks=[comp_idx]).ravel()
            inspect_duration = min(20.0, raw_ica_fit.times[-1])
            inspect_start = max(0.0, raw_ica_fit.times[-1] - inspect_duration)
            inspect_end = raw_ica_fit.times[-1]
            inspect_s0 = int(inspect_start * sfreq)
            inspect_s1 = int(inspect_end * sfreq)
            times_seg = inspect_start + np.arange(inspect_s1 - inspect_s0) / sfreq
            sources_seg = sources[inspect_s0:inspect_s1]

            ecg_seg_uv = None
            if ecg_ch in raw_ica_fit.ch_names:
                ecg_idx = raw_ica_fit.ch_names.index(ecg_ch)
                ecg_seg_uv = raw_ica_fit.get_data(picks=[ecg_idx])[0, inspect_s0:inspect_s1] * 1e6

            component_map = comp_maps[:, comp_idx]
            reconstructed_eeg = np.outer(sources_seg, component_map) * 1e6
            reconstructed_eeg = reconstructed_eeg[:, reorder_idx]

            fig = plt.figure(figsize=(14, 9))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)
            fig.suptitle(
                f"{subject} — Component {comp_idx}: {final_label} — {exclusion_status}\n"
                f"Showing last {inspect_duration:.1f}s ({inspect_start:.1f}–{inspect_end:.1f}s)",
                fontsize=13,
                fontweight="bold",
                y=0.98,
            )

            ax_topo = fig.add_subplot(gs[0, 0])
            ica.plot_components([comp_idx], axes=ax_topo, show=False)

            ax_recon = fig.add_subplot(gs[0, 1])
            trace_height = 100
            trace_scale = 25
            for i, ch_name in enumerate(eeg_names):
                trace = reconstructed_eeg[:, i]
                trace_max = np.max(np.abs(trace))
                trace_norm = trace * (trace_scale / trace_max) if trace_max > 0 else trace
                y_offset = (len(eeg_names) - 1 - i) * trace_height
                ax_recon.plot(times_seg, trace_norm + y_offset, lw=0.7, color="C0", alpha=0.8)
                ax_recon.text(inspect_start - 0.5, y_offset, ch_name, fontsize=7, va="center", ha="right", fontweight="bold")

            if ecg_seg_uv is not None:
                ecg_max = np.max(np.abs(ecg_seg_uv))
                ecg_norm = ecg_seg_uv * (trace_scale / ecg_max) if ecg_max > 0 else ecg_seg_uv
                ecg_y = -trace_height
                ax_recon.plot(times_seg, ecg_norm + ecg_y, lw=0.8, color="C3", alpha=0.9)
                ax_recon.text(inspect_start - 0.5, ecg_y, "ECG", fontsize=7, va="center", ha="right", fontweight="bold", color="C3")

            ax_recon.set_xlim(inspect_start, inspect_end)
            ax_recon.set_ylim(-trace_height * 1.5, trace_height * (len(eeg_names) - 1) + trace_height * 0.5)
            ax_recon.set_xlabel("Time (s)", fontsize=10)
            ax_recon.set_title(f"Component {comp_idx}: reconstructed channel contributions", fontsize=11, fontweight="bold")
            ax_recon.grid(True, axis="x", lw=0.3, alpha=0.4)
            ax_recon.set_yticks([])

            fig.subplots_adjust(top=0.90)
            pdf.savefig(fig)
            plt.close(fig)


def _merge_assignments(
    auto_rows: List[Dict[str, object]],
    manual_map: Dict[int, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[int]]:
    merged = []
    changed_components = []

    for row in auto_rows:
        comp = int(row["component"])
        auto_label = str(row["auto_label"])
        auto_exclude = int(row["auto_exclude"])

        override = manual_map.get(comp)
        manual_label = None
        manual_exclude = None
        reviewer = ""
        comment = ""

        if override is not None:
            manual_label = override.get("manual_label")
            manual_exclude = override.get("manual_exclude")
            reviewer = str(override.get("reviewer", ""))
            comment = str(override.get("comment", ""))

        final_label = manual_label if manual_label is not None else auto_label
        final_exclude = int(manual_exclude) if manual_exclude is not None else auto_exclude

        changed = int((final_label != auto_label) or (final_exclude != auto_exclude))
        if changed:
            changed_components.append(comp)

        merged.append(
            {
                "subject": row["subject"],
                "component": comp,
                "auto_label": auto_label,
                "auto_exclude": auto_exclude,
                "manual_label": "" if manual_label is None else manual_label,
                "manual_exclude": "" if manual_exclude is None else int(manual_exclude),
                "final_label": final_label,
                "final_exclude": final_exclude,
                "changed": changed,
                "reviewer": reviewer,
                "comment": comment,
                "auto_score": row.get("auto_score", ""),
                "blink_score": row.get("blink_score", ""),
                "heog_score": row.get("heog_score", ""),
                "ecg_score": row.get("ecg_score", ""),
            }
        )

    return merged, sorted(set(changed_components))


def _write_corrected_assignments(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "component",
                "auto_label",
                "auto_exclude",
                "manual_label",
                "manual_exclude",
                "final_label",
                "final_exclude",
                "changed",
                "reviewer",
                "comment",
                "auto_score",
                "blink_score",
                "heog_score",
                "ecg_score",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["subject"],
                    r["component"],
                    r["auto_label"],
                    r["auto_exclude"],
                    r["manual_label"],
                    r["manual_exclude"],
                    r["final_label"],
                    r["final_exclude"],
                    r["changed"],
                    r["reviewer"],
                    r["comment"],
                    r["auto_score"],
                    r["blink_score"],
                    r["heog_score"],
                    r["ecg_score"],
                ]
            )


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_corrected_assignments_json(path: Path, rows: List[Dict[str, object]]):
    _write_json(path, {"n_components": len(rows), "rows": rows})


def _write_manual_processing_json(
    path: Path,
    subject: str,
    raw_file: Path,
    auto_ica_file: Path,
    auto_assignments_file: Path,
    bad_electrodes: List[str],
    ica_mode: str,
    final_exclude_components: List[int],
    changed_components: List[int],
    outputs: Dict[str, str],
):
    _write_json(
        path,
        {
            "subject": subject,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "manual",
            "inputs": {
                "raw_eeg": str(raw_file),
                "auto_fitted_ica": str(auto_ica_file),
                "auto_assignments_csv": str(auto_assignments_file),
                "manual_components_csv": "manual/components.csv",
                "manual_bad_electrodes_csv": "manual/bad_electrodes.csv",
            },
            "preprocessing": {
                "notch_freqs": NOTCH_FREQS,
                "l_freq": L_FREQ,
                "h_freq": H_FREQ,
                "bad_electrodes": bad_electrodes,
            },
            "ica": {
                "mode": ica_mode,
                "final_exclude_components": final_exclude_components,
                "n_excluded": len(final_exclude_components),
                "changed_components": changed_components,
                "n_changed": len(changed_components),
            },
            "outputs": outputs,
        },
    )


def _write_corrected_components_summary(path: Path, subject: str, exclude_components: List[int], changed_components: List[int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"subject,{subject}\n")
        f.write(f"final_exclude_components,{';'.join(map(str, exclude_components))}\n")
        f.write(f"changed_components,{';'.join(map(str, changed_components))}\n")


def _process_subject(
    project_root: Path,
    subject: str,
    manual_map: Dict[int, Dict[str, object]],
    bad_electrodes: List[str],
):
    auto_dir = project_root / DERIVATIVES_DIR / AUTO_PIPELINE_DIR / subject
    manual_dir = project_root / DERIVATIVES_DIR / MANUAL_PIPELINE_DIR / subject
    manual_dir.mkdir(parents=True, exist_ok=True)

    raw_file = project_root / "data" / "raw" / f"{subject}.edf"
    denoised_file = manual_dir / f"{subject}_desc-denoised_eeg.edf"
    ica_file = auto_dir / f"{subject}_desc-fitted_ica.fif"
    auto_csv = auto_dir / f"{subject}_desc-auto_ica-assignments.csv"
    corrected_csv = manual_dir / f"{subject}_desc-corrected_ica-assignments.csv"
    corrected_json = manual_dir / f"{subject}_desc-corrected_ica-assignments.json"
    processing_json = manual_dir / f"{subject}_desc-processing_manual.json"

    required = [raw_file]
    if not bad_electrodes:
        required.extend([ica_file, auto_csv])
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"Skipping {subject}: missing files -> {missing}")
        return

    raw = mne.io.read_raw_edf(raw_file, preload=True)
    preprocess_raw_for_pipeline(raw, notch_freqs=NOTCH_FREQS, l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)

    ecg_ch = next((ch for ch in raw.ch_names if "ECG" in ch.upper()), None)

    if bad_electrodes:
        valid_bad = [ch for ch in bad_electrodes if ch in raw.ch_names]
        missing_bad = [ch for ch in bad_electrodes if ch not in raw.ch_names]
        if missing_bad:
            print(f"{subject}: bad electrodes not found after preprocessing: {missing_bad}")
        raw.info["bads"] = valid_bad
        print(f"{subject}: re-fitting ICA with bad electrodes excluded: {valid_bad}")

        raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None)
        picks = mne.pick_types(raw_ica_fit.info, eeg=True, exclude="bads")
        if len(picks) == 0:
            raise RuntimeError(f"{subject}: no EEG channels available to fit ICA after excluding bad electrodes.")

        ica = mne.preprocessing.ICA(
            n_components=0.99,
            method="fastica",
            random_state=97,
            max_iter="auto",
        )
        ica.fit(raw_ica_fit, picks=picks, reject_by_annotation=True)
        auto_rows = _compute_auto_assignments_from_ica(subject, ica, raw_ica_fit, ecg_ch)

        # Keep a traceable fitted ICA from manual refit.
        manual_refit_ica_path = manual_dir / f"{subject}_desc-fitted-manual_ica.fif"
        ica_mode = "refit_with_bad_electrodes"
        try:
            ica.save(manual_refit_ica_path, overwrite=True)
            print(f"{subject}: saved re-fitted ICA -> {manual_refit_ica_path}")
        except Exception as exc:
            print(f"{subject}: could not save re-fitted ICA ({exc})")
    else:
        auto_rows = _load_auto_assignments(auto_csv)
        raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None)
        ica = mne.preprocessing.read_ica(ica_file)
        ica_mode = "loaded_auto_fit"

    merged_rows, changed_components = _merge_assignments(auto_rows, manual_map)
    exclude_components = sorted(int(r["component"]) for r in merged_rows if int(r["final_exclude"]) == 1)

    _write_corrected_assignments(corrected_csv, merged_rows)
    _write_corrected_assignments_json(corrected_json, merged_rows)

    print(f"{subject}: applying corrected exclusion list {exclude_components}")

    ica.exclude = exclude_components

    raw_before_ica = raw.copy()
    raw_denoised = raw.copy()
    ica.apply(raw_denoised)

    denoised_file.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(denoised_file, raw_denoised, fmt="edf", overwrite=True)

    # Generate manual EEG/ECG report PDF (same format as auto report)
    if ecg_ch is not None:
        try:
            events, _, _ = mne.preprocessing.find_ecg_events(raw, ch_name=ecg_ch)
        except Exception as exc:
            print(f"{subject}: could not detect R-peaks for report ({exc})")
            events = None
    else:
        events = None

    if ecg_ch is not None and events is not None and len(events) > 0:
        annotations = [ann for ann in raw.annotations if not ann["description"].upper().startswith("BAD")]
        bad_times: list = []
        bad_times_csv = auto_dir / f"{subject}_desc-bad-segments_segments.csv"
        if bad_times_csv.exists():
            import csv as _csv
            with open(bad_times_csv, newline="") as _f:
                for row in _csv.DictReader(_f):
                    bad_times.append((float(row["onset_sec"]), float(row["duration_sec"])))
        report_path = manual_dir / f"{subject}_desc-manual_eeg-ecg-report.pdf"
        save_eeg_ecg_report(
            report_path=report_path,
            raw_before_ica=raw_before_ica,
            raw_denoised=raw_denoised,
            raw=raw,
            ecg_ch=ecg_ch,
            events=events,
            bad_times=bad_times,
            annotations=annotations,
            ica_exclude=exclude_components,
            subject_name=f"{subject}.edf (manual)",
        )
    else:
        report_path = None
        print(f"{subject}: no ECG or R-peaks found, skipping report")

    # Generate manual ICA components report (always)
    ica_report_path = manual_dir / f"{subject}_desc-manual_ica-components.pdf"
    _save_manual_ica_components_report(
        report_path=ica_report_path,
        subject=subject,
        ica=ica,
        raw_ica_fit=raw_ica_fit,
        ecg_ch=ecg_ch,
        merged_rows=merged_rows,
    )
    print(f"{subject}: wrote manual ICA report -> {ica_report_path}")

    _write_manual_processing_json(
        path=processing_json,
        subject=subject,
        raw_file=raw_file,
        auto_ica_file=ica_file,
        auto_assignments_file=auto_csv,
        bad_electrodes=bad_electrodes,
        ica_mode=ica_mode,
        final_exclude_components=exclude_components,
        changed_components=changed_components,
        outputs={
            "corrected_assignments_csv": str(corrected_csv),
            "corrected_assignments_json": str(corrected_json),
            "denoised_eeg": str(denoised_file),
            "manual_eeg_ecg_report_pdf": "" if report_path is None else str(report_path),
            "manual_ica_components_pdf": str(ica_report_path),
        },
    )
    print(f"{subject}: wrote manual processing JSON -> {processing_json}")

    print(f"{subject}: wrote corrected assignments -> {corrected_csv}")
    print(f"{subject}: wrote corrected assignments JSON -> {corrected_json}")
    print(f"{subject}: overwrote denoised EEG -> {denoised_file}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    manual_csv_path = project_root / "manual" / "components.csv"
    manual_bad_electrodes_csv_path = project_root / "manual" / "bad_electrodes.csv"

    if not manual_bad_electrodes_csv_path.exists():
        xlsx_path = project_root / "manual" / "bad_electrodes.xlsx"
        if xlsx_path.exists():
            print(
                "Warning: manual/bad_electrodes.xlsx found, but manual correction reads CSV. "
                "Please export to manual/bad_electrodes.csv"
            )

    corrections = _load_manual_components(manual_csv_path) if manual_csv_path.exists() else {}
    bad_electrodes_by_subject = _load_bad_electrodes(manual_bad_electrodes_csv_path)

    subjects = sorted(set(corrections.keys()) | set(bad_electrodes_by_subject.keys()))

    if PROCESS_SELECTED_SUBJECTS:
        selected_set = set(SELECTED_SUBJECTS)
        subjects = [s for s in subjects if s in selected_set]
        print(f"Selected-subject mode enabled. Processing: {subjects}")

    if not subjects:
        print("No manual ICA corrections found in manual/components.csv or manual/bad_electrodes.csv")
        return

    print(f"Found manual ICA corrections for {len(subjects)} subject(s): {subjects}")
    for subject in subjects:
        try:
            _process_subject(
                project_root=project_root,
                subject=subject,
                manual_map=corrections.get(subject, {}),
                bad_electrodes=bad_electrodes_by_subject.get(subject, []),
            )
        except Exception as exc:
            print(f"Failed subject {subject}: {exc}")


if __name__ == "__main__":
    main()

from pathlib import Path
import csv
import json
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from eeg_config import DERIVATIVES_DIR, AUTO_PIPELINE_DIR, MANUAL_PIPELINE_DIR

# Easy run mode toggle:
# - Set to True to process only subjects listed in SELECTED_SUBJECTS
# - Set to False to process all subjects found in derivatives
PROCESS_SELECTED_SUBJECTS = False
SELECTED_SUBJECTS = ["sub-eeg-23"]

FINAL_PIPELINE_DIR = "eeg-final"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _subject_dirs(root: Path) -> Set[str]:
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _copy_file(src: Path, dst: Path, required: bool, logs: List[str]) -> bool:
    if not src.exists():
        msg = f"Missing {'required' if required else 'optional'} file: {src}"
        logs.append(msg)
        if required:
            print(msg)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _read_conditions_rows(conditions_csv: Path, subject: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not conditions_csv.exists():
        return rows

    with open(conditions_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows

        def _norm_header(name: str) -> str:
            return (name or "").replace("\ufeff", "").strip().lower()

        field_map = {_norm_header(h): h for h in reader.fieldnames}
        required_cols = ["subject", "condition", "start", "end"]
        if any(col not in field_map for col in required_cols):
            raise ValueError(
                "manual/conditions.csv must contain columns: subject, condition, start, end"
            )

        subj_col = field_map["subject"]
        cond_col = field_map["condition"]
        start_col = field_map["start"]
        end_col = field_map["end"]
        comment_col = field_map.get("comment")

        for row in reader:
            subj = (row.get(subj_col) or "").strip()
            if subj != subject:
                continue

            condition = (row.get(cond_col) or "").strip()
            if not condition:
                continue

            try:
                start = float(row.get(start_col, ""))
                end = float(row.get(end_col, ""))
            except (TypeError, ValueError):
                continue

            if end <= start or start < 0:
                continue

            entry = {
                "onset": start,
                "duration": end - start,
                "trial_type": condition,
            }
            if comment_col is not None:
                comment = (row.get(comment_col) or "").strip()
                if comment:
                    entry["comment"] = comment
            rows.append(entry)

    rows.sort(key=lambda x: x["onset"])
    return rows


def _write_events_files(events_tsv: Path, events_json: Path, rows: List[Dict[str, object]]):
    events_tsv.parent.mkdir(parents=True, exist_ok=True)
    has_comment = any("comment" in r for r in rows)

    with open(events_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        headers = ["onset", "duration", "trial_type"] + (["comment"] if has_comment else [])
        writer.writerow(headers)
        for r in rows:
            row = [f"{r['onset']:.6f}", f"{r['duration']:.6f}", r["trial_type"]]
            if has_comment:
                row.append(r.get("comment", ""))
            writer.writerow(row)

    sidecar = {
        "onset": {"Description": "Event onset", "Units": "seconds"},
        "duration": {"Description": "Event duration", "Units": "seconds"},
        "trial_type": {"Description": "Condition label from manual/conditions.csv"},
    }
    if has_comment:
        sidecar["comment"] = {"Description": "Optional reviewer comment from manual/conditions.csv"}

    with open(events_json, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)


def _auto_paths(auto_subj_dir: Path, subject: str) -> Dict[str, Path]:
    return {
        "denoised": auto_subj_dir / f"{subject}_desc-denoised_eeg.edf",
        "report": auto_subj_dir / f"{subject}_desc-auto_eeg-ecg-report.pdf",
        "ica_pdf": auto_subj_dir / f"{subject}_desc-auto_ica-components.pdf",
        "assignments_csv": auto_subj_dir / f"{subject}_desc-auto_ica-assignments.csv",
        "assignments_json": auto_subj_dir / f"{subject}_desc-auto_ica-assignments.json",
        "rpeaks_csv": auto_subj_dir / f"{subject}_desc-rpeaks_events.csv",
        "processing_json": auto_subj_dir / f"{subject}_desc-processing_auto.json",
    }


def _manual_paths(manual_subj_dir: Path, subject: str) -> Dict[str, Path]:
    return {
        "denoised": manual_subj_dir / f"{subject}_desc-denoised_eeg.edf",
        "report": manual_subj_dir / f"{subject}_desc-manual_eeg-ecg-report.pdf",
        "ica_pdf": manual_subj_dir / f"{subject}_desc-manual_ica-components.pdf",
        "assignments_csv": manual_subj_dir / f"{subject}_desc-corrected_ica-assignments.csv",
        "assignments_json": manual_subj_dir / f"{subject}_desc-corrected_ica-assignments.json",
        "processing_json": manual_subj_dir / f"{subject}_desc-processing_manual.json",
    }


def _final_paths(final_subj_dir: Path, subject: str) -> Dict[str, Path]:
    return {
        "denoised": final_subj_dir / f"{subject}_desc-final_denoised_eeg.edf",
        "report": final_subj_dir / f"{subject}_desc-final_eeg-ecg-report.pdf",
        "ica_pdf": final_subj_dir / f"{subject}_desc-final_ica-components.pdf",
        "assignments_csv": final_subj_dir / f"{subject}_desc-final_ica-assignments.csv",
        "assignments_json": final_subj_dir / f"{subject}_desc-final_ica-assignments.json",
        "rpeaks_csv": final_subj_dir / f"{subject}_desc-rpeaks_events.csv",
        "events_tsv": final_subj_dir / f"{subject}_desc-final_events.tsv",
        "events_json": final_subj_dir / f"{subject}_desc-final_events.json",
        "processing_json": final_subj_dir / f"{subject}_desc-processing_final.json",
    }


def _build_subject(
    project_root: Path,
    subject: str,
    conditions_csv: Path,
) -> bool:
    auto_root = project_root / DERIVATIVES_DIR / AUTO_PIPELINE_DIR
    manual_root = project_root / DERIVATIVES_DIR / MANUAL_PIPELINE_DIR
    final_root = project_root / DERIVATIVES_DIR / FINAL_PIPELINE_DIR

    auto_subj_dir = auto_root / subject
    manual_subj_dir = manual_root / subject
    final_subj_dir = final_root / subject

    auto = _auto_paths(auto_subj_dir, subject)
    manual = _manual_paths(manual_subj_dir, subject)
    final = _final_paths(final_subj_dir, subject)

    source_stage = "manual" if manual["denoised"].exists() and manual["assignments_csv"].exists() else "auto"
    source = manual if source_stage == "manual" else auto

    logs: List[str] = []
    copied_from: Dict[str, str] = {}

    # Required final files (denoised + assignments) come from chosen source stage
    required_map = {
        "denoised": True,
        "assignments_csv": True,
    }

    # Optional final files from chosen source stage
    optional_map = {
        "report": False,
        "ica_pdf": False,
        "assignments_json": False,
    }

    success = True
    for key, required in {**required_map, **optional_map}.items():
        ok = _copy_file(source[key], final[key], required=required, logs=logs)
        if required and not ok:
            success = False
        if ok:
            copied_from[key] = str(source[key])

    # R-peaks are always copied from auto stage
    ok_r = _copy_file(auto["rpeaks_csv"], final["rpeaks_csv"], required=True, logs=logs)
    if not ok_r:
        success = False
    else:
        copied_from["rpeaks_csv"] = str(auto["rpeaks_csv"])

    if not success:
        print(f"{subject}: skipped final build due to missing required files")
        return False

    # Conditions -> final events TSV/JSON
    condition_rows = _read_conditions_rows(conditions_csv, subject)
    _write_events_files(final["events_tsv"], final["events_json"], condition_rows)

    # Final provenance JSON
    payload = {
        "subject": subject,
        "timestamp_utc": _now_utc_iso(),
        "stage": "final",
        "source_stage": source_stage,
        "copied_from": copied_from,
        "conditions_source_csv": str(conditions_csv),
        "n_condition_rows_written": len(condition_rows),
        "outputs": {k: str(v) for k, v in final.items()},
        "notes": [
            "rpeaks_events.csv is always copied from auto stage",
        ],
        "warnings": logs,
    }

    final["processing_json"].parent.mkdir(parents=True, exist_ok=True)
    with open(final["processing_json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"{subject}: built final outputs in {final_subj_dir}")
    return True


def main():
    project_root = Path(__file__).resolve().parents[1]
    auto_root = project_root / DERIVATIVES_DIR / AUTO_PIPELINE_DIR
    manual_root = project_root / DERIVATIVES_DIR / MANUAL_PIPELINE_DIR
    conditions_csv = project_root / "manual" / "conditions.csv"

    subjects = sorted(_subject_dirs(auto_root) | _subject_dirs(manual_root))

    if PROCESS_SELECTED_SUBJECTS:
        selected = set(SELECTED_SUBJECTS)
        subjects = [s for s in subjects if s in selected]
        print(f"Selected-subject mode enabled. Processing: {subjects}")

    if not subjects:
        print("No subjects found to build final derivatives.")
        return

    built = 0
    for subject in subjects:
        try:
            if _build_subject(project_root, subject, conditions_csv):
                built += 1
        except Exception as exc:
            print(f"Failed subject {subject}: {exc}")

    print(f"Final build complete: {built}/{len(subjects)} subject(s) built.")


if __name__ == "__main__":
    main()

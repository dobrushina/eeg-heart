# EEG-HEART pipeline

## What this code does

There are two scripts in [src](src):

- [src/eeg_auto_denoising.py](src/eeg_auto_denoising.py): automatic processing
  - preprocesses EEG/ECG,
  - runs ICA,
  - auto-labels components (`blink`, `heog`, `ecg`, `other`),
  - applies auto exclusion,
  - writes denoised EEG and QA outputs.

- [src/eeg_manual_correction.py](src/eeg_manual_correction.py): manual ICA correction
  - reads human edits from [manual/components.csv](manual/components.csv),
  - loads saved ICA + auto assignments,
  - merges manual overrides,
  - applies corrected exclusion list to **raw EEG**,
  - supports bad-electrode ICA refit from [manual/bad_electrodes.csv](manual/bad_electrodes.csv),
  - writes manual-stage derivatives.

Shared constants are in [src/eeg_config.py](src/eeg_config.py). Shared preprocessing/helpers are in [src/eeg_common.py](src/eeg_common.py).

## Human workflow

0. Please change files in the manual/ folder only! Log your work in journal.txt
1. Run auto pipeline (eeg_auto_denoising.py)
2. Inspect auto outputs in [derivatives/eeg-auto-denoise](derivatives/eeg-auto-denoise):
  - `*_desc-auto_ica-components.pdf`
  - `*_desc-auto_eeg-ecg-report.pdf`
3. Add manual corrections to [manual/components.csv](manual/components.csv).
  Optionally add bad electrodes to [manual/bad_electrodes.csv](manual/bad_electrodes.csv).
4. Run manual correction script.
5. Inspect manual outputs in [derivatives/eeg-manual-correction](derivatives/eeg-manual-correction):
  - `*_desc-manual_eeg-ecg-report.pdf`
  - `*_desc-manual_ica-components.pdf`
Adjust components excluded if neccessary.
Tip: you can re-process just one subject using the toggle.
6. Inspect the final filtered eeg for each participant: manually corrected if that was applied or auto-denoised. Iterate with corrections when neccessary. Label conditions while excluding bad (really bad, like significant movement across electrodes) fragments. Uuse several rows for one condition if it was disrupted by a bad fragment.
Note that automatic labelling of bad fragments is FIY only; use your own judjement.

## Manual file formats

### `manual/components.csv`
Used to override ICA component decisions.

Columns:
- `subject` — subject ID, e.g. `sub-eeg-1`
- `component` — ICA component number
- `manual_label` — optional label: `blink`, `heog`, `lem`, `ecg`, `other`
- `manual_exclude` — optional: `1` exclude, `0` keep
- `reviewer` — optional
- `comment` — optional

At least one of `manual_label` or `manual_exclude` must be provided.

### `manual/bad_electrodes.csv`
Used to mark EEG channels as bad for ICA refitting.

One row per bad electrode.

Columns:
- `subject` — subject ID, e.g. `sub-eeg-1`
- `bad_electrode` — channel name, e.g. `Fp1`
- `reviewer` — optional
- `comment` — optional

If a subject appears here, ICA is re-fitted after excluding these electrodes.

### `manual/conditions.csv`
Used to annotate conditions

Typical columns:
- `subject` — subject ID
- `condition` — condition name (EC or EO)
- `comment` — optional note
- `reviewer` — optional

Use several lines per condition when a condition is disrupted by a bad eeg frament.

## Key outputs

Auto stage per subject in [derivatives/eeg-auto-denoise](derivatives/eeg-auto-denoise):

- `*_desc-fitted_ica.fif`
- `*_desc-auto_ica-assignments.csv`
- `*_desc-auto_ica-assignments.json`
- `*_desc-auto_ica-components.pdf`
- `*_desc-auto_eeg-ecg-report.pdf`
- `*_desc-bad-segments_segments.csv`
- `*_desc-rpeaks_events.csv`
- `*_desc-denoised_eeg.edf`
- `*_desc-processing_auto.json`

Manual stage per subject in [derivatives/eeg-manual-correction](derivatives/eeg-manual-correction):

- `*_desc-fitted-manual_ica.fif` (if bad-electrode refit is used)
- `*_desc-corrected_ica-assignments.csv`
- `*_desc-corrected_ica-assignments.json`
- `*_desc-manual_ica-components.pdf`
- `*_desc-manual_eeg-ecg-report.pdf`
- `*_desc-denoised_eeg.edf`
- `*_desc-processing_manual.json`

## Run

From project root:

- Auto: `python src/eeg_auto_denoising.py`
- Manual correction: `python src/eeg_manual_correction.py`

CONVENTIONAL_EEG_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2",
]

NOTCH_FREQS = [50, 100]
L_FREQ = 0.5
H_FREQ = 40

# BIDS-inspired derivatives layout
DERIVATIVES_DIR = "derivatives"
AUTO_PIPELINE_DIR = "eeg-auto-denoise"
MANUAL_PIPELINE_DIR = "eeg-corrected-denoise"
FINAL_PIPELINE_DIR = "eeg-final"
HEP_PIPELINE_DIR = "eeg-hep"

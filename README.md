# Amstelland XLOT Analysis

This repository contains Python scripts and helper modules to process and visualize data from **XLOT (Extended Leak-Off Test)** and related hydraulic fracturing stress tests in the Amstelland field.

The code supports:
- Reading and cleaning raw surface and downhole gauge data
- Applying wellbore and hydrostatic corrections
- Performing closure and fracture analysis
- Generating publication-ready plots

---

## 📂 Project Structure

```
.
├── XLOT1_Amstelland.py        # Main script for analyzing XLOT 1
├── XLOT2_Amstelland.py        # Main script for analyzing XLOT 2
├── testing file.py            # Sandbox for testing functions
├── plotting.py                # Centralized plotting functions (Matplotlib)
├── well_corrections.py        # Current wellbore correction utilities
├── well_corrections_old.py    # Legacy correction code (archival reference)
├── time_difference.py         # Utilities for time alignment (surface ↔ downhole)
├── closure_analysis.py        # Methods for closure/frac pressure interpretation
└── Data/Amstelland/           # Folder for raw input data files (.txt, .csv)
```

---

## ⚙️ Requirements

- Python **3.9+**
- Dependencies (install via pip):

```bash
pip install pandas numpy matplotlib
```

---

## 🚀 Usage

1. **Prepare data**  
   Place raw NLOG or other test data files into:
   ```
   Data/Amstelland/
   ```

2. **Run an analysis script**  
   Example (XLOT 1):

   ```bash
   python XLOT1_Amstelland.py
   ```

   Example (XLOT 2):

   ```bash
   python XLOT2_Amstelland.py
   ```

3. **Outputs**  
   - Cleaned data tables (Pandas DataFrames)  
   - Plots: fracture initiation, pressure decline, closure interpretation  
   - Console log of corrected pressures and time shifts

---

## 📌 Notes

- **Cross-platform paths**:  
  All paths are handled via [`pathlib`](https://docs.python.org/3/library/pathlib.html) to ensure compatibility on **Windows** (`\`) and **macOS/Linux** (`/`).

- **Case sensitivity**:  
  On macOS/Linux, file extensions (`.txt` vs `.TXT`) must match exactly.

- **Legacy files**:  
  `well_corrections_old.py` is retained for reference but not actively used.

---

## 🔬 Background

Extended Leak-Off Tests (XLOTs) are used to determine:
- **Minimum horizontal stress (Shmin)**
- **Fracture closure pressure**
- **Fracture propagation behavior**

The provided workflow integrates corrections, time alignment, and closure analysis to produce reproducible stress test interpretations.

---

## 🧑‍💻 Authors

- Internal use at **EBN BV / TU Delft projects**
- Scripts maintained and iteratively improved for geothermal and geomechanics research

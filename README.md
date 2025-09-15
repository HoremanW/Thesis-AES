# Amstelland XLOT Analysis

This repository contains Python scripts and helper modules to process and visualize data from **XLOT (Extended Leak-Off Test)** and related hydraulic fracturing stress tests in the Amstelland field.

The workflow supports:
- Reading and cleaning raw surface and downhole gauge data
- Applying wellbore and hydrostatic corrections
- Performing closure and fracture analysis (√t, G-function, Barree-type, etc.)
- Generating publication-ready plots for reporting and research

---

## 📂 Project Structure

```
.
├── XLOT1_Amstelland.py        # Main script for analyzing XLOT 1
├── XLOT2_Amstelland.py        # Main script for analyzing XLOT 2
├── testing file.py            # Sandbox for testing functions
├── plotting.py                # Centralized plotting functions (Matplotlib)
├── closure_analysis.py        # Closure/frac pressure interpretation methods
├── well_corrections.py        # Wellbore correction utilities
├── well_corrections_old.py    # Legacy correction code (archival reference)
├── time_difference.py         # Time alignment utilities (surface ↔ downhole)
└── Data/Amstelland/           # Raw input data files (.txt, .csv)
```

---

## ⚙️ Requirements

- Python **3.9+**
- Dependencies (install via pip):

```bash
pip install pandas numpy matplotlib
```

Optional:  
Create a reproducible environment with `conda`:

```bash
conda create -n xlot python=3.9 pandas numpy matplotlib
conda activate xlot
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
   - Corrected data tables (Pandas DataFrames)  
   - Plots of fracture initiation, pressure decline, and closure interpretation  
   - Console log with key parameters (ISIP, closure pressure, lag time, etc.)

---

## 📌 Notes

- **Cross-platform paths**:  
  Paths are handled via [`pathlib`](https://docs.python.org/3/library/pathlib.html) to ensure compatibility on **Windows** (`\`) and **macOS/Linux** (`/`).

- **Case sensitivity**:  
  On macOS/Linux, file extensions (`.txt` vs `.TXT`) must match exactly.

- **Legacy code**:  
  `well_corrections_old.py` is retained for reference but not actively used.

---

## 🔬 Background

Extended Leak-Off Tests (XLOTs) are performed in geothermal and petroleum wells to determine:
- **Minimum horizontal stress (Shmin)**
- **Fracture closure pressure**
- **Fracture propagation behavior**

This workflow integrates data cleaning, corrections, time alignment, and multiple closure analysis methods to enable reproducible and transparent stress test interpretation.

---

## 🧑‍💻 Authors

- Developed for **EBN BV / TU Delft projects**  
- Scripts maintained and iteratively improved for geothermal and geomechanics research

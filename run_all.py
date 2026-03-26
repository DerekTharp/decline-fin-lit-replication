"""
Master script: reproduces all results from raw HRS data.

Usage:
    python run_all.py

Prerequisites:
    1. Install dependencies: pip install -r requirements.txt
    2. Place HRS data files in the HRS/ directory (see README.md)
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

PIPELINE = [
    ("00_inspect_finlit_2010.py", "Instrument inspection"),
    ("01_build_analytic_sample.py", "Build analytic sample"),
    ("02_replicate_table9.py", "Replicate Table 9"),
    ("03_extract_longitudinal_cognition.py", "Extract longitudinal cognition"),
    ("04_estimate_cognitive_trajectories.py", "Estimate cognitive trajectories"),
    ("05_cognitive_slopes_predict_finlit.py", "Cognitive slopes predict financial literacy"),
    ("06_robustness.py", "Robustness checks"),
    ("07_figures_and_tables.py", "Generate figures and tables"),
]


def run_script(filename, description):
    path = SCRIPTS_DIR / filename
    print(f"\n{'='*60}")
    print(f"  Running: {filename}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(SCRIPTS_DIR),
    )
    if result.returncode != 0:
        print(f"\nFAILED: {filename} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  Completed: {filename}")


if __name__ == "__main__":
    print("Finke et al. (2017) HRS Replication Pipeline")
    print(f"Python: {sys.version}")
    print(f"Scripts directory: {SCRIPTS_DIR}")

    for filename, description in PIPELINE:
        run_script(filename, description)

    print(f"\n{'='*60}")
    print("  All scripts completed successfully.")
    print(f"{'='*60}")

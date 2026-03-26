"""
00: Instrument Reconstruction

Inspect the 2010 HRS fat file financial literacy module variables.
Document value labels, coding, missingness, and DK/Refused patterns.

Inputs:  HRS 2010 fat file (hd10f6b.dta)
Outputs: output/00_finlit_2010_inspection.txt
"""

import pandas as pd
from config import (
    make_logger,
    FAT2010, OUTPUT_DIR, FINLIT_SCORING, FINLIT_HRS_VARS,
    FAT2010_EXTRA_VARS, DK_CODE, RF_CODE, ensure_dirs, check_raw_data,
)

ensure_dirs()
log, lines = make_logger()
check_raw_data()

# -- Load -----------------------------------------------------------------

all_vars_to_load = FAT2010_EXTRA_VARS + FINLIT_HRS_VARS
reader = pd.io.stata.StataReader(FAT2010)
var_labels = reader.variable_labels()
df = pd.read_stata(FAT2010, columns=all_vars_to_load)
reader.close()



log(f"2010 HRS Fat File: {len(df):,} total respondents")

# -- Tabulate each financial literacy item --------------------------------

log(f"\n{'='*70}")
log("  Value Distributions for Financial Literacy Items")
log(f"{'='*70}")

for var in FINLIT_HRS_VARS:
    info = FINLIT_SCORING[var]
    label = var_labels.get(var, "")
    log(f"\n  {var}: {label}")
    log(f"  Question: {info['question']}")
    log(f"  Correct answer: code {info['correct_val']}")

    col = df[var]
    vc = col.value_counts(dropna=False).sort_index()
    n_miss = col.isna().sum()

    for val, ct in vc.items():
        pct = 100 * ct / len(df)
        marker = ""
        if pd.isna(val):
            marker = " [SYSTEM MISSING]"
        elif val == DK_CODE:
            marker = " [DK -> scored 0]"
        elif val == RF_CODE:
            marker = " [RF -> scored 0]"
        elif val == info["correct_val"]:
            marker = " [CORRECT]"
        log(f"    {val:>6}  {ct:>6,}  ({pct:5.1f}%){marker}")

    log(f"    Non-missing: {len(df) - n_miss:,}  |  Missing: {n_miss:,}")

# -- Module participation -------------------------------------------------

finke_df = df[FINLIT_HRS_VARS]
n_nonmiss = finke_df.notna().sum(axis=1)

log(f"\n{'='*70}")
log("  Module Participation")
log(f"{'='*70}")
log(f"  Total respondents in fat file:    {len(df):,}")
log(f"  With at least 1 item non-missing: {(n_nonmiss > 0).sum():,}")
log(f"  With all 7 items non-missing:     {(n_nonmiss == 7).sum():,}")

# -- Save -----------------------------------------------------------------

outpath = OUTPUT_DIR / "00_finlit_2010_inspection.txt"
with open(outpath, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {outpath}")

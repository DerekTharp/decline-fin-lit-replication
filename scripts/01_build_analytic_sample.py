"""
01: Build Analytic Sample

Score the 7 Finke financial literacy items, merge with RAND longitudinal
file for demographics and cognition, restrict to age 60+, and save.

Inputs:  HRS 2010 fat file, RAND longitudinal file
Outputs: data/analytic_2010_age60plus.parquet
         output/01_sample_construction.txt
"""

import pandas as pd
import numpy as np
from config import (
    make_logger,
    FAT2010, RAND_LONGITUDINAL, DATA_DIR, OUTPUT_DIR,
    FINLIT_SCORING, FINLIT_HRS_VARS, FINLIT_ITEM_NAMES,
    BIG3_ITEMS, FAT2010_EXTRA_VARS, DK_CODE, RF_CODE,
    RAND_VARS, AGE_LOWER_BOUND,
    EXPECTED_N_FINLIT, EXPECTED_N_COGNITION, SAMPLE_SIZE_TOLERANCE,
    ANALYTIC_2010, ensure_dirs, check_raw_data,
)

ensure_dirs()
log, lines = make_logger()
check_raw_data()



# ===========================================================================
# 1. LOAD AND SCORE FINANCIAL LITERACY
# ===========================================================================

log("Loading 2010 fat file...")
cols_to_load = FAT2010_EXTRA_VARS + FINLIT_HRS_VARS
df_fat = pd.read_stata(FAT2010, columns=cols_to_load)

# Keep respondents who received the module (all-or-nothing assignment)
df_fl = df_fat.dropna(subset=["mv351"]).copy()
log(f"Module participants: {len(df_fl):,}")

# Score each item: correct=1, everything else (wrong, DK=8, RF=9) = 0
for var, info in FINLIT_SCORING.items():
    col = df_fl[var].astype(float)
    df_fl[info["name"]] = (col == info["correct_val"]).astype(int)

# Composite scores
df_fl["finlit7_correct"] = df_fl[FINLIT_ITEM_NAMES].sum(axis=1)
df_fl["finlit7_pct"] = df_fl["finlit7_correct"] / 7 * 100

df_fl["finlit3_correct"] = df_fl[BIG3_ITEMS].sum(axis=1)
df_fl["finlit3_pct"] = df_fl["finlit3_correct"] / 3 * 100

# Confidence: 1-7 scale; 8/9 are DK/RF -> missing
df_fl["confidence"] = df_fl["mv347"].astype(float)
df_fl.loc[df_fl["confidence"] >= DK_CODE, "confidence"] = np.nan

log(f"\nScoring summary (all module participants, n={len(df_fl):,}):")
log(f"  finlit7 mean: {df_fl['finlit7_pct'].mean():.1f}%  sd: {df_fl['finlit7_pct'].std():.1f}")
for var, info in FINLIT_SCORING.items():
    pct = df_fl[info["name"]].mean() * 100
    n_dk = (df_fl[var].astype(float) == DK_CODE).sum()
    log(f"  {info['name']:25s}: {pct:5.1f}% correct  ({n_dk:,} DK)")

# ===========================================================================
# 2. LOAD DEMOGRAPHICS AND COGNITION FROM RAND
# ===========================================================================

log("\nLoading RAND longitudinal file...")
df_rand = pd.read_stata(RAND_LONGITUDINAL, columns=RAND_VARS, convert_categoricals=False)
log(f"RAND file: {len(df_rand):,} respondents")

# ===========================================================================
# 3. MERGE
# ===========================================================================

# Construct hhidpn in the fat file to match RAND's integer key
df_fl["hhid_str"] = df_fl["hhid"].astype(str).str.strip()
df_fl["pn_str"] = df_fl["pn"].astype(str).str.strip().str.zfill(3)
df_fl["hhidpn"] = pd.to_numeric(df_fl["hhid_str"] + df_fl["pn_str"], errors="coerce")

df = df_fl.merge(df_rand, on="hhidpn", how="left", indicator=True)

n_matched = (df["_merge"] == "both").sum()
n_unmatched = (df["_merge"] == "left_only").sum()
log(f"\nMerge: {n_matched:,} matched, {n_unmatched:,} unmatched")
assert n_unmatched == 0, f"{n_unmatched} finlit respondents failed to merge with RAND"

# ===========================================================================
# 4. RESTRICT TO AGE 60+
# ===========================================================================

df["age"] = pd.to_numeric(df["r10agey_e"], errors="coerce")
df60 = df[df["age"] >= AGE_LOWER_BOUND].copy()

# Verify sample sizes against Finke et al.
n_finlit = len(df60)
has_recall = df60["r10tr20"].notna()
has_vocab = df60["r10vocab"].notna()
n_cog = (has_recall & has_vocab).sum()

log(f"\nSample size verification:")
log(f"  Age 60+ with finlit:         {n_finlit:,}  (Finke: {EXPECTED_N_FINLIT:,}, diff: {n_finlit - EXPECTED_N_FINLIT:+d})")
log(f"  Age 60+ with finlit + cog:   {n_cog:,}  (Finke: {EXPECTED_N_COGNITION:,}, diff: {n_cog - EXPECTED_N_COGNITION:+d})")

assert abs(n_finlit - EXPECTED_N_FINLIT) <= SAMPLE_SIZE_TOLERANCE, (
    f"Finlit sample size {n_finlit} deviates from expected {EXPECTED_N_FINLIT} "
    f"by more than {SAMPLE_SIZE_TOLERANCE}"
)
assert abs(n_cog - EXPECTED_N_COGNITION) <= SAMPLE_SIZE_TOLERANCE, (
    f"Cognition sample size {n_cog} deviates from expected {EXPECTED_N_COGNITION} "
    f"by more than {SAMPLE_SIZE_TOLERANCE}"
)

# ===========================================================================
# 5. SAVE
# ===========================================================================

keep_cols = (
    ["hhidpn", "age"]
    + FINLIT_ITEM_NAMES
    + ["finlit7_correct", "finlit7_pct", "finlit3_correct", "finlit3_pct", "confidence"]
    + [v for v in RAND_VARS if v != "hhidpn"]
)
# Deduplicate while preserving order, and only keep columns that exist
seen = set()
keep_cols = [c for c in keep_cols if c in df60.columns and not (c in seen or seen.add(c))]

save_df = df60[keep_cols].copy()
save_df.to_parquet(ANALYTIC_2010)
log(f"\nSaved: {ANALYTIC_2010}")
log(f"  n={len(save_df):,}, columns={len(save_df.columns)}")

# Save construction log
logpath = OUTPUT_DIR / "01_sample_construction.txt"
with open(logpath, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {logpath}")

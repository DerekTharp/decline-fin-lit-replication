"""
03: Extract Longitudinal Cognition Data (Waves 3-9, 1996-2008)

Reshape RAND HRS data from wide to person-wave long format for
cognitive trajectory estimation. Only uses pre-2010 waves so that
trajectories causally precede the 2010 financial literacy measurement.

Inputs:  RAND longitudinal file (randhrs1992_2022v1.dta)
Outputs: data/cognition_long_waves3to9.parquet
         output/03_longitudinal_cognition.txt
"""

import pandas as pd
import numpy as np
from config import (
    make_logger,
    RAND_LONGITUDINAL, DATA_DIR, OUTPUT_DIR,
    LONGITUDINAL_RAND_VARS, PRE2010_WAVES, PRE2010_YEARS,
    LONGITUDINAL_VARS_PER_WAVE, COG_MEASURES, MIN_WAVES_FOR_SLOPE,
    LONGITUDINAL_LONG, ensure_dirs, check_raw_data,
)

ensure_dirs()
log, lines = make_logger()
check_raw_data()



# ===========================================================================
# 1. LOAD WIDE-FORMAT DATA
# ===========================================================================

log("Loading RAND longitudinal file (waves 3-9 cognition)...")
df_wide = pd.read_stata(RAND_LONGITUDINAL, columns=LONGITUDINAL_RAND_VARS, convert_categoricals=False)
log(f"Loaded {len(df_wide):,} respondents, {len(df_wide.columns)} columns")

# ===========================================================================
# 2. RESHAPE TO LONG FORMAT
# ===========================================================================

log("\nReshaping to person-wave long format...")

# Time-invariant columns
id_vars = ["hhidpn", "rabyear", "ragender", "raracem", "rahispan", "raedyrs", "raeduc"]

# Build long format by stacking waves
wave_frames = []
for w in PRE2010_WAVES:
    year = PRE2010_YEARS[w]
    wave_cols = {f"r{w}{var}": var for var in LONGITUDINAL_VARS_PER_WAVE}

    sub = df_wide[id_vars + list(wave_cols.keys())].copy()
    sub = sub.rename(columns=wave_cols)
    sub["wave"] = w
    sub["year"] = year

    wave_frames.append(sub)

df_long = pd.concat(wave_frames, ignore_index=True)

# Convert cognition columns to numeric
for var in LONGITUDINAL_VARS_PER_WAVE:
    df_long[var] = pd.to_numeric(df_long[var], errors="coerce")

df_long["age"] = pd.to_numeric(df_long["agey_e"], errors="coerce")

log(f"Long format: {len(df_long):,} person-wave rows ({len(df_wide):,} persons x {len(PRE2010_WAVES)} waves)")

# ===========================================================================
# 3. DROP ROWS WITH NO COGNITIVE DATA
# ===========================================================================

# A row is useful if the person has at least one cognitive score in that wave
cog_cols = list(COG_MEASURES.keys())
df_long["has_any_cog"] = df_long[cog_cols].notna().any(axis=1)
df_long["has_tr20"] = df_long["tr20"].notna()

n_before = len(df_long)
df_long = df_long[df_long["has_any_cog"]].copy()
log(f"\nDropped {n_before - len(df_long):,} rows with no cognitive data")
log(f"Remaining: {len(df_long):,} person-wave rows")

# ===========================================================================
# 4. COUNT WAVES PER PERSON
# ===========================================================================

# Count waves with total word recall (the primary trajectory measure)
waves_per_person = df_long.groupby("hhidpn")["has_tr20"].sum().rename("n_waves_tr20")
df_long = df_long.merge(waves_per_person, on="hhidpn", how="left")

# Also count total waves with any cog data
any_cog_per_person = df_long.groupby("hhidpn").size().rename("n_waves_any")
df_long = df_long.merge(any_cog_per_person, on="hhidpn", how="left")

log(f"\nWaves of word recall data per person:")
dist = waves_per_person.value_counts().sort_index()
for n, ct in dist.items():
    marker = " *" if n >= MIN_WAVES_FOR_SLOPE else ""
    log(f"  {int(n)} waves: {ct:>6,} persons{marker}")

n_enough = (waves_per_person >= MIN_WAVES_FOR_SLOPE).sum()
log(f"\nPersons with {MIN_WAVES_FOR_SLOPE}+ waves of word recall: {n_enough:,}")

# ===========================================================================
# 5. DESCRIPTIVE STATISTICS
# ===========================================================================

log(f"\n{'='*60}")
log("  COGNITION BY WAVE (person-wave level)")
log(f"{'='*60}")

for measure, info in COG_MEASURES.items():
    log(f"\n  {info['label']}:")
    log(f"  {'Wave':>6s}  {'Year':>6s}  {'N':>8s}  {'Mean':>8s}  {'SD':>8s}")
    for w in PRE2010_WAVES:
        mask = df_long["wave"] == w
        col = df_long.loc[mask, measure]
        n = col.notna().sum()
        if n > 0:
            log(f"  {w:>6d}  {PRE2010_YEARS[w]:>6d}  {n:>8,}  {col.mean():>8.2f}  {col.std():>8.2f}")
        else:
            log(f"  {w:>6d}  {PRE2010_YEARS[w]:>6d}  {0:>8,}  {'—':>8s}  {'—':>8s}")

# Age distribution across waves
log(f"\n  Age at interview by wave:")
log(f"  {'Wave':>6s}  {'Year':>6s}  {'N':>8s}  {'Mean age':>10s}  {'Min':>6s}  {'Max':>6s}")
for w in PRE2010_WAVES:
    mask = df_long["wave"] == w
    col = df_long.loc[mask, "age"]
    n = col.notna().sum()
    if n > 0:
        log(f"  {w:>6d}  {PRE2010_YEARS[w]:>6d}  {n:>8,}  {col.mean():>10.1f}  {col.min():>6.0f}  {col.max():>6.0f}")

# ===========================================================================
# 6. SAVE
# ===========================================================================

keep_cols = (
    ["hhidpn", "wave", "year", "age",
     "rabyear", "ragender", "raracem", "rahispan", "raedyrs", "raeduc"]
    + list(COG_MEASURES.keys())
    + ["proxy", "n_waves_tr20", "n_waves_any"]
)
save_df = df_long[keep_cols].copy()
save_df.to_parquet(LONGITUDINAL_LONG)
log(f"\nSaved: {LONGITUDINAL_LONG}")
log(f"  {len(save_df):,} rows, {save_df['hhidpn'].nunique():,} unique persons")

logpath = OUTPUT_DIR / "03_longitudinal_cognition.txt"
with open(logpath, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {logpath}")

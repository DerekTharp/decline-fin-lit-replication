"""
04: Estimate Individual Cognitive Trajectories (Mixed-Effects Models)

Fit random-intercept/random-slope growth models to pre-2010 cognition data.
Extract empirical Bayes estimates of each person's cognitive level (intercept)
and rate of change (slope). These become the key predictors for Phase 3.

Model: COG_it = (b0 + u0i) + (b1 + u1i) * age_centered_it + e_it

Age is centered at 70 so that the random intercept represents predicted
cognition at age 70 (the middle of our 60+ target range).

Inputs:  data/cognition_long_waves3to9.parquet
Outputs: data/cognitive_slopes.parquet
         output/04_cognitive_trajectories.txt
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from config import (
    make_logger,
    DATA_DIR, OUTPUT_DIR, LONGITUDINAL_LONG, MIN_WAVES_FOR_SLOPE,
    ensure_dirs,
)

ensure_dirs()
log, lines = make_logger()



# ===========================================================================
# 1. LOAD AND PREPARE
# ===========================================================================

log("Loading longitudinal cognition data...")
df = pd.read_parquet(LONGITUDINAL_LONG)
log(f"Loaded: {len(df):,} person-wave rows, {df['hhidpn'].nunique():,} persons")

# Center age at 70 for interpretable intercepts
AGE_CENTER = 70
df["age_c"] = df["age"] - AGE_CENTER

# Birth cohort (5-year bins for fixed effects)
df["cohort5"] = (df["rabyear"] // 5) * 5

# ===========================================================================
# 2. ESTIMATE TRAJECTORIES FOR EACH COGNITION MEASURE
# ===========================================================================

trajectory_measures = {
    "tr20": {"label": "Total word recall (0-20)", "min_waves_col": "n_waves_tr20"},
    "ser7": {"label": "Serial 7s (0-5)", "min_waves_col": None},
}

all_slopes = []

for measure, info in trajectory_measures.items():
    log(f"\n{'='*60}")
    log(f"  TRAJECTORY MODEL: {info['label']}")
    log(f"{'='*60}")

    # Restrict to persons with enough waves of this measure
    df_m = df.dropna(subset=[measure, "age_c"]).copy()

    # Count waves per person for this measure
    waves_pp = df_m.groupby("hhidpn").size()
    eligible = waves_pp[waves_pp >= MIN_WAVES_FOR_SLOPE].index
    df_m = df_m[df_m["hhidpn"].isin(eligible)].copy()

    n_persons = df_m["hhidpn"].nunique()
    n_obs = len(df_m)
    log(f"\n  Persons with {MIN_WAVES_FOR_SLOPE}+ waves: {n_persons:,}")
    log(f"  Total observations: {n_obs:,}")
    log(f"  Mean waves per person: {n_obs/n_persons:.1f}")

    # -- Fit mixed-effects model ------------------------------------------
    #
    # Random intercept + random slope on age_c, grouped by person.
    # Fixed effects: age_c only (we want person-specific deviations).
    #
    # This gives us:
    #   u0i = person i's predicted cognition at age 70 (relative to mean)
    #   u1i = person i's annual rate of cognitive change (relative to mean)
    #
    log(f"\n  Fitting mixed-effects model...")
    log(f"    {measure} ~ age_c, random = ~age_c | hhidpn")

    model = smf.mixedlm(
        f"{measure} ~ age_c",
        data=df_m,
        groups=df_m["hhidpn"],
        re_formula="~age_c",
    )
    result = model.fit(reml=True, method="lbfgs")

    log(f"\n  Fixed effects:")
    log(f"    Intercept (mean {measure} at age {AGE_CENTER}): {result.fe_params['Intercept']:.3f}")
    log(f"    age_c (mean annual change): {result.fe_params['age_c']:.4f}")
    log(f"  Random effects covariance:")
    log(f"    Var(intercept): {result.cov_re.iloc[0,0]:.4f}")
    log(f"    Var(slope):     {result.cov_re.iloc[1,1]:.6f}")
    log(f"    Cov(int,slope): {result.cov_re.iloc[0,1]:.6f}")
    log(f"  Residual variance: {result.scale:.4f}")
    log(f"  Log-likelihood: {result.llf:.1f}")

    # -- Extract empirical Bayes estimates --------------------------------
    re = result.random_effects  # dict: hhidpn -> Series with Group, age_c
    re_df = pd.DataFrame({
        "hhidpn": list(re.keys()),
        f"{measure}_intercept": [re[k]["Group"] for k in re.keys()],
        f"{measure}_slope": [re[k]["age_c"] for k in re.keys()],
    })

    # Add fixed effects to get person-specific predicted values
    re_df[f"{measure}_level"] = result.fe_params["Intercept"] + re_df[f"{measure}_intercept"]
    re_df[f"{measure}_change"] = result.fe_params["age_c"] + re_df[f"{measure}_slope"]

    all_slopes.append(re_df)

    # -- Descriptive statistics of slopes ---------------------------------
    log(f"\n  Distribution of person-specific annual change ({measure}):")
    change = re_df[f"{measure}_change"]
    log(f"    Mean:   {change.mean():.4f}")
    log(f"    SD:     {change.std():.4f}")
    log(f"    Min:    {change.min():.4f}")
    log(f"    Q25:    {change.quantile(0.25):.4f}")
    log(f"    Median: {change.median():.4f}")
    log(f"    Q75:    {change.quantile(0.75):.4f}")
    log(f"    Max:    {change.max():.4f}")

    n_decline = (change < 0).sum()
    log(f"    Declining ({measure}_change < 0): {n_decline:,} ({100*n_decline/len(change):.1f}%)")

    log(f"\n  Distribution of person-specific level at age {AGE_CENTER} ({measure}):")
    level = re_df[f"{measure}_level"]
    log(f"    Mean:   {level.mean():.2f}")
    log(f"    SD:     {level.std():.2f}")

# ===========================================================================
# 3. MERGE AND SAVE
# ===========================================================================

# Merge all trajectory measures into one person-level file
slopes_df = all_slopes[0]
for extra in all_slopes[1:]:
    slopes_df = slopes_df.merge(extra, on="hhidpn", how="outer")

slopes_path = DATA_DIR / "cognitive_slopes.parquet"
slopes_df.to_parquet(slopes_path)
log(f"\nSaved: {slopes_path}")
log(f"  {len(slopes_df):,} persons with trajectory estimates")

# -- Quick validation: do slopes correlate with known predictors? ---------
log(f"\n{'='*60}")
log("  VALIDATION: Slope correlates with education and age")
log(f"{'='*60}")

# Merge education
demo = pd.read_parquet(LONGITUDINAL_LONG)[["hhidpn", "raedyrs"]].drop_duplicates("hhidpn")
val = slopes_df.merge(demo, on="hhidpn", how="left")
val["raedyrs"] = pd.to_numeric(val["raedyrs"], errors="coerce")

for measure in trajectory_measures:
    r_educ = val[[f"{measure}_level", "raedyrs"]].dropna().corr().iloc[0, 1]
    r_slope_educ = val[[f"{measure}_change", "raedyrs"]].dropna().corr().iloc[0, 1]
    log(f"\n  {measure}:")
    log(f"    r(level, education years): {r_educ:.3f}  (expect positive)")
    log(f"    r(slope, education years): {r_slope_educ:.3f}  (expect ~0 or small positive)")

logpath = OUTPUT_DIR / "04_cognitive_trajectories.txt"
with open(logpath, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {logpath}")

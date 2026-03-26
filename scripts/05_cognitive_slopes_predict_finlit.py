"""
05: Do Pre-2010 Cognitive Slopes Predict 2010 Financial Literacy?

This is the core contribution. Finke et al. cannot distinguish age-related
cognitive decline from cohort effects because they use cross-sectional data.
We test: within birth cohorts, do people whose cognition declined more
steeply (pre-2010) have lower financial literacy (in 2010)?

Key model:
    finlit_i = a + b1*cog_level_i + b2*cog_slope_i + b3*age_i
               + cohort_FE + controls + e_i

If b2 < 0 and significant: cognitive decline predicts financial literacy
within cohorts, supporting Finke's interpretation.

If b2 ≈ 0: the cross-sectional age-finlit gradient is mostly cohort
differences, not cognitive decline.

Inputs:  data/analytic_2010_age60plus.parquet
         data/cognitive_slopes.parquet
Outputs: output/05_slopes_predict_finlit.csv
         output/05_slopes_predict_finlit.txt
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from config import (
    make_logger,
    ANALYTIC_2010, DATA_DIR, OUTPUT_DIR,
    EDUC_MAP, EDUC_CATEGORIES, EDUC_REFERENCE,
    HIGH_INCOME_WEALTH_QUANTILE, AGE_LOWER_BOUND,
    ensure_dirs,
)

ensure_dirs()
log, lines = make_logger()



# ===========================================================================
# 1. LOAD AND MERGE
# ===========================================================================

log("Loading analytic datasets...")
df_fl = pd.read_parquet(ANALYTIC_2010)
df_slopes = pd.read_parquet(DATA_DIR / "cognitive_slopes.parquet")

log(f"  Financial literacy sample (age 60+): {len(df_fl):,}")
log(f"  Persons with cognitive slopes: {len(df_slopes):,}")

df = df_fl.merge(df_slopes, on="hhidpn", how="inner")
log(f"  Merged (inner join): {len(df):,}")

# ===========================================================================
# 2. PREPARE VARIABLES
# ===========================================================================

df["finlit_pct"] = df["finlit7_pct"]
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Birth cohort fixed effects (5-year bins)
df["rabyear"] = pd.to_numeric(df["rabyear"], errors="coerce")
df["cohort5"] = (df["rabyear"] // 5) * 5

# Controls (same as Table 9 replication)
df["educ_cat"] = pd.Categorical(
    pd.to_numeric(df["raeduc"], errors="coerce").map(EDUC_MAP),
    categories=EDUC_CATEGORIES,
)
df["male"] = (pd.to_numeric(df["ragender"], errors="coerce") == 1).astype(int)
df["white"] = (
    (pd.to_numeric(df["raracem"], errors="coerce") == 1)
    & (pd.to_numeric(df["rahispan"], errors="coerce") == 0)
).astype(int)
df["married"] = pd.to_numeric(df["r10mstat"], errors="coerce").isin([1, 2, 3]).astype(int)
df["homeowner"] = (pd.to_numeric(df["h10ahous"], errors="coerce") > 0).astype(int)
df["stock_owner"] = (pd.to_numeric(df["h10astck"], errors="coerce") > 0).astype(int)
df["tax_sheltered"] = (pd.to_numeric(df["h10aira"], errors="coerce") > 0).astype(int)

for var, name in [("h10atotw", "wealth"), ("h10itot", "income")]:
    col = pd.to_numeric(df[var], errors="coerce")
    df[f"high_{name}"] = (col >= col.quantile(HIGH_INCOME_WEALTH_QUANTILE)).astype(int)

# Cognitive trajectory variables (from script 04)
# tr20_level: predicted word recall at age 70
# tr20_change: person-specific annual change in word recall
# ser7_level, ser7_change: same for serial 7s

# Drop rows with missing controls
ctrl_cols = ["finlit_pct", "age", "male", "white", "married", "homeowner",
             "stock_owner", "tax_sheltered", "high_income", "high_wealth",
             "educ_cat", "cohort5", "tr20_level", "tr20_change"]
df_m = df.dropna(subset=ctrl_cols).copy()
log(f"  Analysis sample (non-missing controls + trajectories): {len(df_m):,}")

# Descriptives
log(f"\n  Cognitive slope descriptives (analysis sample):")
for var in ["tr20_level", "tr20_change", "ser7_level", "ser7_change"]:
    if var in df_m.columns:
        col = df_m[var]
        log(f"    {var:20s}: mean={col.mean():.4f}, sd={col.std():.4f}")

log(f"\n  Birth cohort distribution:")
cohort_dist = df_m["cohort5"].value_counts().sort_index()
for c, n in cohort_dist.items():
    log(f"    {int(c)}-{int(c)+4}: n={n:,}")

# ===========================================================================
# 3. MODEL SPECIFICATIONS
# ===========================================================================

ref_educ = f"C(educ_cat, Treatment(reference='{EDUC_REFERENCE}'))"
ctrl_str = f"male + white + married + homeowner + stock_owner + tax_sheltered + high_income + high_wealth + {ref_educ}"
cohort_fe = "C(cohort5)"

results_rows = []

log(f"\n{'='*70}")
log("  CORE MODELS: Cognitive Slopes Predict Financial Literacy")
log(f"{'='*70}")

models = {
    # Baseline: replicate cross-sectional age effect
    "M1_age_only": f"finlit_pct ~ age",
    "M2_age_controls": f"finlit_pct ~ age + {ctrl_str}",
    # Add cohort FE: what remains of "age" after absorbing cohort?
    "M3_age_cohortFE": f"finlit_pct ~ age + {cohort_fe} + {ctrl_str}",
    # Core test: cognitive level and slope, with cohort FE
    "M4_cog_level_only": f"finlit_pct ~ tr20_level + {cohort_fe} + {ctrl_str}",
    "M5_cog_slope_only": f"finlit_pct ~ tr20_change + {cohort_fe} + {ctrl_str}",
    "M6_full": f"finlit_pct ~ tr20_level + tr20_change + age + {cohort_fe} + {ctrl_str}",
    # Alternative: serial 7s instead of word recall
    "M7_ser7": f"finlit_pct ~ ser7_level + ser7_change + age + {cohort_fe} + {ctrl_str}",
    # Both measures
    "M8_both": f"finlit_pct ~ tr20_level + tr20_change + ser7_level + ser7_change + age + {cohort_fe} + {ctrl_str}",
}

# For models with ser7, ensure non-missing
df_ser7 = df_m.dropna(subset=["ser7_level", "ser7_change"]).copy()

for label, formula in models.items():
    data = df_ser7 if "ser7" in label else df_m
    m = smf.ols(formula, data=data).fit()

    log(f"\n--- {label} (n={int(m.nobs):,}, R2={m.rsquared:.3f}, adj-R2={m.rsquared_adj:.3f}) ---")

    # Report key variables (skip cohort dummies and education for readability)
    key_vars = ["age", "tr20_level", "tr20_change", "ser7_level", "ser7_change",
                "male", "white", "married", "homeowner", "stock_owner",
                "tax_sheltered", "high_income", "high_wealth"]

    for param in m.params.index:
        coef = m.params[param]
        se = m.bse[param]
        p = m.pvalues[param]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

        # Only print key variables (not intercept, not cohort/educ dummies)
        is_key = any(param == kv or param.startswith(kv) for kv in key_vars)
        is_educ = "educ_cat" in param
        if is_key or is_educ:
            name = param
            if "educ_cat" in param and "[T." in param:
                name = "educ_" + param.split("[T.")[1].rstrip("]")
            log(f"  {name:25s}  {coef:8.3f} ({se:.3f}){sig}")

        results_rows.append({
            "model": label,
            "variable": param,
            "coef": round(coef, 5),
            "se": round(se, 5),
            "pvalue": round(p, 5),
            "n": int(m.nobs),
            "r2": round(m.rsquared, 4),
            "adj_r2": round(m.rsquared_adj, 4),
        })

# ===========================================================================
# 4. DECOMPOSITION: How much of the age-finlit gradient survives?
# ===========================================================================

log(f"\n{'='*70}")
log("  DECOMPOSITION OF THE AGE-FINLIT GRADIENT")
log(f"{'='*70}")

# All estimated on the same sample (df_m)
m_base = smf.ols(f"finlit_pct ~ age + {ctrl_str}", data=df_m).fit()
m_cohort = smf.ols(f"finlit_pct ~ age + {cohort_fe} + {ctrl_str}", data=df_m).fit()
m_full = smf.ols(f"finlit_pct ~ tr20_level + tr20_change + age + {cohort_fe} + {ctrl_str}", data=df_m).fit()

age_base = m_base.params["age"]
age_cohort = m_cohort.params["age"]
age_full = m_full.params["age"]

log(f"\n  Age coefficient progression (same sample, n={int(m_base.nobs):,}):")
log(f"    M2 (age + controls):                    {age_base:.3f}")
log(f"    M3 (+ cohort FE):                       {age_cohort:.3f}  ({100*(1-age_cohort/age_base):+.1f}% vs M2)")
log(f"    M6 (+ cohort FE + cog level + slope):   {age_full:.3f}  ({100*(1-age_full/age_base):+.1f}% vs M2)")

# ===========================================================================
# 5. SAVE
# ===========================================================================

results_df = pd.DataFrame(results_rows)
csv_path = OUTPUT_DIR / "05_slopes_predict_finlit.csv"
results_df.to_csv(csv_path, index=False)
log(f"\nSaved: {csv_path}")

logpath = OUTPUT_DIR / "05_slopes_predict_finlit.txt"
with open(logpath, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {logpath}")

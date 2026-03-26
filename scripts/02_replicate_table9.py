"""
02: Replicate Finke et al. Table 9

HRS Regressions of Financial Literacy Scores, Age, and Cognitive Ability.
Six model specifications: univariate, with controls, with controls + cognition,
each with linear age and 5-year age dummies.

Inputs:  data/analytic_2010_age60plus.parquet
Outputs: output/02_table9_replication.csv
         output/02_table9_replication.txt
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from config import (
    make_logger,
    ANALYTIC_2010, OUTPUT_DIR,
    EDUC_MAP, EDUC_CATEGORIES, EDUC_REFERENCE,
    HIGH_INCOME_WEALTH_QUANTILE, AGE_LOWER_BOUND, AGE_UPPER_BOUND,
    ensure_dirs,
)

ensure_dirs()
log, lines = make_logger()




# ===========================================================================
# 1. LOAD AND PREPARE
# ===========================================================================

df = pd.read_parquet(ANALYTIC_2010)
log(f"Loaded n={len(df):,} (age {AGE_LOWER_BOUND}+)")

# Dependent variable
df["finlit_pct"] = df["finlit7_pct"]

# Age bins (60-64 as reference)
df["age_cat"] = pd.cut(
    df["age"],
    bins=[59, 64, 69, 74, 79, 84, 89, AGE_UPPER_BOUND],
    labels=["60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"],
)

# Education
df["educ_cat"] = pd.Categorical(
    df["raeduc"].map(EDUC_MAP), categories=EDUC_CATEGORIES
)

# Binary controls
df["male"] = (df["ragender"] == 1).astype(int)
df["white"] = ((df["raracem"] == 1) & (df["rahispan"] == 0)).astype(int)  # non-Hispanic white
df["married"] = df["r10mstat"].isin([1, 2, 3]).astype(int)
df["homeowner"] = (pd.to_numeric(df["h10ahous"], errors="coerce") > 0).astype(int)
df["stock_owner"] = (pd.to_numeric(df["h10astck"], errors="coerce") > 0).astype(int)
df["tax_sheltered"] = (pd.to_numeric(df["h10aira"], errors="coerce") > 0).astype(int)

# Top-quintile dummies for income and wealth
for var, name in [("h10atotw", "wealth"), ("h10itot", "income")]:
    col = pd.to_numeric(df[var], errors="coerce")
    df[f"high_{name}"] = (col >= col.quantile(HIGH_INCOME_WEALTH_QUANTILE)).astype(int)

# Cognition
df["word_recall"] = pd.to_numeric(df["r10tr20"], errors="coerce")
df["vocabulary"] = pd.to_numeric(df["r10vocab"], errors="coerce")

# Analysis-ready subsets
controls_list = [
    "finlit_pct", "age", "male", "white", "married", "homeowner",
    "stock_owner", "tax_sheltered", "high_income", "high_wealth", "educ_cat",
]
df_ctrl = df.dropna(subset=controls_list).copy()
df_cog = df_ctrl.dropna(subset=["word_recall", "vocabulary"]).copy()

log(f"  Controls sample: n={len(df_ctrl):,}")
log(f"  Cognition sample: n={len(df_cog):,}")

# ===========================================================================
# 2. MODEL SPECIFICATIONS
# ===========================================================================

ref_educ = f"C(educ_cat, Treatment(reference='{EDUC_REFERENCE}'))"
ref_age = "C(age_cat, Treatment(reference='60-64'))"
ctrl_str = f"male + white + married + homeowner + stock_owner + tax_sheltered + high_income + high_wealth + {ref_educ}"

models = {
    "Col1_age_univar":       ("finlit_pct ~ age",                     df),
    "Col2_agecat_univar":    (f"finlit_pct ~ {ref_age}",             df),
    "Col3_age_controls":     (f"finlit_pct ~ age + {ctrl_str}",      df_ctrl),
    "Col4_agecat_controls":  (f"finlit_pct ~ {ref_age} + {ctrl_str}", df_ctrl),
    "Col5_age_cog":          (f"finlit_pct ~ age + word_recall + vocabulary + {ctrl_str}", df_cog),
    "Col6_agecat_cog":       (f"finlit_pct ~ {ref_age} + word_recall + vocabulary + {ctrl_str}", df_cog),
}

# ===========================================================================
# 3. ESTIMATE AND REPORT
# ===========================================================================

results_rows = []

log(f"\n{'='*70}")
log("  TABLE 9 REPLICATION")
log(f"{'='*70}")

for label, (formula, data) in models.items():
    m = smf.ols(formula, data=data).fit()
    log(f"\n--- {label} (n={int(m.nobs):,}, R2={m.rsquared:.3f}, adj-R2={m.rsquared_adj:.3f}) ---")

    for param in m.params.index:
        coef = m.params[param]
        se = m.bse[param]
        p = m.pvalues[param]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

        # Clean display name
        name = param
        if "age_cat" in param and "[T." in param:
            name = "age_" + param.split("[T.")[1].rstrip("]")
        elif "educ_cat" in param and "[T." in param:
            name = "educ_" + param.split("[T.")[1].rstrip("]")

        if param != "Intercept":
            log(f"  {name:30s}  {coef:8.3f} ({se:.3f}){sig}")

        results_rows.append({
            "model": label,
            "variable": name,
            "coef": round(coef, 4),
            "se": round(se, 4),
            "pvalue": round(p, 4),
            "n": int(m.nobs),
            "r2": round(m.rsquared, 4),
            "adj_r2": round(m.rsquared_adj, 4),
        })

# ===========================================================================
# 4. AGE ATTENUATION — two definitions
# ===========================================================================

log(f"\n{'='*70}")
log("  AGE COEFFICIENT ATTENUATION")
log(f"{'='*70}")

# Definition A: Univariate age → cognition-adjusted age (Finke's "nearly half")
# This compares the broadest age gradient to the cognition-adjusted one.
m_univar_cog = smf.ols("finlit_pct ~ age", data=df_cog).fit()
m_full_cog = smf.ols(f"finlit_pct ~ age + word_recall + vocabulary + {ctrl_str}", data=df_cog).fit()
atten_A = 1 - (m_full_cog.params["age"] / m_univar_cog.params["age"])
log(f"\n  Definition A: univariate -> cognition-adjusted (n={int(m_full_cog.nobs):,})")
log(f"    Univariate age:      {m_univar_cog.params['age']:.3f}")
log(f"    With cog + controls: {m_full_cog.params['age']:.3f}")
log(f"    Attenuation:         {atten_A*100:.1f}%  (Finke: ~41%)")

# Definition B: Same-sample controls → controls+cognition (stricter)
m_ctrl_only = smf.ols(f"finlit_pct ~ age + {ctrl_str}", data=df_cog).fit()
atten_B = 1 - (m_full_cog.params["age"] / m_ctrl_only.params["age"])
log(f"\n  Definition B: controls-only -> controls+cognition (same sample)")
log(f"    Controls only:       {m_ctrl_only.params['age']:.3f}")
log(f"    With cog + controls: {m_full_cog.params['age']:.3f}")
log(f"    Attenuation:         {atten_B*100:.1f}%")

# ===========================================================================
# 5. COGNITION VARIABLE AUDIT
# ===========================================================================

log(f"\n{'='*70}")
log("  COGNITION VARIABLE AUDIT")
log(f"{'='*70}")

# Finke reports: r(finlit, word recall) ≈ 0.25, r(finlit, vocab) ≈ 0.29
# Check which cognition variable best matches their reported correlation.
cog_candidates = {
    "r10tr20 (total recall 0-20)": pd.to_numeric(df_cog["r10tr20"], errors="coerce"),
    "r10imrc (immediate recall 0-10)": pd.to_numeric(df_cog["r10imrc"], errors="coerce"),
    "r10dlrc (delayed recall 0-10)": pd.to_numeric(df_cog["r10dlrc"], errors="coerce"),
    "r10cogtot (total cognition 0-35)": pd.to_numeric(df_cog["r10cogtot"], errors="coerce"),
    "r10ser7 (serial 7s 0-5)": pd.to_numeric(df_cog["r10ser7"], errors="coerce"),
    "vocabulary (r10vocab 0-10)": df_cog["vocabulary"],
}

log(f"\n  Correlations with finlit_pct (cognition sample, n={len(df_cog):,}):")
log(f"  {'Variable':45s}  {'r':>6s}  {'Finke':>8s}")
finke_targets = {
    "r10tr20 (total recall 0-20)": "0.25",
    "r10imrc (immediate recall 0-10)": "",
    "r10dlrc (delayed recall 0-10)": "",
    "r10cogtot (total cognition 0-35)": "",
    "r10ser7 (serial 7s 0-5)": "",
    "vocabulary (r10vocab 0-10)": "0.29",
}
for label, col in cog_candidates.items():
    valid = df_cog["finlit_pct"].notna() & col.notna()
    r = df_cog.loc[valid, "finlit_pct"].corr(col[valid])
    target = finke_targets.get(label, "")
    log(f"  {label:45s}  {r:6.3f}  {target:>8s}")

# ===========================================================================
# 5. SAVE
# ===========================================================================

results_df = pd.DataFrame(results_rows)
csv_path = OUTPUT_DIR / "02_table9_replication.csv"
results_df.to_csv(csv_path, index=False)
log(f"\nSaved: {csv_path}")

txt_path = OUTPUT_DIR / "02_table9_replication.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {txt_path}")

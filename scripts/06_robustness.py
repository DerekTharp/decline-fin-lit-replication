"""
06: Robustness Checks

Tests sensitivity of the core Phase 3 finding (cognitive level predicts
financial literacy but cognitive slope does not) to:
  1. Alternative cohort bin widths (3-year, 10-year vs baseline 5-year)
  2. Item-level financial literacy analysis
  3. Subsample analyses (males, college-educated, stockowners)
  4. Minimum waves requirement for slope estimation (re-estimated, 3-6 waves)
  5. Interaction tests for heterogeneity (gender, education, stock ownership)
  6. 2010→2016 within-person Big 3 change (if sample permits)

Inputs:  data/analytic_2010_age60plus.parquet
         data/cognitive_slopes.parquet
         data/cognition_long_waves3to9.parquet
         HRS 2016 fat file (for within-person analysis)
Outputs: output/06_robustness.txt
         output/06_robustness.csv
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from config import (
    make_logger,
    ANALYTIC_2010, DATA_DIR, OUTPUT_DIR, FAT2016,
    EDUC_MAP, EDUC_CATEGORIES, EDUC_REFERENCE,
    HIGH_INCOME_WEALTH_QUANTILE, FINLIT_ITEM_NAMES, FINLIT_SCORING, BIG3_ITEMS,
    BIG3_2016_SCORING,
    ensure_dirs,
)

ensure_dirs()
log, lines = make_logger()

results_rows = []


def run_model(label, formula, data, key_vars=None):
    """Estimate OLS, log key coefficients, store results."""
    m = smf.ols(formula, data=data).fit()
    log(f"\n  {label} (n={int(m.nobs):,}, R2={m.rsquared:.3f})")
    if key_vars is None:
        key_vars = ["age", "tr20_level", "tr20_change", "ser7_level", "ser7_change"]
    for kv in key_vars:
        if kv in m.params.index:
            c, se, p = m.params[kv], m.bse[kv], m.pvalues[kv]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            log(f"    {kv:25s}  {c:8.3f} ({se:.3f}){sig}")
            results_rows.append({"check": label, "variable": kv, "coef": round(c, 5),
                                 "se": round(se, 5), "pvalue": round(p, 5),
                                 "n": int(m.nobs), "r2": round(m.rsquared, 4)})
    return m


def _estimate_tr20_slopes(df_long, min_waves):
    """Re-estimate tr20 random-slope model with a given minimum-waves threshold."""
    df = df_long.dropna(subset=["tr20", "age"]).copy()
    df["age_c"] = df["age"] - 70
    waves_pp = df.groupby("hhidpn").size()
    eligible = waves_pp[waves_pp >= min_waves].index
    df = df[df["hhidpn"].isin(eligible)]
    model = smf.mixedlm("tr20 ~ age_c", data=df, groups=df["hhidpn"], re_formula="~age_c")
    result = model.fit(reml=True, method="lbfgs")
    re = result.random_effects
    return pd.DataFrame({
        "hhidpn": list(re.keys()),
        "tr20_level": [result.fe_params["Intercept"] + re[k]["Group"] for k in re],
        "tr20_change": [result.fe_params["age_c"] + re[k]["age_c"] for k in re],
    })


# ===========================================================================
# LOAD AND PREPARE (same as script 05)
# ===========================================================================

log("Loading data...")
df_fl = pd.read_parquet(ANALYTIC_2010)
df_slopes = pd.read_parquet(DATA_DIR / "cognitive_slopes.parquet")
df = df_fl.merge(df_slopes, on="hhidpn", how="inner")

df["finlit_pct"] = df["finlit7_pct"]
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["rabyear"] = pd.to_numeric(df["rabyear"], errors="coerce")

df["educ_cat"] = pd.Categorical(
    pd.to_numeric(df["raeduc"], errors="coerce").map(EDUC_MAP), categories=EDUC_CATEGORIES)
df["male"] = (pd.to_numeric(df["ragender"], errors="coerce") == 1).astype(int)
df["white"] = ((pd.to_numeric(df["raracem"], errors="coerce") == 1) &
               (pd.to_numeric(df["rahispan"], errors="coerce") == 0)).astype(int)
df["married"] = pd.to_numeric(df["r10mstat"], errors="coerce").isin([1, 2, 3]).astype(int)
df["homeowner"] = (pd.to_numeric(df["h10ahous"], errors="coerce") > 0).astype(int)
df["stock_owner"] = (pd.to_numeric(df["h10astck"], errors="coerce") > 0).astype(int)
df["tax_sheltered"] = (pd.to_numeric(df["h10aira"], errors="coerce") > 0).astype(int)
for var, name in [("h10atotw", "wealth"), ("h10itot", "income")]:
    col = pd.to_numeric(df[var], errors="coerce")
    df[f"high_{name}"] = (col >= col.quantile(HIGH_INCOME_WEALTH_QUANTILE)).astype(int)

# Score individual items
for var, info in FINLIT_SCORING.items():
    if info["name"] not in df.columns:
        pass  # Already scored in script 01

ref_educ = f"C(educ_cat, Treatment(reference='{EDUC_REFERENCE}'))"
ctrl_str = f"male + white + married + homeowner + stock_owner + tax_sheltered + high_income + high_wealth + {ref_educ}"

# Drop missing
ctrl_cols = ["finlit_pct", "age", "male", "white", "married", "homeowner",
             "stock_owner", "tax_sheltered", "high_income", "high_wealth",
             "educ_cat", "tr20_level", "tr20_change"]
df_m = df.dropna(subset=ctrl_cols).copy()
log(f"Analysis sample: n={len(df_m):,}")

# ===========================================================================
# 1. ALTERNATIVE COHORT BIN WIDTHS
# ===========================================================================

log(f"\n{'='*70}")
log("  1. COHORT BIN WIDTH SENSITIVITY")
log(f"{'='*70}")

for width, label in [(3, "3-year"), (5, "5-year (baseline)"), (10, "10-year")]:
    df_m[f"cohort{width}"] = (df_m["rabyear"] // width) * width
    fe = f"C(cohort{width})"
    formula = f"finlit_pct ~ tr20_level + tr20_change + age + {fe} + {ctrl_str}"
    run_model(f"Cohort_{label}", formula, df_m)

# ===========================================================================
# 2. ITEM-LEVEL ANALYSIS
# ===========================================================================

log(f"\n{'='*70}")
log("  2. ITEM-LEVEL ANALYSIS")
log(f"{'='*70}")
log("  Which financial literacy items are predicted by cog level vs slope?")

df_m["cohort5"] = (df_m["rabyear"] // 5) * 5
cohort_fe = "C(cohort5)"

for item_name in FINLIT_ITEM_NAMES:
    # Convert 0/1 item to percentage for consistent scaling
    df_m[f"{item_name}_pct"] = df_m[item_name] * 100
    formula = f"{item_name}_pct ~ tr20_level + tr20_change + age + {cohort_fe} + {ctrl_str}"
    run_model(f"Item_{item_name}", formula, df_m)

# ===========================================================================
# 3. SUBSAMPLE ANALYSES
# ===========================================================================

log(f"\n{'='*70}")
log("  3. SUBSAMPLE ANALYSES")
log(f"{'='*70}")

formula_full = f"finlit_pct ~ tr20_level + tr20_change + age + {cohort_fe} + {ctrl_str}"

subsamples = {
    "Males_only": df_m[df_m["male"] == 1],
    "Females_only": df_m[df_m["male"] == 0],
    "College_plus": df_m[df_m["educ_cat"] == "college_plus"],
    "No_college": df_m[df_m["educ_cat"] != "college_plus"],
    "Stock_owners": df_m[df_m["stock_owner"] == 1],
    "Non_stock_owners": df_m[df_m["stock_owner"] == 0],
}

for sub_label, sub_df in subsamples.items():
    if len(sub_df) >= 50:
        run_model(f"Sub_{sub_label}", formula_full, sub_df)
    else:
        log(f"\n  Sub_{sub_label}: SKIPPED (n={len(sub_df)} < 50)")

# ===========================================================================
# 4. MINIMUM WAVES SENSITIVITY
# ===========================================================================

log(f"\n{'='*70}")
log("  4. MINIMUM WAVES SENSITIVITY")
log(f"{'='*70}")
log("  Re-estimates the mixed-effects trajectory model with different")
log("  minimum-wave thresholds, then re-runs the core regression.")

df_long = pd.read_parquet(DATA_DIR / "cognition_long_waves3to9.parquet")

for min_w in [3, 4, 5, 6]:
    log(f"\n  Re-estimating tr20 trajectories with min_waves={min_w}...")
    slopes_mw = _estimate_tr20_slopes(df_long, min_w)
    sub = df_m.drop(columns=["tr20_level", "tr20_change"]).merge(
        slopes_mw, on="hhidpn", how="inner",
    )
    sub = sub.dropna(subset=["tr20_level", "tr20_change"])
    if len(sub) >= 50:
        run_model(f"MinWaves_{min_w}", formula_full, sub)
    else:
        log(f"\n  MinWaves_{min_w}: SKIPPED (n={len(sub)} < 50)")

# ===========================================================================
# 5. INTERACTION TESTS FOR HETEROGENEITY
# ===========================================================================

log(f"\n{'='*70}")
log("  5. INTERACTION TESTS FOR HETEROGENEITY")
log(f"{'='*70}")
log("  Pooled models with interaction terms testing whether cognitive-level")
log("  effects differ by gender, education, or stock ownership.")

# Gender × cognitive level
df_m["tr20_x_male"] = df_m["tr20_level"] * df_m["male"]
formula_int = f"finlit_pct ~ tr20_level + tr20_x_male + tr20_change + age + {cohort_fe} + {ctrl_str}"
run_model("Interaction_tr20_x_male", formula_int, df_m,
          key_vars=["tr20_level", "tr20_x_male"])

# College education × cognitive level
df_m["college_plus_bin"] = (df_m["educ_cat"] == "college_plus").astype(int)
df_m["tr20_x_college"] = df_m["tr20_level"] * df_m["college_plus_bin"]
formula_int = f"finlit_pct ~ tr20_level + tr20_x_college + tr20_change + age + {cohort_fe} + {ctrl_str}"
run_model("Interaction_tr20_x_college", formula_int, df_m,
          key_vars=["tr20_level", "tr20_x_college"])

# Stock ownership × cognitive level
df_m["tr20_x_stock"] = df_m["tr20_level"] * df_m["stock_owner"]
formula_int = f"finlit_pct ~ tr20_level + tr20_x_stock + tr20_change + age + {cohort_fe} + {ctrl_str}"
run_model("Interaction_tr20_x_stock", formula_int, df_m,
          key_vars=["tr20_level", "tr20_x_stock"])

# ===========================================================================
# 6. 2010→2016 WITHIN-PERSON BIG 3 CHANGE
# ===========================================================================

log(f"\n{'='*70}")
log("  6. WITHIN-PERSON FINANCIAL LITERACY CHANGE (2010→2016)")
log(f"{'='*70}")

if FAT2016.exists():
    log("  Loading 2016 fat file...")
    vars_2016 = ["hhid", "pn"]
    for item_info in BIG3_2016_SCORING.values():
        vars_2016.extend([item_info["split_a"], item_info["split_b"]])
    df_2016 = pd.read_stata(FAT2016, columns=vars_2016)

    df_2016["hhidpn"] = pd.to_numeric(
        df_2016["hhid"].astype(str).str.strip() +
        df_2016["pn"].astype(str).str.strip().str.zfill(3),
        errors="coerce",
    )

    def score_big3_2016(row):
        """Score Big 3 from whichever split the respondent received."""
        scores = []
        for item_info in BIG3_2016_SCORING.values():
            val = row.get(item_info["split_a"])
            if pd.isna(val):
                val = row.get(item_info["split_b"])
            if pd.notna(val):
                scores.append(1 if float(val) == item_info["correct_val"] else 0)
        if len(scores) == 3:
            return sum(scores)
        return np.nan

    df_2016["finlit3_2016"] = df_2016.apply(score_big3_2016, axis=1)
    n_scored = df_2016["finlit3_2016"].notna().sum()
    log(f"  2016 respondents with Big 3 scored: {n_scored:,}")

    df_panel = df_m.merge(
        df_2016[["hhidpn", "finlit3_2016"]].dropna(),
        on="hhidpn", how="inner",
    )
    log(f"  Matched to 2010 analysis sample: {len(df_panel):,}")

    if len(df_panel) >= 30:
        df_panel["finlit3_2010"] = df_panel["finlit3_correct"]
        df_panel["delta_finlit3"] = df_panel["finlit3_2016"] - df_panel["finlit3_2010"]

        log(f"\n  Within-person Big 3 change (2010→2016):")
        log(f"    Mean 2010: {df_panel['finlit3_2010'].mean():.2f}")
        log(f"    Mean 2016: {df_panel['finlit3_2016'].mean():.2f}")
        log(f"    Mean change: {df_panel['delta_finlit3'].mean():.2f}")
        log(f"    SD change: {df_panel['delta_finlit3'].std():.2f}")

        formula_delta = f"delta_finlit3 ~ tr20_level + tr20_change + {ctrl_str}"
        run_model("Within_person_Big3", formula_delta, df_panel,
                  key_vars=["tr20_level", "tr20_change"])
    else:
        log(f"  Too few matched respondents ({len(df_panel)}) for within-person analysis")
else:
    log("  2016 fat file not found — skipping within-person analysis")

# ===========================================================================
# SAVE
# ===========================================================================

results_df = pd.DataFrame(results_rows)
csv_path = OUTPUT_DIR / "06_robustness.csv"
results_df.to_csv(csv_path, index=False)
log(f"\nSaved: {csv_path}")

logpath = OUTPUT_DIR / "06_robustness.txt"
with open(logpath, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {logpath}")

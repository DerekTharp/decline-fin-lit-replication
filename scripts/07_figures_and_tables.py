"""
07: Create Publication-Ready Figures and Tables

Inputs:  data/analytic_2010_age60plus.parquet
         data/cognitive_slopes.parquet
         data/cognition_long_waves3to9.parquet
         output/02_table9_replication.csv
         output/05_slopes_predict_finlit.csv
         output/06_robustness.csv
Outputs: output/figure1_age_finlit_gradient.png
         output/figure2_trajectories_by_finlit.png
         output/figure3_decomposition.png
         output/figure_comment_coefficients.png
         output/table1_replication.csv
         output/table2_main_models.csv
         output/table3_robustness_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from config import (
    ANALYTIC_2010, DATA_DIR, OUTPUT_DIR, FINLIT_SCORING, FINLIT_ITEM_NAMES,
    PRE2010_YEARS, EDUC_MAP, EDUC_CATEGORIES, EDUC_REFERENCE,
    HIGH_INCOME_WEALTH_QUANTILE, ensure_dirs,
)

ensure_dirs()

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

# ===========================================================================
# LOAD DATA
# ===========================================================================

df = pd.read_parquet(ANALYTIC_2010)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df_slopes = pd.read_parquet(DATA_DIR / "cognitive_slopes.parquet")
df = df.merge(df_slopes, on="hhidpn", how="inner")
df_long = pd.read_parquet(DATA_DIR / "cognition_long_waves3to9.parquet")

# Variables used in the comment models
df["finlit_pct"] = df["finlit7_pct"]
df["rabyear"] = pd.to_numeric(df["rabyear"], errors="coerce")
df["cohort5"] = (df["rabyear"] // 5) * 5
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
df["tr20_decline"] = -df["tr20_change"]

comment_cols = [
    "finlit_pct", "male", "white", "married", "homeowner", "stock_owner",
    "tax_sheltered", "high_income", "high_wealth", "educ_cat", "cohort5",
    "tr20_level", "tr20_decline",
]
df_comment = df.dropna(subset=comment_cols).copy()

# ===========================================================================
# FIGURE 1: Cross-Sectional Age-Financial Literacy Gradient
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

# Panel A: Mean finlit by age year
age_means = df.groupby("age")["finlit7_pct"].agg(["mean", "sem", "count"])
age_means = age_means[age_means["count"] >= 5]

ax = axes[0]
ax.plot(age_means.index, age_means["mean"], "o-", color="#2c3e50", markersize=3, linewidth=1)
ax.fill_between(age_means.index,
                age_means["mean"] - 1.96 * age_means["sem"],
                age_means["mean"] + 1.96 * age_means["sem"],
                alpha=0.2, color="#2c3e50")
ax.set_xlabel("Age")
ax.set_ylabel("Financial literacy score (% correct)")
ax.set_title("A. Mean financial literacy by age")
ax.set_xlim(59, 96)
ax.set_ylim(20, 85)

# Panel B: By item
ax = axes[1]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#34495e"]
for i, item in enumerate(FINLIT_ITEM_NAMES):
    item_by_age = df.groupby("age")[item].mean() * 100
    item_by_age = item_by_age[item_by_age.index.map(lambda a: age_means.index.isin([a]).any())]
    short_name = item.replace("_", " ").title()
    if len(short_name) > 15:
        short_name = short_name[:14] + "."
    ax.plot(item_by_age.index, item_by_age.values, "-", color=colors[i],
            linewidth=1, alpha=0.8, label=short_name)

ax.set_xlabel("Age")
ax.set_title("B. Item-level scores by age")
ax.legend(fontsize=7, loc="lower left", frameon=False)
ax.set_xlim(59, 96)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure1_age_finlit_gradient.png", bbox_inches="tight")
plt.close()
print("Saved figure1_age_finlit_gradient.png")

# ===========================================================================
# FIGURE 2: Cognitive Trajectories by Financial Literacy Quartile
# ===========================================================================

# Assign finlit quartiles
df["finlit_q"] = pd.qcut(df["finlit7_pct"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

# Get the hhidpn-to-quartile mapping
q_map = df[["hhidpn", "finlit_q"]].drop_duplicates("hhidpn")

# Merge quartile onto longitudinal data
df_long_q = df_long.merge(q_map, on="hhidpn", how="inner")

fig, ax = plt.subplots(figsize=(6, 4.5))
colors_q = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

for i, (q, grp) in enumerate(df_long_q.groupby("finlit_q", observed=True)):
    # Mean trajectory by age
    age_bins = pd.cut(grp["age"], bins=range(50, 101, 2))
    means = grp.groupby(age_bins, observed=True)["tr20"].mean()
    # Convert interval midpoints
    midpoints = [interval.mid for interval in means.index]
    ax.plot(midpoints, means.values, "-", color=colors_q[i], linewidth=2, label=str(q))

ax.set_xlabel("Age")
ax.set_ylabel("Total word recall (0-20)")
ax.set_title("Cognitive trajectories by 2010 financial literacy quartile")
ax.legend(title="Finlit quartile", frameon=False)
ax.set_xlim(50, 100)
ax.set_ylim(3, 16)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure2_trajectories_by_finlit.png", bbox_inches="tight")
plt.close()
print("Saved figure2_trajectories_by_finlit.png")

# ===========================================================================
# FIGURE 3: Decomposition Bar Chart
# ===========================================================================

fig, ax = plt.subplots(figsize=(6, 4))

labels = ["Age +\ncontrols", "+ Cohort FE", "+ Cohort FE\n+ Cog level\n+ Cog slope"]
values = [-0.395, -0.244, -0.329]  # From script 05 decomposition
colors_bar = ["#2c3e50", "#e67e22", "#27ae60"]

bars = ax.bar(range(3), [abs(v) for v in values], color=colors_bar, width=0.6, edgecolor="white")

# Add coefficient labels
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)

# Significance markers
ax.text(0, abs(values[0]) + 0.035, "***", ha="center", fontsize=10)
ax.text(1, abs(values[1]) + 0.035, "n.s.", ha="center", fontsize=9, fontstyle="italic")
ax.text(2, abs(values[2]) + 0.035, "n.s.", ha="center", fontsize=9, fontstyle="italic")

ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Age coefficient magnitude\n(pp decline per year)")
ax.set_title("Decomposition of the age-financial literacy gradient")
ax.set_ylim(0, 0.55)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure3_decomposition.png", bbox_inches="tight")
plt.close()
print("Saved figure3_decomposition.png")

# ===========================================================================
# FIGURE 4: Comment coefficient plot (standardized pp per SD)
# ===========================================================================

ref_educ = f"C(educ_cat, Treatment(reference='{EDUC_REFERENCE}'))"
ctrl_str = (
    "male + white + married + homeowner + stock_owner + tax_sheltered + "
    f"high_income + high_wealth + {ref_educ}"
)
cohort_fe = "C(cohort5)"

m_decline = smf.ols(
    f"finlit_pct ~ tr20_decline + {cohort_fe} + {ctrl_str}",
    data=df_comment,
).fit()
m_joint = smf.ols(
    f"finlit_pct ~ tr20_level + tr20_decline + {cohort_fe} + {ctrl_str}",
    data=df_comment,
).fit()

rows = [
    {
        "label": "Cognitive decline rate\n(without level control)",
        "coef": m_decline.params["tr20_decline"] * df_comment["tr20_decline"].std(),
        "ci_low": (m_decline.params["tr20_decline"] - 1.96 * m_decline.bse["tr20_decline"])
        * df_comment["tr20_decline"].std(),
        "ci_high": (m_decline.params["tr20_decline"] + 1.96 * m_decline.bse["tr20_decline"])
        * df_comment["tr20_decline"].std(),
        "color": "#2c3e50",
    },
    {
        "label": "Cognitive decline rate\n(with level control)",
        "coef": m_joint.params["tr20_decline"] * df_comment["tr20_decline"].std(),
        "ci_low": (m_joint.params["tr20_decline"] - 1.96 * m_joint.bse["tr20_decline"])
        * df_comment["tr20_decline"].std(),
        "ci_high": (m_joint.params["tr20_decline"] + 1.96 * m_joint.bse["tr20_decline"])
        * df_comment["tr20_decline"].std(),
        "color": "#2c3e50",
    },
    {
        "label": "Cognitive level\n(with decline rate control)",
        "coef": m_joint.params["tr20_level"] * df_comment["tr20_level"].std(),
        "ci_low": (m_joint.params["tr20_level"] - 1.96 * m_joint.bse["tr20_level"])
        * df_comment["tr20_level"].std(),
        "ci_high": (m_joint.params["tr20_level"] + 1.96 * m_joint.bse["tr20_level"])
        * df_comment["tr20_level"].std(),
        "color": "#c0392b",
    },
]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ypos = np.array([2, 1, 0])
for y, row in zip(ypos, rows):
    ax.errorbar(
        row["coef"],
        y,
        xerr=[[row["coef"] - row["ci_low"]], [row["ci_high"] - row["coef"]]],
        fmt="o",
        color=row["color"],
        ecolor=row["color"],
        elinewidth=2.0,
        capsize=6,
        capthick=1.6,
        markersize=9,
    )

ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
ax.axhline(0.5, color="lightgray", linewidth=1)
ax.set_yticks(ypos)
ax.set_yticklabels([row["label"] for row in rows])
ax.set_xlabel("Effect on financial literacy (pp per SD)")
ax.set_xlim(-1.4, 4.7)
ax.set_ylim(-0.6, 2.6)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_comment_coefficients.png", bbox_inches="tight")
plt.close()
print("Saved figure_comment_coefficients.png")

# ===========================================================================
# FORMATTED TABLES
# ===========================================================================

# Table 1: Replication summary (from script 02 output)
t9 = pd.read_csv(OUTPUT_DIR / "02_table9_replication.csv")
t9_pivot = t9.pivot_table(index="variable", columns="model", values=["coef", "se", "pvalue"], aggfunc="first")
t9_pivot.to_csv(OUTPUT_DIR / "table1_replication.csv")
print("Saved table1_replication.csv")

# Table 2: Main models from script 05
t5 = pd.read_csv(OUTPUT_DIR / "05_slopes_predict_finlit.csv")
key_vars = ["age", "tr20_level", "tr20_change", "ser7_level", "ser7_change",
            "word_recall", "vocabulary"]
t5_key = t5[t5["variable"].isin(key_vars)]
t5_pivot = t5_key.pivot_table(index="variable", columns="model", values=["coef", "se", "pvalue"], aggfunc="first")
t5_pivot.to_csv(OUTPUT_DIR / "table2_main_models.csv")
print("Saved table2_main_models.csv")

# Table 3: Robustness summary
t6 = pd.read_csv(OUTPUT_DIR / "06_robustness.csv")
t6.to_csv(OUTPUT_DIR / "table3_robustness_summary.csv", index=False)
print("Saved table3_robustness_summary.csv")

print("\nAll figures and tables generated.")

# Cognitive Decline and Financial Literacy: HRS Replication and Reanalysis

## Overview

This repository reproduces the Health and Retirement Study (HRS) analysis associated with:

> Finke, M.S., Howe, J.S., & Huston, S.J. (2017). Old age and the decline in financial literacy. *Management Science*, 63(1), 213-230. doi:10.1287/mnsc.2015.2293

The project has two linked goals:

1. Replicate the HRS results reported in Table 9 of Finke et al. (2017).
2. Reanalyze the HRS evidence using repeated pre-2010 cognition to test whether heterogeneity in cognitive decline rates predicts 2010 financial literacy beyond cognitive level.

The repository contains the full runnable pipeline, logs, and generated outputs used to support the current comment manuscript and the broader replication/extension project.

## Data Availability

This project uses HRS data that are publicly available via registration from the University of Michigan. The data may be downloaded by registered users, but they may not be redistributed here.

### Data files used

Place the following files inside the `HRS/` directory:

| File | Description | Status | Version used |
|------|-------------|--------|--------------|
| `HRS Fat Files/hd10f6b_STATA/hd10f6b.dta` | 2010 HRS Fat File | Required | `hd10f6b` |
| `randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta` | RAND HRS Longitudinal File | Required | `1992-2022 v1` |
| `HRS Fat Files/h16f2c_STATA/h16f2c.dta` | 2016 HRS Fat File | Optional for the core comment; required to reproduce the 2010 to 2016 within-person robustness section in `output/06_robustness.*` | `h16f2c` |

If the 2016 fat file is not available, `scripts/06_robustness.py` will skip the within-person 2010 to 2016 analysis rather than fail.

## Software Requirements

- Python `>= 3.10`
- Packages listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

Developed and tested with Python `3.14.3` on macOS.

## Reproducing Results

Run the full pipeline from the project root:

```bash
python3 run_all.py
```

Expected runtime: approximately 5 to 10 minutes, dominated by loading the RAND longitudinal file and fitting mixed-effects models.

The full pipeline executes:

1. `00_inspect_finlit_2010.py`
2. `01_build_analytic_sample.py`
3. `02_replicate_table9.py`
4. `03_extract_longitudinal_cognition.py`
5. `04_estimate_cognitive_trajectories.py`
6. `05_cognitive_slopes_predict_finlit.py`
7. `06_robustness.py`
8. `07_figures_and_tables.py`

## Comment-Relevant Outputs

The current comment manuscript relies primarily on:

| Item | Source |
|------|--------|
| Cross-sectional replication counts and coefficients | `output/02_table9_replication.txt`, `output/02_table9_replication.csv` |
| Longitudinal level-versus-decline models | `output/05_slopes_predict_finlit.txt`, `output/05_slopes_predict_finlit.csv` |
| Robustness check for Serial 7s | `output/06_robustness.txt`, `output/06_robustness.csv` |
| Comment figure | `output/figure_comment_coefficients.png` |

## Script-to-Output Mapping

| Script | Description | Primary outputs |
|--------|-------------|-----------------|
| `scripts/00_inspect_finlit_2010.py` | Inspect raw 2010 financial literacy coding | `output/00_finlit_2010_inspection.txt` |
| `scripts/01_build_analytic_sample.py` | Score 2010 items, merge with RAND, and construct the age-60+ analytic sample | `data/analytic_2010_age60plus.parquet`, `output/01_sample_construction.txt` |
| `scripts/02_replicate_table9.py` | Replicate HRS Table 9 specifications | `output/02_table9_replication.csv`, `output/02_table9_replication.txt` |
| `scripts/03_extract_longitudinal_cognition.py` | Reshape pre-2010 RAND cognition to person-wave long format | `data/cognition_long_waves3to9.parquet`, `output/03_longitudinal_cognition.txt` |
| `scripts/04_estimate_cognitive_trajectories.py` | Estimate person-specific cognition intercepts and slopes | `data/cognitive_slopes.parquet`, `output/04_cognitive_trajectories.txt` |
| `scripts/05_cognitive_slopes_predict_finlit.py` | Estimate core level-versus-decline financial literacy models | `output/05_slopes_predict_finlit.csv`, `output/05_slopes_predict_finlit.txt` |
| `scripts/06_robustness.py` | Run robustness checks and 2010 to 2016 within-person analysis | `output/06_robustness.csv`, `output/06_robustness.txt` |
| `scripts/07_figures_and_tables.py` | Generate publication-ready figures and summary tables | `output/figure1_age_finlit_gradient.png`, `output/figure2_trajectories_by_finlit.png`, `output/figure3_decomposition.png`, `output/figure_comment_coefficients.png`, `output/table1_replication.csv`, `output/table2_main_models.csv`, `output/table3_robustness_summary.csv` |

## Data Dictionary

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for:

- dataset-level descriptions,
- keys and row units,
- variable-level descriptions for derived datasets,
- and the list of raw HRS variables used from each source file.

## Directory Structure

```text
project/
  README.md
  DATA_DICTIONARY.md
  requirements.txt
  run_all.py
  scripts/
  output/
  data/                  Generated locally; not redistributed
  HRS/                   Raw HRS files obtained directly from HRS; not tracked
```

## Known Deviations from Finke et al. (2017)

1. Sample sizes are modestly above the published HRS counts (`1,126` versus `1,109`; `907` versus `887`). This is likely due to minor differences in harmonization releases and undocumented sample restrictions, but the replicated coefficients are substantively similar.
2. RAND `raeduc` does not separate college from graduate degree, so the highest education category is combined as `college_plus`.
3. Tax-sheltered status is proxied using positive IRA/Keogh assets (`h10aira > 0`).
4. The core comment reanalysis uses pre-2010 trajectory estimates from repeated cognition rather than contemporaneous 2010 cognition controls.

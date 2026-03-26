# Data Dictionary

This file documents the datasets and variables used in the replication package. Official HRS and RAND codebooks remain the authoritative source for raw-data definitions; the tables below describe the specific files and variables used in this project.

## Raw input files

### `HRS/HRS Fat Files/hd10f6b_STATA/hd10f6b.dta`

- Row unit: respondent in the 2010 HRS fat file
- Key fields used: `hhid`, `pn`
- Purpose: 2010 experimental financial literacy module

| Variable | Description |
|----------|-------------|
| `hhid` | Household identifier |
| `pn` | Person number within household |
| `mv347` | Self-rated financial understanding |
| `mv351` | Compound-interest item |
| `mv352` | Inflation item |
| `mv353` | Stock versus mutual fund item |
| `mv354` | Highest historical returns item |
| `mv365` | Company stock concentration item |
| `mv366` | International diversification item |
| `mv370` | Bond-pricing / interest-rate item |

### `HRS/HRS Fat Files/h16f2c_STATA/h16f2c.dta`

- Row unit: respondent in the 2016 HRS fat file
- Key fields used: `hhid`, `pn`
- Purpose: optional 2010 to 2016 within-person Big 3 robustness analysis

| Variable | Description |
|----------|-------------|
| `hhid` | Household identifier |
| `pn` | Person number within household |
| `pv052`, `pv102` | Compound-interest item, split A / split B |
| `pv053`, `pv103` | Inflation item, split A / split B |
| `pv054`, `pv104` | Diversification item, split A / split B |

### `HRS/randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta`

- Row unit: respondent, wide format across waves
- Key field used: `hhidpn`
- Purpose: harmonized demographics, finances, and cognition

| Variable | Description |
|----------|-------------|
| `hhidpn` | Combined respondent identifier |
| `rabyear` | Birth year |
| `ragender` | Gender |
| `raracem` | Race |
| `rahispan` | Hispanic ethnicity indicator |
| `raedyrs` | Years of education |
| `raeduc` | Education category |
| `r10agey_e` | Age at 2010 interview |
| `r10mstat` | Marital status in 2010 |
| `r10shlt` | Self-rated health in 2010 |
| `r10cesd` | CES-D depression score in 2010 |
| `h10atotw` | Total household wealth in 2010 |
| `h10itot` | Total household income in 2010 |
| `h10ahous` | Value of primary residence in 2010 |
| `h10astck` | Value of stocks / mutual funds in 2010 |
| `h10aira` | IRA / Keogh assets in 2010 |
| `r10imrc` | Immediate word recall in 2010 |
| `r10dlrc` | Delayed word recall in 2010 |
| `r10tr20` | Total word recall in 2010 |
| `r10ser7` | Serial 7s in 2010 |
| `r10bwc20` | Backward counting in 2010 |
| `r10cogtot` | Total cognition in 2010 |
| `r10vocab` | Vocabulary in 2010 |
| `r10nsscre` | Number series score in 2010 |
| `r10iwstat` | Interview status in 2010 |
| `r10proxy` | Proxy interview indicator in 2010 |
| `r3agey_e`-`r9agey_e` | Age across pre-2010 waves |
| `r3tr20`-`r9tr20` | Total word recall across pre-2010 waves |
| `r3imrc`-`r9imrc` | Immediate word recall across pre-2010 waves |
| `r3dlrc`-`r9dlrc` | Delayed word recall across pre-2010 waves |
| `r3ser7`-`r9ser7` | Serial 7s across pre-2010 waves |
| `r3bwc20`-`r9bwc20` | Backward counting across pre-2010 waves |
| `r3vocab`-`r9vocab` | Vocabulary across pre-2010 waves |
| `r3proxy`-`r9proxy` | Proxy interview indicator across pre-2010 waves |
| `r3iwstat`-`r9iwstat` | Interview status across pre-2010 waves |

## Derived datasets

### `data/analytic_2010_age60plus.parquet`

- Row unit: respondent age 60 or older in 2010
- Key field: `hhidpn`
- Constructed by: `scripts/01_build_analytic_sample.py`

| Variable | Description |
|----------|-------------|
| `hhidpn` | Combined respondent identifier |
| `age` | Age in 2010 |
| `compound_interest` | 2010 compound-interest item scored 0/1 |
| `inflation` | 2010 inflation item scored 0/1 |
| `stock_vs_mutualfund` | 2010 diversification item scored 0/1 |
| `highest_returns` | 2010 highest-returns item scored 0/1 |
| `company_stock` | 2010 employer-stock item scored 0/1 |
| `foreign_stocks` | 2010 international-diversification item scored 0/1 |
| `bond_interest` | 2010 bond-pricing item scored 0/1 |
| `finlit7_correct` | Count of correct answers across the seven-item instrument |
| `finlit7_pct` | Percent correct across the seven-item instrument |
| `finlit3_correct` | Count correct on the Big 3 subset |
| `finlit3_pct` | Percent correct on the Big 3 subset |
| `confidence` | Self-rated financial understanding |
| `rabyear` | Birth year |
| `ragender` | Gender |
| `raracem` | Race |
| `rahispan` | Hispanic ethnicity indicator |
| `raedyrs` | Years of education |
| `raeduc` | Education category |
| `r10agey_e` | Age at 2010 interview |
| `r10mstat` | Marital status in 2010 |
| `r10shlt` | Self-rated health in 2010 |
| `r10cesd` | CES-D depression score in 2010 |
| `h10atotw` | Total household wealth in 2010 |
| `h10itot` | Total household income in 2010 |
| `h10ahous` | Value of primary residence in 2010 |
| `h10astck` | Value of stocks / mutual funds in 2010 |
| `h10aira` | IRA / Keogh assets in 2010 |
| `r10imrc` | Immediate word recall in 2010 |
| `r10dlrc` | Delayed word recall in 2010 |
| `r10tr20` | Total word recall in 2010 |
| `r10ser7` | Serial 7s in 2010 |
| `r10bwc20` | Backward counting in 2010 |
| `r10cogtot` | Total cognition in 2010 |
| `r10vocab` | Vocabulary in 2010 |
| `r10nsscre` | Number series score in 2010 |
| `r10iwstat` | Interview status in 2010 |
| `r10proxy` | Proxy interview indicator in 2010 |

### `data/cognition_long_waves3to9.parquet`

- Row unit: respondent-wave observation for waves 3 to 9 (1996 to 2008)
- Key fields: `hhidpn`, `wave`
- Constructed by: `scripts/03_extract_longitudinal_cognition.py`

| Variable | Description |
|----------|-------------|
| `hhidpn` | Combined respondent identifier |
| `wave` | HRS wave number |
| `year` | Calendar year corresponding to the wave |
| `age` | Age at interview |
| `rabyear` | Birth year |
| `ragender` | Gender |
| `raracem` | Race |
| `rahispan` | Hispanic ethnicity indicator |
| `raedyrs` | Years of education |
| `raeduc` | Education category |
| `tr20` | Total word recall |
| `imrc` | Immediate word recall |
| `dlrc` | Delayed word recall |
| `ser7` | Serial 7s |
| `bwc20` | Backward counting |
| `vocab` | Vocabulary |
| `proxy` | Proxy interview indicator |
| `n_waves_tr20` | Number of non-missing word-recall waves for respondent |
| `n_waves_any` | Number of waves with any cognition observed for respondent |

### `data/cognitive_slopes.parquet`

- Row unit: respondent
- Key field: `hhidpn`
- Constructed by: `scripts/04_estimate_cognitive_trajectories.py`

| Variable | Description |
|----------|-------------|
| `hhidpn` | Combined respondent identifier |
| `tr20_intercept` | EB random-intercept term for total word recall |
| `tr20_slope` | EB random-slope term for total word recall |
| `tr20_level` | Predicted total word recall at age 70 |
| `tr20_change` | Predicted annual change in total word recall |
| `ser7_intercept` | EB random-intercept term for Serial 7s |
| `ser7_slope` | EB random-slope term for Serial 7s |
| `ser7_level` | Predicted Serial 7s score at age 70 |
| `ser7_change` | Predicted annual change in Serial 7s |

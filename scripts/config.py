"""
Shared configuration for the Finke et al. (2016) replication project.

All paths, variable definitions, scoring rules, and sample expectations
are defined here. Every analysis script imports from this module.
"""

import sys
from pathlib import Path

# ===========================================================================
# PATHS — derived from this file's location, never hardcoded
# ===========================================================================

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPTS_DIR.parent
DATA_DIR = PROJ_DIR / "data"
OUTPUT_DIR = PROJ_DIR / "output"
HRS_DIR = PROJ_DIR / "HRS"

# Raw data files
FAT2010 = HRS_DIR / "HRS Fat Files" / "hd10f6b_STATA" / "hd10f6b.dta"
FAT2016 = HRS_DIR / "HRS Fat Files" / "h16f2c_STATA" / "h16f2c.dta"
FAT2020 = HRS_DIR / "HRS Fat Files" / "h20f1b_STATA" / "h20f1b.dta"
RAND_LONGITUDINAL = HRS_DIR / "randhrs1992_2022v1_STATA" / "randhrs1992_2022v1.dta"

# Intermediate data
ANALYTIC_2010 = DATA_DIR / "analytic_2010_age60plus.parquet"


def ensure_dirs():
    """Create output directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def check_raw_data():
    """Verify raw data files exist. Exit with helpful message if not."""
    missing = []
    for label, path in [
        ("2010 HRS Fat File", FAT2010),
        ("RAND Longitudinal File", RAND_LONGITUDINAL),
    ]:
        if not path.exists():
            missing.append(f"  {label}: {path}")
    if missing:
        sys.exit(
            "Raw data files not found:\n"
            + "\n".join(missing)
            + "\n\nSee README.md for HRS data download instructions."
        )


# ===========================================================================
# FINANCIAL LITERACY INSTRUMENT — 2010 HRS Module (Section V)
# ===========================================================================

# Scoring rules for the 7 Finke et al. items.
#
# Source: 2010 HRS Codebook + Finke et al. (2016) Appendix B
#
# Coding conventions for HRS Section V:
#   True/False items: 1=True, 5=False, 8=DK, 9=RF
#   Multiple-choice:  numbered responses, 8=DK, 9=RF
#
# Decision: DK (8) and RF (9) are scored as INCORRECT (=0), not missing.
# Rationale: This follows Lusardi & Mitchell (2014) convention and is
# required to match Finke et al.'s reported sample sizes.

FINLIT_SCORING = {
    "mv351": {
        "name": "compound_interest",
        "question": "$100 at 2% interest for 5 years: more/exactly/less than $102?",
        "correct_val": 1,   # 1=More than $102
        "codes": {1: "More than $102", 2: "Exactly $102", 3: "Less than $102"},
    },
    "mv352": {
        "name": "inflation",
        "question": "Interest rate 1%, inflation 2%: buy more/same/less after 1 year?",
        "correct_val": 3,   # 3=Less
        "codes": {1: "More", 2: "Same", 3: "Less"},
    },
    "mv353": {
        "name": "stock_vs_mutualfund",
        "question": "Single company stock usually provides safer return than mutual fund?",
        "correct_val": 5,   # 5=False
        "codes": {1: "True", 5: "False"},
    },
    "mv354": {
        "name": "highest_returns",
        "question": "Which asset has historically paid highest returns?",
        "correct_val": 3,   # 3=Stocks
        "codes": {1: "Savings accounts", 2: "Bonds", 3: "Stocks"},
    },
    "mv365": {
        "name": "company_stock",
        "question": "Employee should have a lot of retirement savings in company stock?",
        "correct_val": 5,   # 5=False (concentration risk)
        "codes": {1: "True", 5: "False"},
    },
    "mv366": {
        "name": "foreign_stocks",
        "question": "It is best to avoid owning stocks of foreign companies?",
        "correct_val": 5,   # 5=False (should diversify internationally)
        "codes": {1: "True", 5: "False"},
    },
    "mv370": {
        "name": "bond_interest",
        "question": "If the interest rate falls, bond prices will rise?",
        "correct_val": 1,   # 1=True (confirmed against HRS codebook)
        "codes": {1: "True", 5: "False"},
    },
}

# Convenience lists
FINLIT_HRS_VARS = list(FINLIT_SCORING.keys())
FINLIT_ITEM_NAMES = [v["name"] for v in FINLIT_SCORING.values()]
BIG3_ITEMS = ["compound_interest", "inflation", "stock_vs_mutualfund"]

# 2016 Big 3 scoring (standard HRS module, pv052-054 / pv102-104)
# KNOWN DEVIATION from 2010 Section V coding:
#   2010 Section V True/False items use 1=True, 5=False
#   2016 standard module True/False items use 1=True, 2=False
# Source: 2016 HRS codebook, Section V (Financial Literacy)
BIG3_2016_SCORING = {
    "compound_interest": {"split_a": "pv052", "split_b": "pv102", "correct_val": 1},
    "inflation": {"split_a": "pv053", "split_b": "pv103", "correct_val": 3},
    "diversification": {"split_a": "pv054", "split_b": "pv104", "correct_val": 2},
}

DK_CODE = 8
RF_CODE = 9

# Additional variables to extract from the 2010 fat file
FAT2010_EXTRA_VARS = [
    "hhid", "pn",
    "mv347",   # self-rated financial understanding (1-7 scale, 8=DK, 9=RF)
]

# ===========================================================================
# RAND LONGITUDINAL FILE — Variable extraction for wave 10 (2010)
# ===========================================================================

RAND_VARS = [
    "hhidpn",
    # Demographics (time-invariant)
    "rabyear",     # birth year
    "ragender",    # 1=male, 2=female
    "raracem",     # 1=white, 2=black, 3=other
    "rahispan",    # 0=not hispanic, 1=hispanic
    "raedyrs",     # education years
    "raeduc",      # 1=lt hs, 2=ged, 3=hs, 4=some college, 5=college+
    # Wave 10 (2010) time-varying
    "r10agey_e",   # age at 2010 interview
    "r10mstat",    # marital status (1-3=married/partnered, 4-8=not)
    "r10shlt",     # self-rated health (1=excellent to 5=poor)
    "r10cesd",     # CES-D depression score (0-8)
    "h10atotw",    # total household wealth
    "h10itot",     # total household income
    "h10ahous",    # value of primary residence
    "h10astck",    # net value of stocks/mutual funds
    "h10aira",     # IRA/Keogh assets (proxy for tax-sheltered status)
    # Wave 10 cognition
    "r10imrc",     # immediate word recall (0-10)
    "r10dlrc",     # delayed word recall (0-10)
    "r10tr20",     # total word recall = imrc + dlrc (0-20)
    "r10ser7",     # serial 7s (0-5)
    "r10bwc20",    # backwards counting from 20 (0-2)
    "r10cogtot",   # total cognition score (0-35)
    "r10vocab",    # vocabulary (0-10)
    "r10nsscre",   # number series score (continuous)
    # Interview metadata
    "r10iwstat",   # interview status
    "r10proxy",    # 0=self-respondent, 1=proxy
]

# ===========================================================================
# LONGITUDINAL COGNITION — waves 3-9 (pre-2010) for trajectory estimation
# ===========================================================================

# Waves 3-9 correspond to HRS years 1996-2008.
# We use pre-2010 cognition only so the trajectory causally precedes
# the 2010 financial literacy measurement.
PRE2010_WAVES = [3, 4, 5, 6, 7, 8, 9]
PRE2010_YEARS = {3: 1996, 4: 1998, 5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008}

# Cognition variables available across these waves (RAND naming: r{w}{varname})
COG_MEASURES = {
    "tr20":  {"label": "Total word recall (0-20)", "range": (0, 20)},
    "imrc":  {"label": "Immediate word recall (0-10)", "range": (0, 10)},
    "dlrc":  {"label": "Delayed word recall (0-10)", "range": (0, 10)},
    "ser7":  {"label": "Serial 7s (0-5)", "range": (0, 5)},
    "bwc20": {"label": "Backwards counting (0-2)", "range": (0, 2)},
    "vocab": {"label": "Vocabulary (0-10)", "range": (0, 10)},
}

# Variables to extract per wave for the longitudinal file
LONGITUDINAL_VARS_PER_WAVE = ["agey_e", "tr20", "imrc", "dlrc", "ser7", "bwc20", "vocab", "proxy", "iwstat"]

# Build the full variable list for extraction
LONGITUDINAL_RAND_VARS = ["hhidpn", "rabyear", "ragender", "raracem", "rahispan", "raedyrs", "raeduc"]
for w in PRE2010_WAVES:
    for var in LONGITUDINAL_VARS_PER_WAVE:
        LONGITUDINAL_RAND_VARS.append(f"r{w}{var}")

# Minimum waves of cognitive data required for reliable slope estimation
MIN_WAVES_FOR_SLOPE = 3

# Intermediate data
LONGITUDINAL_LONG = DATA_DIR / "cognition_long_waves3to9.parquet"

# ===========================================================================
# SAMPLE RESTRICTIONS AND EXPECTED SIZES
# ===========================================================================

AGE_LOWER_BOUND = 60     # Finke restricts to age 60+
AGE_UPPER_BOUND = 120    # Reasonable upper bound for age bins

# Expected sample sizes from Finke et al. Table 9 (HRS analysis)
EXPECTED_N_FINLIT = 1109      # age 60+ with financial literacy
EXPECTED_N_COGNITION = 887    # age 60+ with financial literacy + word recall + vocabulary
SAMPLE_SIZE_TOLERANCE = 30    # acceptable deviation from Finke's reported n

# ===========================================================================
# CONTROL VARIABLE SPECIFICATIONS — matching Finke et al. Table 9
# ===========================================================================

# Education categories (Finke Table 9: <High school, Some college, College, Graduate)
# NOTE: RAND raeduc=5 is "college and above" — does not distinguish college
# from graduate. This is a KNOWN DEVIATION from Finke, who may have used
# a finer-grained education variable from the fat file.
EDUC_MAP = {1: "lt_hs", 2: "hs", 3: "hs", 4: "some_college", 5: "college_plus"}
EDUC_CATEGORIES = ["lt_hs", "hs", "some_college", "college_plus"]
EDUC_REFERENCE = "hs"

# Finke uses top-quintile dummies for income and wealth.
# Source: Finke et al. (2016), Section 3.2, p. 6: "We use the top income
# and wealth quintile to capture the incentive to invest in financial
# information among those with the most money to manage."
HIGH_INCOME_WEALTH_QUANTILE = 0.80


# ===========================================================================
# SHARED LOGGING
# ===========================================================================

def make_logger():
    """Return a (log, get_lines) pair for dual stdout/buffer logging."""
    lines = []
    def log(text=""):
        print(text)
        lines.append(text)
    return log, lines

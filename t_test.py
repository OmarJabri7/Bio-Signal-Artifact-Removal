import pandas as pd
from scipy import stats
import random
SAMPLE_SIZE = 30
significance_lvl = 0.05


def get_stats():
    snrs_alpha = pd.read_excel(r"Results-Generation/t-test/snrs.xlsx", "Alpha")
    snrs_delta = pd.read_excel(r"Results-Generation/t-test/snrs.xlsx", "Delta")
    DNF_alpha = snrs_alpha["DNF"]
    ORIG_alpha = snrs_alpha["ORIG"]
    DNF_delta = snrs_delta["DNF"]
    ORIG_delta = snrs_delta["ORIG"]
    snrs_dnf_a = random.choices(DNF_alpha.values, k=SAMPLE_SIZE)
    snrs_orig_a = random.choices(ORIG_alpha.values, k=SAMPLE_SIZE)
    snrs_dnf_d = random.choices(DNF_delta.values, k=SAMPLE_SIZE)
    snrs_orig_d = random.choices(ORIG_delta.values, k=SAMPLE_SIZE)
    _, p_val_a = stats.ttest_ind(snrs_dnf_a, snrs_orig_a, equal_var=False)
    _, p_val_d = stats.ttest_ind(snrs_dnf_d, snrs_orig_d, equal_var=False)
    print(f"P-Value for Alpha waves: {p_val_a}")
    print(f"P-Value for Delta waves: {p_val_d}")
    if(p_val_a <= significance_lvl and p_val_d <= significance_lvl):
        print("Null Hypothesis H0 rejected")
    else:
        print("Failed to reject Null Hypothesis H0")


get_stats()

"""Generate a small SYNTHETIC file with the Lending Club schema -- for smoke-testing the code only.

Nothing produced from this file is ever reported as a result.
Usage: python tests/make_synthetic.py OUT.csv.gz [n_rows]
"""
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
out = sys.argv[1]

fico = rng.integers(660, 820, n) // 5 * 5
dti = rng.gamma(4, 4.5, n)
inc = np.exp(rng.normal(11, 0.5, n))
amt = (rng.uniform(1000, 35000, n) // 25) * 25
util = rng.uniform(0, 100, n)
inq = rng.poisson(0.8, n)
grade_idx = np.clip(((820 - fico) / 25 + rng.normal(0, 1, n)).astype(int), 0, 6)
grade = np.array(list("ABCDEFG"))[grade_idx]
logit = -2.0 - 0.03 * (fico - 700) + 0.03 * (dti - 18) + 0.01 * (util - 50) + 0.2 * inq
p = 1 / (1 + np.exp(-logit))
bad = rng.random(n) < p
year = rng.choice([2011, 2012, 2013, 2014, 2015, 2016], n, p=[.05, .1, .15, .25, .3, .15])
month = rng.choice(["Jan", "Apr", "Jul", "Oct"], n)
status = np.where(bad, "Charged Off", "Fully Paid")
status = np.where((year == 2016) & (rng.random(n) < 0.3), "Current", status)
prncp_paid = np.where(bad, amt * rng.beta(2, 3, n), amt)
rec = np.where(bad & (rng.random(n) < 0.6), (amt - prncp_paid) * rng.beta(1.2, 6, n), 0)
df = pd.DataFrame({
    "loan_amnt": amt, "funded_amnt": amt, "term": rng.choice([" 36 months", " 60 months"], n, p=[.8, .2]),
    "int_rate": 6 + grade_idx * 3 + rng.normal(0, .5, n), "installment": amt / 30,
    "grade": grade, "sub_grade": [g + str(rng.integers(1, 6)) for g in grade],
    "emp_length": rng.choice(["< 1 year", "1 year", "5 years", "10+ years", None], n),
    "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], n, p=[.45, .45, .099, .001]),
    "annual_inc": inc, "verification_status": rng.choice(["Verified", "Not Verified", "Source Verified"], n),
    "issue_d": [f"{m}-{y}" for m, y in zip(month, year)], "loan_status": status,
    "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "car", "medical", "other"], n),
    "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL"], n), "dti": dti,
    "delinq_2yrs": rng.poisson(0.3, n), "earliest_cr_line": [f"Jan-{y}" for y in rng.integers(1980, 2008, n)],
    "fico_range_low": fico, "fico_range_high": fico + 4, "inq_last_6mths": inq,
    "open_acc": rng.poisson(11, n), "pub_rec": rng.poisson(0.2, n), "revol_bal": rng.gamma(2, 8000, n),
    "revol_util": util, "total_acc": rng.poisson(25, n), "total_pymnt": prncp_paid * 1.1,
    "total_rec_prncp": prncp_paid, "recoveries": rec, "collection_recovery_fee": rec * 0.18,
    "last_pymnt_d": [f"Jan-{y + 1}" for y in year], "mort_acc": rng.poisson(1.5, n),
    "pub_rec_bankruptcies": rng.poisson(0.1, n),
})
df.to_csv(out, index=False)
print("wrote", out, len(df))

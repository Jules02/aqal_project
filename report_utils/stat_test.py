import json
import numpy as np
from scipy import stats

# Read data
with open("../experiments_log.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Group by seed for pairing
parametrized   = {r["seed"]: r["final_test_acc"] for r in records if r["experiment_name"] == "parametrized_filtereddataset"}
unparametrized = {r["seed"]: r["final_test_acc"] for r in records if r["experiment_name"] == "unparametrized_filtereddataset"}

# Keep only paired seeds
common_seeds = sorted(set(parametrized) & set(unparametrized))
assert len(common_seeds) > 0, "No paired seeds found"

X = np.array([parametrized[s]   for s in common_seeds])   # parametrized accuracies
Y = np.array([unparametrized[s] for s in common_seeds])   # unparametrized accuracies
d = X - Y                                                  # paired differences (positive = parametrized is better)

# One-sided paired t-test: H0: mu_param - mu_fixed <= 0
#                          HA: mu_param - mu_fixed >  0
t_stat, p_two_sided = stats.ttest_rel(X, Y)
p_one_sided = p_two_sided / 2  # one-sided p-value (valid since t_stat > 0 confirmed below)

print(f"Number of paired trials : {len(common_seeds)}")
print(f"Seeds                   : {common_seeds}")
print()
print(f"Mean parametrized acc   : {X.mean():.4f} ± {X.std():.4f}")
print(f"Mean unparametrized acc : {Y.mean():.4f} ± {Y.std():.4f}")
print(f"Mean paired difference  : {d.mean():.4f} ± {d.std():.4f}")
print()
print(f"t-statistic             : {t_stat:.4f}")
print(f"p-value (two-sided)     : {p_two_sided:.6f}")
print(f"p-value (one-sided)     : {p_one_sided:.6f}")
print()

alpha = 0.05
if t_stat > 0 and p_one_sided < alpha:
    print(f"✓ Reject H0 at significance level α = {alpha}.")
    print(f"  Parametrizing the encoding layers significantly improves test accuracy (p = {p_one_sided:.4f}).")
else:
    print(f"✗ Fail to reject H0 at significance level α = {alpha} (p = {p_one_sided:.4f}).")
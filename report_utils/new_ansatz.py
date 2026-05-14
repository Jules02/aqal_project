import json
import numpy as np

THRESHOLD = 0.75
MAX_EPOCHS = 50

with open("../experiments_log.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Separate by experiment name
filtered   = {r["seed"]: r for r in records if r["experiment_name"] == "filtered_scaling_exp"}
unfiltered = {r["seed"]: r for r in records if r["experiment_name"] == "epochs_new_ansatz_filtered"}

common_seeds = sorted(set(filtered) & set(unfiltered))

# ── Per-run classification ────────────────────────────────────────────────────
def classify_run(r):
    """Returns (reached_threshold, epoch_reached, wall_time_to_threshold)"""
    reached  = r["last_epoch"] is not None
    epoch    = r["last_epoch"] if reached else None
    walltime = epoch * r["avg_time_per_epoch_sec"] if reached else None
    return reached, epoch, walltime

rows_filt = [classify_run(filtered[s])   for s in common_seeds]
rows_unf  = [classify_run(unfiltered[s]) for s in common_seeds]

# ── Metrics ───────────────────────────────────────────────────────────────────
def summarise(rows, records_by_seed, seeds):
    success_mask  = [r[0] for r in rows]
    epochs        = [r[1] for r in rows if r[0]]
    walltimes     = [r[2] for r in rows if r[0]]
    test_accs_all = [records_by_seed[s]["final_test_acc"] for s in seeds]
    test_accs_suc = [records_by_seed[s]["final_test_acc"] for s, ok in zip(seeds, success_mask) if ok]

    n_total   = len(rows)
    n_success = sum(success_mask)

    return {
        "n_total"          : n_total,
        "n_success"        : n_success,
        "success_rate"     : f"{n_success}/{n_total}",
        "mean_epoch"       : f"{np.mean(epochs):.1f} ± {np.std(epochs):.1f}"       if epochs    else "N/A",
        "mean_avg_time_per_epoch_sec" : f"{np.mean([r[2]/r[1] for r in rows if r[0]]):.2f} ± {np.std([r[2]/r[1] for r in rows if r[0]]):.2f}" if epochs and walltimes else "N/A",
        "mean_walltime"    : f"{np.mean(walltimes):.1f} ± {np.std(walltimes):.1f}" if walltimes else "N/A",
        "mean_acc_all"     : f"{np.mean(test_accs_all):.3f} ± {np.std(test_accs_all):.3f}",
        "mean_acc_success" : f"{np.mean(test_accs_suc):.3f} ± {np.std(test_accs_suc):.3f}" if test_accs_suc else "N/A",
    }

s_filt = summarise(rows_filt, filtered,   common_seeds)
s_unf  = summarise(rows_unf,  unfiltered, common_seeds)

# ── Print two-metric table ────────────────────────────────────────────────────
col_w = 38
print(f"\nThreshold: {THRESHOLD*100:.0f}% test accuracy | Max epochs: {MAX_EPOCHS}\n")
print(f"{'Metric':<{col_w}} {'Baseline':>18} {'New ansatz':>18}")
print("-" * (col_w + 38))
print(f"{'Success rate':<{col_w}} {s_filt['success_rate']:>18} {s_unf['success_rate']:>18}")
print(f"{'Mean epoch reached threshold':<{col_w}} {s_filt['mean_epoch']:>18} {s_unf['mean_epoch']:>18}")
print(f"{'Mean time per epoch (s)':<{col_w}} {s_filt['mean_avg_time_per_epoch_sec']:>18} {s_unf['mean_avg_time_per_epoch_sec']:>18}")
print(f"{'Mean wall time to threshold (s)':<{col_w}} {s_filt['mean_walltime']:>18} {s_unf['mean_walltime']:>18}")
print(f"{'Mean final test acc (all runs)':<{col_w}} {s_filt['mean_acc_all']:>18} {s_unf['mean_acc_all']:>18}")
print(f"{'Mean final test acc (successes only)':<{col_w}} {s_filt['mean_acc_success']:>18} {s_unf['mean_acc_success']:>18}")
print("-" * (col_w + 38))
print(f"\nNote: 'success' = reached {THRESHOLD*100:.0f}% test accuracy within {MAX_EPOCHS} epochs.")
print(      "      Failed runs (last_epoch = null) are excluded from epoch/walltime means")
print(      "      but counted in the success rate and 'all runs' accuracy.")
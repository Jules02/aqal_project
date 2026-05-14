import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open("../experiments_log.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

def extract_data(records, experiment_name, num_params=2, dataset_size=500):
    filtered = {}
    for r in records:
        if r["experiment_name"] != experiment_name:
            continue
        if r["dataset_size"] != dataset_size:
            continue
        if r["num_params_per_layer"] != num_params:
            continue
        filtered[r["num_layers"]] = r
    data = sorted(filtered.values(), key=lambda r: r["num_layers"])
    layers = [r["num_layers"] for r in data]
    acc    = [r["final_test_acc"] for r in data]
    time   = [r["avg_time_per_epoch_sec"] for r in data]
    return layers, acc, time

layers_unf, acc_unf, time_unf = extract_data(records, "unfiltered")
layers_main, acc_main, time_main = extract_data(records, "main_")

BLUE      = "#2563EB"
ALPHA_UNF = 0.3

# ── Plot 1: Test Accuracy ─────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(layers_unf,  acc_unf,  marker="o", linewidth=2, markersize=7,
         color=BLUE, alpha=ALPHA_UNF, linestyle="--", label="Unfiltered training set")
ax1.plot(layers_main, acc_main, marker="o", linewidth=2, markersize=7,
         color=BLUE, label="Filtered training set")

for x, y in zip(layers_unf, acc_unf):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, -15), ha="center", fontsize=8.5, color=BLUE, alpha=0.4)
for x, y in zip(layers_main, acc_main):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8.5, color=BLUE)

ax1.set_xlabel("Number of layers $L$", fontsize=12)
ax1.set_ylabel("Final test accuracy", fontsize=12)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax1.set_ylim(0.5, 1.02)
ax1.legend(fontsize=11)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)

fig1.tight_layout()
fig1.savefig("test_accuracy_vs_layers.pdf")
print("Saved: test_accuracy_vs_layers.png")

# ── Plot 2: Avg Time per Epoch ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(layers_unf,  time_unf,  marker="o", linewidth=2, markersize=7,
         color=BLUE, alpha=ALPHA_UNF, linestyle="--", label="Unfiltered training set")
ax2.plot(layers_main, time_main, marker="o", linewidth=2, markersize=7,
         color=BLUE, label="Filtered training set")

for x, y in zip(layers_unf, time_unf):
    ax2.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                 xytext=(0, -15), ha="center", fontsize=8.5, color=BLUE, alpha=0.4)
for x, y in zip(layers_main, time_main):
    ax2.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8.5, color=BLUE)

ax2.set_xlabel("Number of layers $L$", fontsize=12)
ax2.set_ylabel("Avg. time per epoch (s)", fontsize=12)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(fontsize=11)

fig2.tight_layout()
fig2.savefig("test_accuracy_vs_layers_time.pdf")
print("Saved: test_accuracy_vs_layers_time.png")

plt.show()
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open("../experiments_log.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

def extract_data(records, experiment_name, num_params, dataset_size=500):
    filtered = {}
    for r in records:
        if r["seed"] != 343:
            continue
        if r["experiment_name"] != experiment_name:
            continue
        if r["dataset_size"] != dataset_size:
            continue
        if r["num_params_per_layer"] != num_params:
            continue
        filtered[r["num_layers"]] = r
    data = sorted(filtered.values(), key=lambda r: r["num_layers"])
    return [r["num_layers"] for r in data], [r["final_test_acc"] for r in data], [r["avg_time_per_epoch_sec"] for r in data], [r["novel_test_acc"] for r in data]

layers_1, acc_1, time_1, novel_1 = extract_data(records, "sec2", num_params=1)
layers_2, acc_2, time_2, novel_2 = extract_data(records, "sec2", num_params=2)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(8, 5))

ax.plot(layers_1, acc_1, marker="s", linewidth=2, markersize=7,
        color="#DC2626", label="1 param / layer")
ax.plot(layers_2, acc_2, marker="o", linewidth=2, markersize=7,
        color="#2563EB", label="2 params / layer")

for x, y in zip(layers_1, acc_1):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, -15), ha="center", fontsize=8.5, color="#DC2626")
for x, y in zip(layers_2, acc_2):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.5, color="#2563EB")

ax.set_xlabel("Number of layers $L$", fontsize=12)
ax.set_ylabel("Final test accuracy", fontsize=12)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim(0.4, 1.02)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

fig1.tight_layout()
fig1.savefig("sec2_acc.pdf")
print("Saved: sec2_acc.pdf")

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(layers_1, time_1, marker="s", linewidth=2, markersize=7,
        color="#DC2626", label="1 param / layer")
ax2.plot(layers_2, time_2, marker="o", linewidth=2, markersize=7,
        color="#2563EB", label="2 params / layer")

for x, y in zip(layers_1, time_1):
    ax2.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                xytext=(0, -15), ha="center", fontsize=8.5, color="#DC2626")
for x, y in zip(layers_2, time_2):
    ax2.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.5, color="#2563EB")

ax2.set_xlabel("Number of layers $L$", fontsize=12)
ax2.set_ylabel("Avg. time per epoch (s)", fontsize=12)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(fontsize=11)

fig2.tight_layout()
fig2.savefig("sec2_time.pdf")
print("Saved: sec2_time.pdf")

fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.plot(layers_1, novel_1, marker="s", linewidth=2, markersize=7,
        color="#DC2626", label="1 param / layer")
ax3.plot(layers_2, novel_2, marker="o", linewidth=2, markersize=7,
        color="#2563EB", label="2 params / layer")

for x, y in zip(layers_1, novel_1):
    ax3.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, -15), ha="center", fontsize=8.5, color="#DC2626")
for x, y in zip(layers_2, novel_2):
    ax3.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.5, color="#2563EB")

ax3.set_xlabel("Number of layers $L$", fontsize=12)
ax3.set_ylabel("Novel test accuracy", fontsize=12)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax3.grid(axis="y", linestyle="--", alpha=0.5)
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_ylim(0.4, 1.02)
ax3.legend(fontsize=11)

fig3.tight_layout()
fig3.savefig("sec2_novel.pdf")
print("Saved: sec2_novel.pdf")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 1. Load the data
records = []
with open("experiments_log.jsonl", "r") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)

# Filter for the correct experiment
df = df[df['experiment_name'] == 'sec2'].copy()

# Map the numerical parameter count to a categorical label for cleaner legend
df['Ansatz Type'] = df['num_params_per_layer'].map({
    1: '1 parameter / layer',
    2: '2 parameters / layer'
})

mask = (df['num_layers'] == 2) & (df['num_params_per_layer'] == 2)
subset = df[mask]['final_test_acc']

print(f"Mean:  {subset.mean():.4f}")
print(f"Std:   {subset.std():.4f}")

# ==========================================
# PLOT 1: Standard Test Accuracy
# ==========================================
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df,
    x='num_layers',
    y='final_test_acc',
    hue='Ansatz Type',
    marker='o',
    errorbar='ci' # 'ci' for confidence interval
)
plt.xlabel('Number of re-uploading layers', fontsize=12)
plt.ylabel('Final test accuracy', fontsize=12)
plt.ylim(0.4, 0.9)
plt.xticks(sorted(df['num_layers'].unique()))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Variational ansatz')
plt.tight_layout()
plt.savefig("plots/layers_params/test_acc.pdf")
plt.show()

# ==========================================
# PLOT 2: Novel Graphs Test Accuracy
# ==========================================
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df,
    x='num_layers',
    y='novel_test_acc',
    hue='Ansatz Type',
    marker='s', # Use square markers to distinguish from the other plot
    errorbar='ci'
)
plt.xlabel('Number of re-uploading layers', fontsize=12)
plt.ylabel('Final novel test accuracy', fontsize=12)
plt.ylim(0.4, 0.9)
plt.xticks(sorted(df['num_layers'].unique()))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Variational ansatz')
plt.tight_layout()
plt.savefig("plots/layers_params/novel_test_acc.pdf")
plt.show()

# ==========================================
# PLOT 3: Average Time per Epoch
# ==========================================
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df,
    x='num_layers',
    y='avg_time_per_epoch_sec',
    hue='Ansatz Type',
    marker='s', # Use square markers to distinguish from the other plot
    errorbar='ci'
)
plt.xlabel('Number of re-uploading layers', fontsize=12)
plt.ylabel('Avg time per epoch (s)', fontsize=12)
plt.xticks(sorted(df['num_layers'].unique()))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Variational ansatz')
plt.tight_layout()
plt.savefig("plots/layers_params/time.pdf")
plt.show()

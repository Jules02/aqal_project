import json
import numpy as np
from scipy import stats

# Read data
with open("../experiments_log.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

quantum_train_acc   = [r["final_train_acc"] for r in records if r["experiment_name"] == "parametrized_filtereddataset"]
quantum_test_acc   = [r["final_test_acc"] for r in records if r["experiment_name"] == "parametrized_filtereddataset"]
quantum_total_time  = [r["total_training_time_sec"] for r in records if r["experiment_name"] == "parametrized_filtereddataset"]
classical_train_acc = [r["final_train_acc"] for r in records if r["experiment_name"] == "classical"]
classical_test_acc = [r["final_test_acc"] for r in records if r["experiment_name"] == "classical"]
classical_total_time = [r["total_training_time_sec"] for r in records if r["experiment_name"] == "classical"]

X_train_acc = np.array(quantum_train_acc)
Y_train_acc = np.array(classical_train_acc)
print(f"Mean quantum train acc   : {X_train_acc.mean():.4f} ± {X_train_acc.std():.4f}")
print(f"Mean classical train acc : {Y_train_acc.mean():.4f} ± {Y_train_acc.std():.4f}")
print()

X_test_acc = np.array(quantum_test_acc)
Y_test_acc = np.array(classical_test_acc)
print(f"Mean quantum train acc   : {X_test_acc.mean():.4f} ± {X_test_acc.std():.4f}")
print(f"Mean classical train acc : {Y_test_acc.mean():.4f} ± {Y_test_acc.std():.4f}")
print()

X_total_time = np.array(quantum_total_time)
Y_total_time = np.array(classical_total_time)
print(f"Mean quantum total_time   : {X_total_time.mean():.4f} ± {X_total_time.std():.4f}")
print(f"Mean classical total_time : {Y_total_time.mean():.4f} ± {Y_total_time.std():.4f}")
